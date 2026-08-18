"""Generate counterfactual pair data for the line verifier.

Each pair is the same word, font, size and optical setting rendered twice --
once with a bare e at the queried position, once with an accent. The label comes
from which glyph the renderer drew, never from what the recognizer said.

The leakage this design exists to prevent has already occurred once here:
cropping both members with geometry measured from the accented one puts the
answer into the bare image. So each member is rendered and cropped from its own
page independently, and the two never consult each other. Their shared
``pair_id`` supports auditing afterwards; it is not an input.

UNKNOWN cases are constructed rather than collected, because the model must
learn to decline when the query cannot be answered: the ordinal is shifted off
the drawn target, the token count is made to disagree, or the target position
fails to yield a clean token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from erratum_interaction_v1 import macro_stratum  # noqa: E402
from generate_v2_split import WriterLock  # noqa: E402
from mine_line_triggers_v1 import build_engine, collapse_ctc, load_labels  # noqa: E402
from ocr_roi_validator.counterfactual_recipes import (  # noqa: E402
    COUNTERFACTUAL_RECIPES, MEMBERS_PER_PAIR, UNKNOWN_KINDS,
    assert_context_independence,
)
from ocr_roi_validator.diagnostic_runner import (  # noqa: E402
    CheckpointWriter, atomic_write_json, load_checkpoint, log_line,
    resumable_units,
)
from ocr_roi_validator.glyph_geometry import measure_line_geometry  # noqa: E402
from ocr_roi_validator.v2_recipes import MACRO_STRATA, STRATUM_TARGETS  # noqa: E402

TEMPLATES = ("{}", "{} zq", "Xk {}", "{} 4,7")
CLASS_ACCENT, CLASS_BARE, CLASS_UNKNOWN = "ACCENT_PRESENT", "BARE_E", "UNKNOWN"


class Rendering:
    """One rendering, determined by the recipe and its index.

    A pair's two members share every setting except which glyph is drawn, and
    each is rendered from its own page: nothing about one member's geometry is
    available while building the other.
    """

    __slots__ = ("index", "recipe", "context_index", "slot", "word_bare",
                 "word_accented", "kind", "pair_id", "accented", "font",
                 "stratum", "size", "upscale", "template", "pad_x", "pad_y",
                 "background", "foreground", "contrast", "blur", "jitter_x",
                 "jitter_y", "resample", "unknown_kind")

    def __init__(self, index: int, recipe) -> None:
        self.index = index
        self.recipe = recipe
        per_context = recipe.renderings_per_context
        self.context_index, self.slot = divmod(index, per_context)
        self.word_bare, self.word_accented = recipe.words[self.context_index]

        member_slots = recipe.pairs_per_context * MEMBERS_PER_PAIR
        if self.slot < member_slots:
            self.kind = "PAIR_MEMBER"
            pair_number, member = divmod(self.slot, MEMBERS_PER_PAIR)
            self.pair_id = f"{recipe.name}|{self.context_index}|{pair_number}"
            self.accented = member == 1
            self.unknown_kind = None
            settings_slot = pair_number
        else:
            self.kind = "UNKNOWN_CASE"
            offset = self.slot - member_slots
            kinds = [name for name, count in UNKNOWN_KINDS.items()
                     for _ in range(count)]
            self.unknown_kind = kinds[offset % len(kinds)]
            self.pair_id = None
            self.accented = offset % 2 == 1
            settings_slot = recipe.pairs_per_context + offset

        # Optical settings depend on the pair, not the member, so both members
        # share them exactly.
        rng = random.Random(recipe.seed + self.context_index * 1000 + settings_slot)
        self.font = recipe.fonts[settings_slot % len(recipe.fonts)]
        self.stratum = MACRO_STRATA[settings_slot % len(MACRO_STRATA)]
        self.template = TEMPLATES[settings_slot % len(TEMPLATES)]
        target = STRATUM_TARGETS[self.stratum]
        self.size = target["sizes"][settings_slot % len(target["sizes"])]
        self.upscale = round(rng.uniform(*target["upscale"]), 4)
        self.pad_x = rng.randint(12, 30)
        self.pad_y = 20
        dark = rng.random() < 0.75
        self.background = rng.randint(8, 34) if dark else rng.randint(226, 250)
        self.foreground = rng.randint(220, 252) if dark else rng.randint(8, 40)
        self.contrast = round(rng.uniform(0.85, 1.20), 4)
        self.blur = round(rng.uniform(0.0, 0.5), 4)
        self.jitter_x = round(rng.uniform(-0.5, 0.5), 4)
        self.jitter_y = round(rng.uniform(-0.5, 0.5), 4)
        self.resample = rng.choice(["bicubic", "lanczos"])

    @property
    def word(self) -> str:
        return self.word_accented if self.accented else self.word_bare

    @property
    def text(self) -> str:
        return self.template.format(self.word)

    @property
    def label(self) -> str:
        """From the renderer's glyph choice alone."""
        if self.kind == "UNKNOWN_CASE":
            return CLASS_UNKNOWN
        return CLASS_ACCENT if self.accented else CLASS_BARE

    def target_position(self) -> int:
        offset = self.template.index("{}")
        return offset + self.word_bare.index("e")


def render(rendering: Rendering, font_dir: Path) -> Image.Image:
    font = ImageFont.truetype(str(font_dir / rendering.font), rendering.size)
    origin_x = rendering.pad_x + rendering.jitter_x
    origin_y = rendering.pad_y + rendering.jitter_y
    box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox(
        (origin_x, origin_y), rendering.text, font=font)
    image = Image.new("RGB", (int(box[2] + rendering.pad_x),
                              int(box[3] + rendering.pad_y)),
                      (rendering.background,) * 3)
    ImageDraw.Draw(image).text((origin_x, origin_y), rendering.text, font=font,
                               fill=(rendering.foreground,) * 3)
    if rendering.blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(rendering.blur))
    if rendering.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(rendering.contrast)
    if rendering.upscale != 1.0:
        resample = (Image.Resampling.BICUBIC if rendering.resample == "bicubic"
                    else Image.Resampling.LANCZOS)
        image = image.resize((int(image.width * rendering.upscale),
                              int(image.height * rendering.upscale)), resample)
    return image


def evaluate(rendering, font_dir, engine, recognizer, labels, rec_height,
             rec_width) -> dict:
    drawn = unicodedata.normalize("NFC", rendering.text)
    position = rendering.target_position()
    row = {
        "index": rendering.index,
        "row_digest": hashlib.sha256(
            f"{rendering.recipe.name}|{rendering.index}".encode()).hexdigest(),
        "split": rendering.recipe.name,
        "context_index": rendering.context_index,
        "word_bare": rendering.word_bare,
        "kind": rendering.kind,
        "pair_id": rendering.pair_id,
        "unknown_kind": rendering.unknown_kind,
        "label": rendering.label,
        "visual_target": "é" if rendering.accented else "e",
        "target_position": position,
        "font": rendering.font, "nominal_size": rendering.size,
        "upscale": rendering.upscale, "target_stratum": rendering.stratum,
        "template": rendering.template,
        "page_sha256": None, "crop_sha256": None,
        "crop_width": None, "crop_height": None,
        "runtime_ink_height": None, "measured_stratum": None,
        "baseline_token_ordinal": None, "baseline_token_count": None,
        "query_ordinal": None, "query_token_count": None,
        "usable": False, "reason": None,
    }
    try:
        page = render(rendering, font_dir)
        bgr = np.asarray(page)[:, :, ::-1].copy()
        row["page_sha256"] = hashlib.sha256(np.asarray(page).tobytes()).hexdigest()
        boxes, _ = engine.auto_text_det(bgr)
    except Exception as error:                        # pragma: no cover
        row["reason"] = f"RENDER_ERROR:{type(error).__name__}"
        return row
    if boxes is None or len(boxes) == 0:
        row["reason"] = "DETECTOR_MISS"
        return row

    crops = engine.get_crop_img_list(bgr, boxes)
    font = ImageFont.truetype(str(font_dir / rendering.font), rendering.size)
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    origin_x = rendering.pad_x + rendering.jitter_x
    before = draw.textlength(drawn[:position], font=font)
    glyph = draw.textlength(drawn[position], font=font)
    target_x = (origin_x + before + glyph / 2.0) * rendering.upscale
    chosen = None
    for order, box in enumerate(boxes):
        xs = [point[0] for point in box]
        if min(xs) <= target_x <= max(xs):
            chosen = order
            break
    if chosen is None:
        row["reason"] = "TARGET_NOT_IN_ANY_BOX"
        return row

    crop = crops[chosen]
    crop_height, crop_width = crop.shape[:2]
    row["crop_width"], row["crop_height"] = int(crop_width), int(crop_height)
    row["crop_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
    geometry = measure_line_geometry(crop, rec_height)
    if geometry is not None:
        row["runtime_ink_height"] = geometry.ink_height
        row["measured_stratum"] = macro_stratum(geometry.ink_height)

    max_wh_ratio = max(rec_width / rec_height, crop_width / float(crop_height))
    try:
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_width / float(crop_height)],
            max_wh_ratio=max_wh_ratio)[0][0]
    except Exception as error:                        # pragma: no cover
        row["reason"] = f"RECOGNIZER_FAILURE:{type(error).__name__}"
        return row
    emitted = collapse_ctc(logits[0].argmax(axis=-1).tolist(), labels)
    if "".join(item["char"] for item in emitted) != decoded:
        row["reason"] = "ALIGNMENT_AMBIGUITY"
        return row

    xs = [point[0] for point in boxes[chosen]]
    low, high = min(xs), max(xs)
    ordinal = -1
    kept = 0
    for order, character in enumerate(drawn):
        offset = draw.textlength(drawn[:order], font=font)
        width = draw.textlength(character, font=font)
        centre = (origin_x + offset + width / 2.0) * rendering.upscale
        if low <= centre <= high:
            if order == position:
                ordinal = kept
            kept += 1
    if ordinal < 0:
        row["reason"] = "TARGET_OUTSIDE_CHOSEN_BOX"
        return row

    row["baseline_token_ordinal"] = ordinal
    row["baseline_token_count"] = len(emitted)

    # The query is a position and a count -- never a character, never text.
    if rendering.kind == "UNKNOWN_CASE":
        kind = rendering.unknown_kind
        if kind == "ORDINAL_SHIFTED":
            shifted = (ordinal + 2) % max(1, len(emitted))
            if shifted == ordinal:
                row["reason"] = "SHIFT_DEGENERATE"
                return row
            row["query_ordinal"] = shifted
            row["query_token_count"] = len(emitted)
        elif kind == "TOKEN_COUNT_MISMATCH":
            row["query_ordinal"] = ordinal
            row["query_token_count"] = len(emitted) + 3
        else:                                        # ORDINAL_OUT_OF_RANGE
            # Constructed rather than waited for: naming a position past the
            # end of the decode is always possible, whereas a target that
            # fails to decode almost never occurs.
            row["query_ordinal"] = len(emitted) + 2
            row["query_token_count"] = len(emitted)
    else:
        if ordinal >= len(emitted):
            row["reason"] = "TARGET_ORDINAL_BEYOND_TOKENS"
            return row
        row["query_ordinal"] = ordinal
        row["query_token_count"] = len(emitted)

    row["usable"] = True
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split", choices=tuple(COUNTERFACTUAL_RECIPES),
                        required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    assert_context_independence()
    recipe = COUNTERFACTUAL_RECIPES[args.split]
    total = min(args.limit or recipe.renderings, recipe.renderings)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    fingerprint = {
        "recipe_sha256": hashlib.sha256(
            json.dumps(recipe.as_dict(), sort_keys=True,
                       ensure_ascii=False).encode()).hexdigest(),
        "generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "detector_sha256": hashlib.sha256(
            package.detector_model.read_bytes()).hexdigest(),
        "dictionary_sha256": hashlib.sha256(
            package.dictionary.read_bytes()).hexdigest(),
    }
    payload = {
        **recipe.as_dict(), "environment": fingerprint,
        "label_source": "renderer glyph choice only; never the decoded text",
        "query_contract": "ordinal and token count only; no character, no text",
        "pair_isolation": ("each member is rendered and cropped from its own "
                           "page; no counterpart geometry is consulted"),
        "prohibited_inputs": ["Expected text", "decoded string", "dictionary",
                              "file name", "product UI captures"],
    }
    recipe_path = args.out_dir / "recipe.json"
    if not recipe_path.is_file():
        atomic_write_json(recipe_path, payload)
    recipe_digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()

    guard_path = args.out_dir / "environment.json"
    if guard_path.is_file():
        previous = json.loads(guard_path.read_text(encoding="utf-8"))
        differing = [k for k in fingerprint if previous.get(k) != fingerprint[k]]
        if differing:
            print(f"refusing to resume: {differing} changed", file=sys.stderr)
            return 2
    else:
        atomic_write_json(guard_path, fingerprint)

    checkpoint = args.out_dir / "checkpoint.jsonl"
    state = load_checkpoint(checkpoint, unit_field="index")
    log_line(f"{recipe.name}: {total:,} renderings, have {len(state.rows)} "
             f"(resume #{state.resume_count}) pid {os.getpid()}")

    rows = list(state.rows)
    started = time.time()
    with WriterLock(checkpoint):
        with CheckpointWriter(checkpoint, state.digests, flush_every=100) as writer:
            for index, rendering in resumable_units(
                    total, lambda i: Rendering(i, recipe),
                    state.completed_units):
                row = evaluate(rendering, args.font_dir, engine, recognizer,
                               labels, rec_height, rec_width)
                writer.append(row)
                rows.append(row)
                if (index + 1) % args.progress_every == 0:
                    writer.sync()
                    usable = sum(1 for r in rows if r["usable"])
                    log_line(f"  {index + 1}/{total} usable={usable} "
                             f"{time.time() - started:.0f}s")
        duplicates = writer.duplicates_rejected

    by_label = Counter(r["label"] for r in rows if r["usable"])
    by_unknown = Counter(r["unknown_kind"] for r in rows
                         if r["usable"] and r["unknown_kind"])
    unusable = Counter(r["reason"] for r in rows if not r["usable"])
    pairs = defaultdict(list)
    for row in rows:
        if row["pair_id"]:
            pairs[row["pair_id"]].append(row)
    complete_pairs = sum(1 for members in pairs.values()
                         if len(members) == MEMBERS_PER_PAIR
                         and all(m["usable"] for m in members))
    page_digests = [r["page_sha256"] for r in rows if r["page_sha256"]]

    manifest = {
        "split": recipe.name, "role": recipe.role,
        "recipe_sha256": recipe_digest, "environment": fingerprint,
        "renderings": len(rows),
        "declared_renderings": recipe.renderings,
        "composition": recipe.as_dict()["composition"],
        "usable": sum(1 for r in rows if r["usable"]),
        "class_counts": dict(by_label),
        "unknown_kind_counts": dict(by_unknown),
        "unusable_reasons": dict(unusable),
        "pairs_declared": recipe.word_context_count * recipe.pairs_per_context,
        "pairs_seen": len(pairs),
        "pairs_complete_and_usable": complete_pairs,
        "unique_row_digests": len({r["row_digest"] for r in rows}),
        "unique_page_digests": len(set(page_digests)),
        "page_digest_collisions": len(page_digests) - len(set(page_digests)),
        "duplicate_digests_rejected": duplicates,
        "resume_count": state.resume_count,
        "wall_time_seconds": round(time.time() - started, 1),
    }
    digest = atomic_write_json(args.out_dir / "manifest.json", manifest)

    log_line(f"\n{recipe.name}: {len(rows):,} rows, usable {manifest['usable']:,}")
    log_line(f"  classes {dict(by_label)}")
    log_line(f"  unknown kinds {dict(by_unknown)}")
    log_line(f"  complete pairs {complete_pairs:,}/{manifest['pairs_declared']:,}")
    log_line(f"  page digest collisions {manifest['page_digest_collisions']}")
    if unusable:
        log_line(f"  unusable {dict(unusable)}")
    log_line(f"  manifest sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
