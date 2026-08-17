"""Generate one v2 split from its immutable recipe.

One generator, three recipes. The split-specific parts are data, so the funnel,
the parity guarantees and the stopping rule cannot drift between splits.

Three properties are enforced rather than assumed, each because its absence has
already cost a run in this work:

*Resume must be byte-identical.* The RNG is replayed from index zero and
completed renderings are skipped, never seeded afresh. Measurement rows carry no
run-local fields -- a resume counter inside a row would make parity unachievable
by construction.

*A resume must not silently continue a different experiment.* The recipe, the
model files and the font files are hashed into the manifest on the first write,
and a resume that disagrees on any of them is refused rather than appended to.

*Stopping must not depend on inspecting results mid-flight.* Quotas are checked
only at rendering boundaries; when they are all met the current rendering is
completed and generation ends. Hitting the maximum with a quota short is a
failure with a non-zero exit, not something to be papered over by generating
more.
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
from mine_line_triggers_v1 import build_engine, collapse_ctc, load_labels  # noqa: E402
from ocr_roi_validator.diagnostic_runner import (  # noqa: E402
    CheckpointWriter, atomic_write_json, load_checkpoint, log_line,
    resumable_units,
)
from ocr_roi_validator.glyph_geometry import measure_line_geometry  # noqa: E402
from ocr_roi_validator.terminal_reason import (  # noqa: E402
    derive_flags, summarise_terminal_reasons,
)
from ocr_roi_validator.v2_recipes import (  # noqa: E402
    MACRO_STRATA, PERTURBATION, STRATUM_TARGETS, V2_RECIPES,
    assert_cohort_independence,
)

ARTIFACT_FORMAT = "line_v2_jsonl"
ARTIFACT_VERSION = 2
CHECKPOINT_FLUSH_EVERY = 100
UNKNOWN_REASONS = {"DELETION", "INSERTION", "MULTIPLE_CHANGES",
                   "CHANGE_ELSEWHERE", "OTHER_SUBSTITUTION", "ACCENT_LOST"}


class Rendering:
    """One rendering, fully determined by the recipe and its index."""

    __slots__ = ("index", "stratum", "font", "size", "upscale", "word_index",
                 "accented", "template_index", "pad_x", "pad_y", "background",
                 "foreground", "contrast", "blur", "jitter_x", "jitter_y",
                 "resample", "recipe")

    def __init__(self, index: int, recipe) -> None:
        self.index = index
        self.recipe = recipe
        # Strata rotate so every prefix of the run is balanced across them;
        # stopping early therefore cannot skew the stratum mix.
        self.stratum = MACRO_STRATA[index % len(MACRO_STRATA)]
        self.font = recipe.fonts[(index // len(MACRO_STRATA)) % len(recipe.fonts)]

        target = STRATUM_TARGETS[self.stratum]
        rng = random.Random(recipe.seed + index)
        self.size = target["sizes"][index % len(target["sizes"])]
        self.upscale = round(rng.uniform(*target["upscale"]), 4)
        self.word_index = rng.randrange(len(recipe.words))
        self.accented = rng.random() < 0.5
        self.template_index = rng.randrange(len(recipe.templates))
        self.pad_x = rng.randint(*PERTURBATION["pad_x"])
        self.pad_y = PERTURBATION["pad_y"]
        dark = rng.random() < PERTURBATION["dark_background_probability"]
        self.background = (rng.randint(*PERTURBATION["dark_background"]) if dark
                           else rng.randint(*PERTURBATION["light_background"]))
        self.foreground = (rng.randint(*PERTURBATION["dark_foreground"]) if dark
                           else rng.randint(*PERTURBATION["light_foreground"]))
        self.contrast = round(rng.uniform(*PERTURBATION["contrast"]), 4)
        self.blur = round(rng.uniform(*PERTURBATION["blur"]), 4)
        self.jitter_x = round(rng.uniform(*PERTURBATION["jitter"]), 4)
        self.jitter_y = round(rng.uniform(*PERTURBATION["jitter"]), 4)
        self.resample = rng.choice(PERTURBATION["resample"])

    @property
    def template(self) -> str:
        return self.recipe.templates[self.template_index]

    @property
    def text(self) -> str:
        plain, accented = self.recipe.words[self.word_index]
        return self.template.format(accented if self.accented else plain)

    @property
    def target_character(self) -> str:
        return "é" if self.accented else "e"

    def target_position(self) -> int:
        """Index of the target, from the drawn string alone, before any OCR."""
        text = unicodedata.normalize("NFC", self.text)
        offset = self.template.index("{}")
        for position in range(offset, len(text)):
            if text[position] in {"e", "é"}:
                return position
        return -1


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
    """Render, detect, decode and classify one occurrence.

    A row is written whatever happens, including a detector miss, so the
    denominator survives -- v1 recorded only rows where an é was emitted and
    could not produce a rate at all.
    """
    drawn = unicodedata.normalize("NFC", rendering.text)
    position = rendering.target_position()
    row = {
        "index": rendering.index,
        "row_digest": hashlib.sha256(
            f"{rendering.recipe.name}|{rendering.index}".encode()).hexdigest(),
        "split": rendering.recipe.name,
        "target_stratum": rendering.stratum,
        "font": rendering.font,
        "nominal_size": rendering.size,
        "upscale": rendering.upscale,
        "pad_x": rendering.pad_x,
        "word_index": rendering.word_index,
        "template_index": rendering.template_index,
        "visual_target": rendering.target_character,
        "target_position": position,
        "terminal_reason": "NOT_EVALUATED",
        "page_sha256": None, "crop_sha256": None,
        "runtime_ink_height": None, "measured_stratum": None,
        "crop_width": None, "crop_height": None,
        "horizontal_padding_ratio": None, "clipped": None,
        "decoded_text": None, "expected_substring": None,
        "decoded_length": None, "detector_box_count": None,
        "clean_hallucination": False, "clean_preservation": False,
    }
    if position < 0:
        row["terminal_reason"] = "NOT_ELIGIBLE_NO_TARGET"
        return row

    try:
        page = render(rendering, font_dir)
        bgr = np.asarray(page)[:, :, ::-1].copy()
        row["page_sha256"] = hashlib.sha256(np.asarray(page).tobytes()).hexdigest()
        boxes, _ = engine.auto_text_det(bgr)
    except Exception as error:                        # pragma: no cover
        row["terminal_reason"] = "RENDER_ERROR"
        row["reason"] = type(error).__name__
        return row
    if boxes is None or len(boxes) == 0:
        row["terminal_reason"] = "DETECTOR_MISS"
        return row

    crops = engine.get_crop_img_list(bgr, boxes)
    row["detector_box_count"] = len(crops)

    # The target's box is chosen from the renderer's own layout. Choosing it
    # from the decode would let the recognizer's output select the sample.
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
        row["terminal_reason"] = "WRONG_LINE_SELECTED"
        return row

    crop = crops[chosen]
    crop_height, crop_width = crop.shape[:2]
    row["crop_width"], row["crop_height"] = int(crop_width), int(crop_height)
    row["crop_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
    geometry = measure_line_geometry(crop, rec_height)
    if geometry is not None:
        row["runtime_ink_height"] = geometry.ink_height
        row["measured_stratum"] = macro_stratum(geometry.ink_height)
        row["horizontal_padding_ratio"] = round(
            geometry.horizontal_padding_ratio, 6)
        row["clipped"] = bool(geometry.ink_top <= 0
                              or geometry.ink_bottom >= crop_height - 1)

    max_wh_ratio = max(rec_width / rec_height, crop_width / float(crop_height))
    try:
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_width / float(crop_height)],
            max_wh_ratio=max_wh_ratio)[0][0]
    except Exception as error:                        # pragma: no cover
        row["terminal_reason"] = "RECOGNIZER_FAILURE"
        row["reason"] = type(error).__name__
        return row
    row["decoded_text"] = decoded
    emitted = collapse_ctc(logits[0].argmax(axis=-1).tolist(), labels)
    if "".join(item["char"] for item in emitted) != decoded:
        row["terminal_reason"] = "ALIGNMENT_AMBIGUITY"
        return row

    # A split line means the crop holds only part of the text, so the expected
    # string is recomputed for the chosen box -- otherwise every split scores
    # as a deletion.
    xs = [point[0] for point in boxes[chosen]]
    low, high = min(xs), max(xs)
    kept, target_index = [], -1
    for order, character in enumerate(drawn):
        offset = draw.textlength(drawn[:order], font=font)
        width = draw.textlength(character, font=font)
        centre = (origin_x + offset + width / 2.0) * rendering.upscale
        if low <= centre <= high:
            if order == position:
                target_index = len(kept)
            kept.append(character)
    expected = "".join(kept)
    row["expected_substring"] = expected
    if target_index < 0:
        row["terminal_reason"] = "NO_TARGET_TOKEN"
        return row

    normalized = unicodedata.normalize("NFC", decoded)
    row["decoded_length"] = len(normalized)
    if len(normalized) != len(expected):
        row["terminal_reason"] = ("INSERTION" if len(normalized) > len(expected)
                                  else "DELETION")
        return row
    differing = [i for i, (a, b) in enumerate(zip(expected, normalized)) if a != b]
    if not differing:
        if rendering.target_character == "é":
            row["clean_preservation"] = True
            row["terminal_reason"] = "CLEAN_PRESERVATION"
        else:
            row["terminal_reason"] = "CLEAN_CORRECT_BARE_E"
        return row
    if len(differing) > 1:
        row["terminal_reason"] = "MULTIPLE_CHANGES"
        return row
    index = differing[0]
    if index != target_index:
        row["terminal_reason"] = "CHANGE_ELSEWHERE"
        return row
    if expected[index] == "e" and normalized[index] == "é":
        row["clean_hallucination"] = True
        row["terminal_reason"] = "CLEAN_HALLUCINATION"
    elif expected[index] == "é" and normalized[index] == "e":
        row["terminal_reason"] = "ACCENT_LOST"
    else:
        row["terminal_reason"] = "OTHER_SUBSTITUTION"
    return row


def tally(rows) -> dict:
    """Counts in the shape SplitRecipe.quota_state expects."""
    counts = {
        "hallucination_total": 0, "preservation_total": 0, "unknown_total": 0,
        "hallucination_by_stratum": Counter(),
        "preservation_by_stratum": Counter(),
        "preservation_by_font": Counter(),
    }
    for row in rows:
        stratum = row.get("measured_stratum")
        if row.get("clean_hallucination"):
            counts["hallucination_total"] += 1
            if stratum:
                counts["hallucination_by_stratum"][stratum] += 1
        if row.get("clean_preservation"):
            counts["preservation_total"] += 1
            if stratum:
                counts["preservation_by_stratum"][stratum] += 1
            counts["preservation_by_font"][row["font"]] += 1
        if row.get("terminal_reason") in UNKNOWN_REASONS:
            counts["unknown_total"] += 1
    return counts


def environment_fingerprint(package, font_dir: Path, recipe) -> dict:
    """Hashes of everything that would change the meaning of a resume."""
    def digest(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return {
        "recipe_sha256": hashlib.sha256(
            json.dumps(recipe.as_dict(), sort_keys=True,
                       ensure_ascii=False).encode()).hexdigest(),
        "generator_sha256": digest(Path(__file__)),
        "detector_sha256": digest(package.detector_model),
        "dictionary_sha256": digest(package.dictionary),
        "fonts": {name: digest(font_dir / name) for name in recipe.fonts},
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_version": ARTIFACT_VERSION,
    }


class WriterLock:
    """Refuse a second concurrent writer on the same checkpoint."""

    def __init__(self, path: Path) -> None:
        self._path = path.with_suffix(path.suffix + ".lock")

    def __enter__(self) -> "WriterLock":
        try:
            handle = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = self._path.read_text(encoding="utf-8").strip()
            raise SystemExit(
                f"another writer holds {self._path.name} ({owner}); refusing to "
                "open the same checkpoint twice")
        with os.fdopen(handle, "w") as stream:
            stream.write(f"pid={os.getpid()} at={time.time():.0f}")
        return self

    def __exit__(self, *exc_info) -> None:
        self._path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--split", choices=tuple(V2_RECIPES), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int,
                        help="development smoke only; never for real output")
    parser.add_argument("--smoke", action="store_true",
                        help="tag output as development_smoke_only")
    args = parser.parse_args()

    recipe = V2_RECIPES[args.split]
    assert_cohort_independence(V2_RECIPES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape
    fingerprint = environment_fingerprint(package, args.font_dir, recipe)

    recipe_path = args.out_dir / "recipe.json"
    payload = {
        **recipe.as_dict(),
        "environment": fingerprint,
        "quota_rule": ("all quotas must be met simultaneously; the observed "
                       "count includes credited train_v1 rows where declared"),
        "stop_rule": ("stop at the first rendering boundary where every quota "
                      "is met, completing that rendering; reaching "
                      "max_renderings with any quota short exits non-zero"),
        "checkpoint_flush_every": CHECKPOINT_FLUSH_EVERY,
        "development_smoke_only": bool(args.smoke or args.limit),
        "prohibited_inputs": ["Expected text", "product UI captures",
                              "target F0 captures"],
    }
    if not recipe_path.is_file():
        atomic_write_json(recipe_path, payload)
    recipe_digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()

    checkpoint = args.out_dir / "checkpoint.jsonl"
    guard_path = args.out_dir / "environment.json"
    if guard_path.is_file():
        previous = json.loads(guard_path.read_text(encoding="utf-8"))
        differing = [k for k in fingerprint
                     if previous.get(k) != fingerprint[k]]
        if differing:
            print(f"refusing to resume: {differing} changed since the first "
                  "write; a resumed run must be the same experiment",
                  file=sys.stderr)
            return 2
    else:
        atomic_write_json(guard_path, fingerprint)

    state = load_checkpoint(checkpoint, unit_field="index")
    log_line(f"{recipe.name}: recipe {recipe_digest[:16]}… "
             f"max {recipe.max_renderings:,} rows {len(state.rows)} "
             f"(resume #{state.resume_count}) pid {os.getpid()}")

    total = min(args.limit or recipe.max_renderings, recipe.max_renderings)
    rows = list(state.rows)
    started = time.time()
    stop_reason = "MAX_RENDERINGS_REACHED"
    rendered = len(state.completed_units)

    with WriterLock(checkpoint):
        with CheckpointWriter(checkpoint, state.digests,
                              flush_every=CHECKPOINT_FLUSH_EVERY) as writer:
            for index, rendering in resumable_units(
                    total, lambda i: Rendering(i, recipe), state.completed_units):
                row = evaluate(rendering, args.font_dir, engine, recognizer,
                               labels, rec_height, rec_width)
                row["diagnostic_flags"] = derive_flags(row)
                writer.append(row)
                rows.append(row)
                rendered += 1

                # Quotas are only consulted at a rendering boundary, and the
                # current rendering is always completed first.
                if recipe.quotas_met(tally(rows)):
                    stop_reason = "QUOTA_MET"
                    log_line(f"  all quotas met at rendering {index + 1}")
                    break
                if (index + 1) % args.progress_every == 0:
                    writer.sync()
                    counts = tally(rows)
                    log_line(f"  {index + 1}/{total} "
                             f"h={counts['hallucination_total']} "
                             f"p={counts['preservation_total']} "
                             f"u={counts['unknown_total']} "
                             f"{time.time() - started:.0f}s")
        duplicates = writer.duplicates_rejected

    counts = tally(rows)
    state_report = recipe.quota_state(counts)
    met = all(entry["met"] for entry in state_report.values())
    exposure = defaultdict(lambda: Counter())
    for row in rows:
        exposure[row["font"]][row["target_stratum"]] += 1

    manifest = {
        "split": recipe.name, "role": recipe.role,
        "recipe_sha256": recipe_digest,
        "environment": fingerprint,
        "development_smoke_only": bool(args.smoke or args.limit),
        "renderings": rendered, "rows": len(rows),
        "max_renderings": recipe.max_renderings,
        "stop_reason": stop_reason if met else "QUOTA_NOT_MET",
        "quota_state": state_report, "quotas_met": met,
        "counts": {
            "hallucination_total": counts["hallucination_total"],
            "preservation_total": counts["preservation_total"],
            "unknown_total": counts["unknown_total"],
            "hallucination_by_stratum": dict(counts["hallucination_by_stratum"]),
            "preservation_by_stratum": dict(counts["preservation_by_stratum"]),
            "preservation_by_font": dict(counts["preservation_by_font"]),
        },
        "font_stratum_exposure": {f: dict(v) for f, v in exposure.items()},
        "terminal_reasons": summarise_terminal_reasons(rows),
        "duplicate_digests_rejected": duplicates,
        "unique_digests": len({r["row_digest"] for r in rows}),
        "resume_count": state.resume_count,
        "wall_time_seconds": round(time.time() - started, 1),
    }
    digest = atomic_write_json(args.out_dir / "manifest.json", manifest)

    log_line(f"\n{recipe.name}: {rendered:,} renderings, stop={manifest['stop_reason']}")
    for name, entry in state_report.items():
        mark = "ok " if entry["met"] else "MISS"
        log_line(f"  {mark} {name:34s} {entry['observed']:6d} / "
                 f"{entry['required']}")
    log_line(f"  manifest sha256 {digest}")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
