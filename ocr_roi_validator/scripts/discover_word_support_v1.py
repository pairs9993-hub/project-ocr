"""Measure which word contexts the baseline's e -> é trigger appears in.

Three cohorts have been assembled without knowing whether their words provoke
the defect. Calibration produced zero events in 39,189 renderings across three
words, which no budget increase can fix: a word with no support has no rate to
scale. So support is measured first, across 150 words and 60,000 renderings,
before any further split is designed.

Every word gets exactly 400 renderings whether or not events appear early.
Stopping a word once it "has enough" would make its exposure depend on its
outcome and bias every rate computed afterwards.

Context is recorded per occurrence -- the neighbouring character classes, the
target's position, the word's length, the measured geometry -- so the report can
distinguish a word-identity effect from a context effect. This stage does not
claim which it is.

Output is development_word_support_diagnostic_only. No row may enter a training,
calibration or preflight quota.
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
from ocr_roi_validator.diagnostic_runner import (  # noqa: E402
    CheckpointWriter, atomic_write_json, load_checkpoint, log_line,
    resumable_units,
)
from ocr_roi_validator.glyph_geometry import measure_line_geometry  # noqa: E402
from ocr_roi_validator.terminal_reason import derive_flags  # noqa: E402
from ocr_roi_validator.v2_recipes import (  # noqa: E402
    MACRO_STRATA, STRATUM_TARGETS, V2_RECIPES,
)
from ocr_roi_validator.word_candidates import (  # noqa: E402
    CANDIDATES, DISCOVERY_MAX_RENDERINGS, RENDERINGS_PER_WORD,
    assert_no_prior_overlap, context_of,
)

DISCOVERY_SEED = 40100000
DISCOVERY_FONTS = (
    "arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf", "corbel.ttf",
    "Candara.ttf", "trebuc.ttf", "georgia.ttf", "palab.ttf", "times.ttf",
    "framd.ttf", "tahoma.ttf", "consola.ttf",
)
# Templates are bare frames so the word under test dominates the line.
DISCOVERY_TEMPLATES = ("{}", "{} zq", "Xk {}", "{} 4,7")

# Support classification, fixed before any inference.
ROBUST_MIN_EVENTS = 5
ROBUST_MIN_BREADTH = 2


class DiscoveryCase:
    """One rendering, determined by its index alone."""

    __slots__ = ("index", "candidate", "font", "stratum", "size", "upscale",
                 "accented", "template", "pad_x", "pad_y", "background",
                 "foreground", "contrast", "blur", "jitter_x", "jitter_y",
                 "resample")

    def __init__(self, index: int) -> None:
        self.index = index
        word_index, within = divmod(index, RENDERINGS_PER_WORD)
        self.candidate = CANDIDATES[word_index]
        # Rotate font, stratum and template so each word's 400 renderings are
        # balanced across them regardless of where the word sits in the list.
        self.font = DISCOVERY_FONTS[within % len(DISCOVERY_FONTS)]
        self.stratum = MACRO_STRATA[within % len(MACRO_STRATA)]
        self.template = DISCOVERY_TEMPLATES[within % len(DISCOVERY_TEMPLATES)]
        # Half the renderings draw the accented form, giving preservation
        # exposure alongside the bare-e denominator.
        self.accented = (within % 2 == 1)

        target = STRATUM_TARGETS[self.stratum]
        rng = random.Random(DISCOVERY_SEED + index)
        self.size = target["sizes"][within % len(target["sizes"])]
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
        return (self.candidate.accented if self.accented
                else self.candidate.bare)

    @property
    def text(self) -> str:
        return self.template.format(self.word)

    @property
    def target_character(self) -> str:
        return "é" if self.accented else "e"

    def target_position(self) -> int:
        """Position of the queried e, from the drawn string only."""
        offset = self.template.index("{}")
        return offset + self.candidate.target_index()


def render(case: DiscoveryCase, font_dir: Path) -> Image.Image:
    font = ImageFont.truetype(str(font_dir / case.font), case.size)
    origin_x = case.pad_x + case.jitter_x
    origin_y = case.pad_y + case.jitter_y
    box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox(
        (origin_x, origin_y), case.text, font=font)
    image = Image.new("RGB", (int(box[2] + case.pad_x), int(box[3] + case.pad_y)),
                      (case.background,) * 3)
    ImageDraw.Draw(image).text((origin_x, origin_y), case.text, font=font,
                               fill=(case.foreground,) * 3)
    if case.blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(case.blur))
    if case.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(case.contrast)
    if case.upscale != 1.0:
        resample = (Image.Resampling.BICUBIC if case.resample == "bicubic"
                    else Image.Resampling.LANCZOS)
        image = image.resize((int(image.width * case.upscale),
                              int(image.height * case.upscale)), resample)
    return image


def evaluate(case, font_dir, engine, recognizer, labels, rec_height, rec_width):
    drawn = unicodedata.normalize("NFC", case.text)
    position = case.target_position()
    context = context_of(case.candidate)
    row = {
        "index": case.index,
        "row_digest": hashlib.sha256(
            f"discovery|{case.index}".encode()).hexdigest(),
        "word": case.candidate.bare,
        "word_group": case.candidate.group,
        "font": case.font, "target_stratum": case.stratum,
        "nominal_size": case.size, "upscale": case.upscale,
        "template": case.template,
        "visual_target": case.target_character,
        "target_position": position,
        "target_ordinal": case.candidate.target_index(),
        "normalized_position": context["normalized_position"],
        "preceding_class": context["preceding_class"],
        "following_class": context["following_class"],
        "local_context": context["local_context"],
        "word_length": context["word_length"],
        "e_count": context["e_count"],
        "has_apostrophe": context["has_apostrophe"],
        "has_digit": context["has_digit"],
        "terminal_reason": "NOT_EVALUATED",
        "line_width": None, "line_aspect_ratio": None,
        "runtime_ink_height": None, "measured_stratum": None,
        "crop_width": None, "crop_height": None,
        "ctc_token_start": None, "ctc_token_end": None, "ctc_span": None,
        "decoded_text": None, "expected_substring": None,
        "clean_hallucination": False, "clean_preservation": False,
    }
    if position < 0 or position >= len(drawn):
        row["terminal_reason"] = "NOT_ELIGIBLE_NO_TARGET"
        return row
    try:
        page = render(case, font_dir)
        bgr = np.asarray(page)[:, :, ::-1].copy()
        row["line_width"] = int(page.width)
        row["line_aspect_ratio"] = round(page.width / max(1, page.height), 4)
        boxes, _ = engine.auto_text_det(bgr)
    except Exception as error:                        # pragma: no cover
        row["terminal_reason"] = "RENDER_ERROR"
        row["reason"] = type(error).__name__
        return row
    if boxes is None or len(boxes) == 0:
        row["terminal_reason"] = "DETECTOR_MISS"
        return row

    crops = engine.get_crop_img_list(bgr, boxes)
    font = ImageFont.truetype(str(font_dir / case.font), case.size)
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    origin_x = case.pad_x + case.jitter_x
    before = draw.textlength(drawn[:position], font=font)
    glyph = draw.textlength(drawn[position], font=font)
    target_x = (origin_x + before + glyph / 2.0) * case.upscale
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
        row["terminal_reason"] = "RECOGNIZER_FAILURE"
        row["reason"] = type(error).__name__
        return row
    row["decoded_text"] = decoded
    emitted = collapse_ctc(logits[0].argmax(axis=-1).tolist(), labels)
    if "".join(item["char"] for item in emitted) != decoded:
        row["terminal_reason"] = "ALIGNMENT_AMBIGUITY"
        return row

    xs = [point[0] for point in boxes[chosen]]
    low, high = min(xs), max(xs)
    kept, target_index = [], -1
    for order, character in enumerate(drawn):
        offset = draw.textlength(drawn[:order], font=font)
        width = draw.textlength(character, font=font)
        centre = (origin_x + offset + width / 2.0) * case.upscale
        if low <= centre <= high:
            if order == position:
                target_index = len(kept)
            kept.append(character)
    expected = "".join(kept)
    row["expected_substring"] = expected
    if target_index < 0:
        row["terminal_reason"] = "NO_TARGET_TOKEN"
        return row
    if target_index < len(emitted):
        token = emitted[target_index]
        row["ctc_token_start"] = token["start"]
        row["ctc_token_end"] = token["end"]
        row["ctc_span"] = token["end"] - token["start"] + 1

    normalized = unicodedata.normalize("NFC", decoded)
    if len(normalized) != len(expected):
        row["terminal_reason"] = ("INSERTION" if len(normalized) > len(expected)
                                  else "DELETION")
        return row
    differing = [i for i, (a, b) in enumerate(zip(expected, normalized)) if a != b]
    if not differing:
        if case.target_character == "é":
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()

    prior: set[str] = set()
    for recipe in V2_RECIPES.values():
        prior |= {w for pair in recipe.words for w in pair}
    from mine_line_triggers_v1 import WORD_SPLITS
    from rate_pilot_v2 import PILOT_WORDS
    from safety_stress_recipe_v2 import STRESS_WORDS
    for pairs in WORD_SPLITS.values():
        prior |= {w for pair in pairs for w in pair}
    prior |= {w for pair in PILOT_WORDS for w in pair}
    prior |= {w for pair in STRESS_WORDS for w in pair}
    assert_no_prior_overlap(prior)

    total = min(args.limit or DISCOVERY_MAX_RENDERINGS, DISCOVERY_MAX_RENDERINGS)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fonts = {}
    for name in DISCOVERY_FONTS:
        path = (args.font_dir / name).resolve()
        if not path.is_file():
            print(f"missing font {name}", file=sys.stderr)
            return 1
        fonts[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    recipe = {
        "dataset": "development_word_context_discovery_v1",
        "role": "development_word_support_diagnostic_only",
        "prohibited_uses": [
            "train/calibration/preflight quota", "model training",
            "threshold determination", "safety gate evidence",
        ],
        "seed": DISCOVERY_SEED,
        "candidates": [{"bare": c.bare, "accented": c.accented,
                        "group": c.group, **context_of(c)} for c in CANDIDATES],
        "candidate_count": len(CANDIDATES),
        "renderings_per_word": RENDERINGS_PER_WORD,
        "total_renderings": len(CANDIDATES) * RENDERINGS_PER_WORD,
        "fonts": fonts, "templates": list(DISCOVERY_TEMPLATES),
        "stratum_targets": {k: {"sizes": list(v["sizes"]),
                                "upscale": list(v["upscale"])}
                            for k, v in STRATUM_TARGETS.items()},
        "support_rules": {
            "ROBUST_SUPPORT": (f"clean hallucination >= {ROBUST_MIN_EVENTS} AND "
                               f"observed in >= {ROBUST_MIN_BREADTH} fonts or "
                               f">= {ROBUST_MIN_BREADTH} strata"),
            "SPARSE_SUPPORT": "clean hallucination 1-4",
            "NOT_OBSERVED": ("clean hallucination 0 -- means no event was seen "
                             "at this exposure, NOT that the rate is zero"),
        },
        "stopping_rule": ("every word renders all 400 cases; stopping a word "
                          "on its event count would tie exposure to outcome"),
        "excluded_strings": "the real UI text and its misread form are excluded",
        "prohibited_inputs": ["Expected text", "product UI captures"],
    }
    recipe_path = args.out_dir / "recipe.json"
    if not recipe_path.is_file():
        atomic_write_json(recipe_path, recipe)
    digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    log_line(f"discovery recipe sha256 {digest}")
    log_line(f"{len(CANDIDATES)} words x {RENDERINGS_PER_WORD} = {total:,} renderings")
    if args.seal_only:
        return 0

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    checkpoint = args.out_dir / "checkpoint.jsonl"
    state = load_checkpoint(checkpoint, unit_field="index")
    log_line(f"resuming with {len(state.rows)} rows "
             f"(resume #{state.resume_count}) pid {os.getpid()}")

    counts = Counter()
    started = time.time()
    with WriterLock(checkpoint):
        with CheckpointWriter(checkpoint, state.digests, flush_every=100) as writer:
            for index, case in resumable_units(total, DiscoveryCase,
                                               state.completed_units):
                row = evaluate(case, args.font_dir, engine, recognizer, labels,
                               rec_height, rec_width)
                row["diagnostic_flags"] = derive_flags(row)
                writer.append(row)
                counts[row["terminal_reason"]] += 1
                if (index + 1) % args.progress_every == 0:
                    writer.sync()
                    log_line(f"  {index + 1}/{total} "
                             f"h={counts['CLEAN_HALLUCINATION']} "
                             f"p={counts['CLEAN_PRESERVATION']} "
                             f"{time.time() - started:.0f}s")
        duplicates = writer.duplicates_rejected

    log_line(f"finished, {duplicates} duplicates rejected, "
             f"{time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
