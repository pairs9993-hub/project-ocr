"""Matched pixel-geometry diagnostic for the e -> é hallucination.

Two earlier attempts failed for reasons worth stating, because this design is
shaped by both.

The v1 mining recorded a row only when the recognizer *emitted* an é. Everything
else -- a visual e read correctly, a visual é flattened to e, a line the detector
never found -- left no trace, so there was no denominator and no rate could be
computed. Here every eligible occurrence is written, whatever the recognizer
does with it, and the funnel is reported stage by stage so nothing drops out
silently.

The matched font diagnostic then showed that all eleven fonts hallucinate once
conditions are held equal, which killed the font hypothesis but left the real
cause open. Nominal point size was the obvious suspect, but nominal size is not
what reaches the recognizer: padding, upscale, the detector's crop and the
recognizer's fixed-height resize all intervene. So the analysis axis here is
*measured* geometry -- ink height, glyph height, occupancy, resize scale -- with
nominal size reported alongside rather than in place of it.

The budget is a sealed integer matrix. Detector output and hallucination counts
are results, so neither may terminate the run: the full matrix is completed even
if triggers appear early.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from mine_line_triggers_v1 import build_engine, collapse_ctc, load_labels  # noqa: E402
from ocr_roi_validator.diagnostic_runner import (  # noqa: E402
    CheckpointWriter, atomic_write_json, load_checkpoint, log_line,
    resumable_units,
)
from ocr_roi_validator.glyph_geometry import (  # noqa: E402
    GLYPH_HEIGHT_BINS, INK_HEIGHT_BINS, OCCUPANCY_BINS, RESIZE_BINS, bin_value,
    measure_line_geometry, measure_target_glyph,
)

# ---------------------------------------------------------------- sealed budget
# Chosen before any inference and never adjusted afterwards. Two fonts are taken
# from each v1 cohort, in cohort order -- not by how many hallucinations they
# produced, which is the result being measured.
FONTS = {
    "arial.ttf": "train_v1", "calibri.ttf": "train_v1",
    "trebuc.ttf": "calibration_v1", "georgia.ttf": "calibration_v1",
    "times.ttf": "preflight_v1", "framd.ttf": "preflight_v1",
}
SIZES = (10, 12, 13, 14, 15, 16, 18, 21, 24, 27)
PADDING_BUCKETS = {"tight": (10, 16), "wide": (28, 44)}
VERTICAL_PADDING = 20               # fixed; see BaseCondition for why
UPSCALE_BUCKETS = {"none": (1.0, 1.0), "high": (1.5, 2.1)}
POLARITY_BUCKETS = ("dark", "light")
CONTRAST_BUCKETS = {"low": (0.80, 0.92), "unit": (1.0, 1.0), "high": (1.12, 1.28)}
BLUR_BUCKETS = {"none": (0.0, 0.0), "soft": (0.20, 0.40), "heavy": (0.50, 0.75)}

RENDERINGS_PER_CELL = 40
CELLS = len(SIZES) * len(PADDING_BUCKETS) * len(UPSCALE_BUCKETS) * len(POLARITY_BUCKETS)
BASE_CONDITIONS = CELLS * RENDERINGS_PER_CELL          # 80 * 40 = 3200
TOTAL_RENDERINGS = BASE_CONDITIONS * len(FONTS)        # 3200 * 6 = 19200
DIAGNOSTIC_SEED = 9100000

WORDS = [
    ("reglage", "réglagé"), ("element", "élémént"), ("decale", "décalé"),
    ("reserve", "résérvé"), ("general", "général"), ("repare", "réparé"),
    ("melange", "mélangé"), ("degage", "dégagé"), ("severe", "sévéré"),
    ("deneige", "dénéigé"),
]
TEMPLATES = [
    "{}", "Vous {} localis", "Il {} lla", "{} kd/hb 2,5", "L'{} du bac 1,5",
    "Application {} tot", "H'{} ktb 1,5", "{} du top", "Tk {} biffl", "{}: tud 30",
]


class BaseCondition:
    """One optical setting, rendered identically in every font."""

    __slots__ = ("index", "size", "padding_bucket", "upscale_bucket",
                 "polarity", "contrast_bucket", "blur_bucket", "word_index",
                 "accented", "template_index", "pad_x", "pad_y", "upscale",
                 "background", "foreground", "contrast", "blur", "jitter_x",
                 "jitter_y", "resample")

    def __init__(self, index: int) -> None:
        # Cell assignment is deterministic from the index, so the matrix is
        # balanced by construction rather than by chance.
        self.index = index
        cell = index // RENDERINGS_PER_CELL
        self.size = SIZES[cell % len(SIZES)]
        cell //= len(SIZES)
        self.padding_bucket = list(PADDING_BUCKETS)[cell % len(PADDING_BUCKETS)]
        cell //= len(PADDING_BUCKETS)
        self.upscale_bucket = list(UPSCALE_BUCKETS)[cell % len(UPSCALE_BUCKETS)]
        cell //= len(UPSCALE_BUCKETS)
        self.polarity = POLARITY_BUCKETS[cell % len(POLARITY_BUCKETS)]

        # Contrast and blur rotate within the cell rather than multiplying the
        # matrix, keeping the sealed budget an exact integer.
        repeat = index % RENDERINGS_PER_CELL
        self.contrast_bucket = list(CONTRAST_BUCKETS)[repeat % len(CONTRAST_BUCKETS)]
        self.blur_bucket = list(BLUR_BUCKETS)[(repeat // 3) % len(BLUR_BUCKETS)]

        rng = random.Random(DIAGNOSTIC_SEED + index)
        self.word_index = rng.randrange(len(WORDS))
        self.accented = rng.random() < 0.5
        self.template_index = rng.randrange(len(TEMPLATES))
        self.pad_x = rng.randint(*PADDING_BUCKETS[self.padding_bucket])
        # Vertical padding is held at the level the product's own pipeline
        # produces, and is NOT part of the padding axis. Deriving it from
        # pad_x gave 3-5px canvases; because the detector resizes the short
        # side to 640, those were being blown up 20-30x and the text was
        # destroyed before detection -- a 1.6% yield versus 100% at pad_y=20.
        # The horizontal padding ratio remains the axis under test.
        self.pad_y = VERTICAL_PADDING
        self.upscale = round(rng.uniform(*UPSCALE_BUCKETS[self.upscale_bucket]), 4)
        if self.polarity == "dark":
            self.background, self.foreground = rng.randint(8, 34), rng.randint(220, 252)
        else:
            self.background, self.foreground = rng.randint(226, 250), rng.randint(8, 40)
        self.contrast = round(rng.uniform(*CONTRAST_BUCKETS[self.contrast_bucket]), 4)
        self.blur = round(rng.uniform(*BLUR_BUCKETS[self.blur_bucket]), 4)
        self.jitter_x = round(rng.uniform(-0.5, 0.5), 4)
        self.jitter_y = round(rng.uniform(-0.5, 0.5), 4)
        self.resample = rng.choice(["bicubic", "lanczos"])

    @property
    def template(self) -> str:
        return TEMPLATES[self.template_index]

    @property
    def text(self) -> str:
        plain, accented = WORDS[self.word_index]
        return self.template.format(accented if self.accented else plain)

    @property
    def target_character(self) -> str:
        return "é" if self.accented else "e"

    def target_position(self) -> int:
        """Index of the target occurrence, fixed by the renderer alone.

        Determined before any OCR runs, from the drawn string only. The first
        e-form inside the substituted word is the target.
        """
        text = unicodedata.normalize("NFC", self.text)
        offset = self.template.index("{}")
        for position in range(offset, len(text)):
            if text[position] in {"e", "é"}:
                return position
        return -1

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}


def render(condition: BaseCondition, font_path: Path) -> Image.Image:
    font = ImageFont.truetype(str(font_path), condition.size)
    text = condition.text
    # textbbox returns absolute coordinates for the anchor, so the canvas must
    # enclose box[3], not merely its height. The earlier version sized by
    # (box[3] - box[1]) and then drew at pad_y, which clipped descenders and
    # whole glyphs at the larger point sizes -- the detector was being asked to
    # find text that had been cut off, and its yield duly collapsed from 97% at
    # size 10 to 16% at size 27.
    origin_x = condition.pad_x + condition.jitter_x
    origin_y = condition.pad_y + condition.jitter_y
    box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox(
        (origin_x, origin_y), text, font=font)
    width = int(math.ceil(box[2] + condition.pad_x))
    height = int(math.ceil(box[3] + condition.pad_y))
    image = Image.new("RGB", (width, height), (condition.background,) * 3)
    ImageDraw.Draw(image).text((origin_x, origin_y), text, font=font,
                               fill=(condition.foreground,) * 3)
    if condition.blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(condition.blur))
    if condition.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(condition.contrast)
    if condition.upscale != 1.0:
        resample = (Image.Resampling.BICUBIC if condition.resample == "bicubic"
                    else Image.Resampling.LANCZOS)
        image = image.resize((int(image.width * condition.upscale),
                              int(image.height * condition.upscale)), resample)
    return image


def glyph_raster_hash(font_path: Path, char: str, size: int = 20) -> str:
    font = ImageFont.truetype(str(font_path), size)
    canvas = Image.new("L", (60, 44), 0)
    ImageDraw.Draw(canvas).text((6, 6), char, font=font, fill=255)
    return hashlib.sha256(canvas.tobytes()).hexdigest()


def seal_fonts(font_dir: Path) -> dict:
    """Record every font's identity before the first inference."""
    sealed: dict[str, dict] = {}
    for name, cohort in FONTS.items():
        path = (font_dir / name).resolve()
        if not path.is_file():
            raise SystemExit(f"missing font: {path}")
        bare, accented = glyph_raster_hash(path, "e"), glyph_raster_hash(path, "é")
        if bare == accented:
            raise SystemExit(f"{name} renders e and é identically")
        sealed[name] = {
            "path": str(path), "cohort": cohort,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "family": ImageFont.truetype(str(path), 20).getname()[0],
            "glyph_raster_sha256_e": bare, "glyph_raster_sha256_eacute": accented,
        }
    shared = defaultdict(list)
    for name, info in sealed.items():
        shared[info["glyph_raster_sha256_eacute"]].append(name)
    collisions = {k: v for k, v in shared.items() if len(v) > 1}
    if collisions:
        raise SystemExit(f"font fallback suspected: {collisions}")
    return sealed


def _target_x_centre(condition: BaseCondition, font_path: str) -> float | None:
    """Horizontal centre of the target glyph on the rendered page.

    Derived from the renderer's own layout -- the drawn string and the font
    metrics -- so it is available before any OCR and cannot be influenced by
    what the recognizer says.
    """
    try:
        font = ImageFont.truetype(font_path, condition.size)
    except OSError:                                   # pragma: no cover
        return None
    position = condition.target_position()
    if position < 0:
        return None
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    origin_x = condition.pad_x + condition.jitter_x
    origin_y = condition.pad_y + condition.jitter_y
    before = draw.textlength(condition.text[:position], font=font)
    glyph = draw.textlength(condition.text[position], font=font)
    return (origin_x + before + glyph / 2.0) * condition.upscale


def _expected_substring(condition: BaseCondition, font_path: str,
                        box) -> tuple[str, int]:
    """Characters whose centres fall inside ``box``, and the target's index.

    Uses only the renderer's layout, so a split line can still be scored
    exactly rather than being written off as a deletion.
    """
    try:
        font = ImageFont.truetype(font_path, condition.size)
    except OSError:                                   # pragma: no cover
        return "", -1
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    text = unicodedata.normalize("NFC", condition.text)
    origin_x = condition.pad_x + condition.jitter_x
    xs = [point[0] for point in box]
    low, high = min(xs), max(xs)

    kept: list[str] = []
    target_index = -1
    target = condition.target_position()
    for index, character in enumerate(text):
        before = draw.textlength(text[:index], font=font)
        width = draw.textlength(character, font=font)
        centre = (origin_x + before + width / 2.0) * condition.upscale
        if low <= centre <= high:
            if index == target:
                target_index = len(kept)
            kept.append(character)
    return "".join(kept), target_index


def evaluate(condition: BaseCondition, font_name: str, font_info: dict,
             engine, recognizer, labels, rec_height: int,
             rec_width: int) -> dict:
    """Render one matched cell and record the full funnel for it.

    Returns one row per eligible occurrence. The row is written whatever
    happens -- detector miss included -- so the denominator survives.
    """
    drawn = unicodedata.normalize("NFC", condition.text)
    position = condition.target_position()
    row = {
        "base_condition": condition.index,
        "occurrence_id": f"bc{condition.index}-p{position}",
        "pair_id": f"bc{condition.index}",
        "row_digest": hashlib.sha256(
            f"{condition.index}|{font_name}|{position}".encode()).hexdigest(),
        "font": font_name,
        "font_cohort": font_info["cohort"],
        "nominal_size": condition.size,
        "size_bucket": condition.size,
        "padding_bucket": condition.padding_bucket,
        "upscale_bucket": condition.upscale_bucket,
        "upscale_factor": condition.upscale,
        "polarity": condition.polarity,
        "contrast_bucket": condition.contrast_bucket,
        "blur_bucket": condition.blur_bucket,
        "template_index": condition.template_index,
        "word_index": condition.word_index,
        "visual_target": condition.target_character,
        "target_position": position,
        "eligible": position >= 0,
        # funnel flags, all false until earned
        "detector_found": False, "target_line_matched": False,
        "recognizer_decoded": False, "sequence_aligned": False,
        "clean_eligible": False, "clean_hallucination": False,
        "clean_preservation": False,
        "outcome": "NOT_EVALUATED", "reason": None,
        # geometry, null until measured
        "rendered_glyph_width": None, "rendered_glyph_height": None,
        "rendered_ink_height": None, "rendered_occupancy": None,
        "crop_width": None, "crop_height": None, "runtime_ink_height": None,
        "recognizer_resize_scale": None, "horizontal_padding_ratio": None,
        "clipped": None,
        "decoded_text": None, "decoded_length": None,
    }
    if position < 0:
        row["outcome"] = "NOT_ELIGIBLE_NO_TARGET"
        return row

    try:
        page = render(condition, Path(font_info["path"]))
        bgr = np.asarray(page)[:, :, ::-1].copy()
    except Exception as error:                       # pragma: no cover
        row["outcome"] = "RENDER_ERROR"
        row["reason"] = type(error).__name__
        return row

    # Rendered geometry: what was drawn, before the detector sees it.
    page_geometry = measure_line_geometry(bgr, rec_height)
    if page_geometry is not None:
        glyph = measure_target_glyph(condition.target_character,
                                     font_info["path"], condition.size,
                                     page_geometry, condition.upscale)
        row["rendered_ink_height"] = page_geometry.ink_height
        if glyph is not None:
            row["rendered_glyph_width"] = glyph.glyph_width
            row["rendered_glyph_height"] = glyph.glyph_height
            row["rendered_occupancy"] = round(glyph.glyph_occupancy, 8)

    try:
        boxes, _ = engine.auto_text_det(bgr)
    except Exception as error:                       # pragma: no cover
        row["outcome"] = "DETECTOR_ERROR"
        row["reason"] = type(error).__name__
        return row
    if boxes is None or len(boxes) == 0:
        row["outcome"] = "DETECTOR_MISS"
        row["reason"] = "no box returned"
        return row
    row["detector_found"] = True

    # The detector often splits one rendered line into several boxes. Discarding
    # those samples threw away 36% of the matrix, and the discard correlated
    # with padding and size -- the very axes under test -- so the survivors were
    # selected by the variable being measured. The target's box is instead
    # chosen geometrically: the renderer knows where the target glyph sits, so
    # the box containing that x-position is the target's. The decoded text is
    # never consulted, which is what kept the earlier rule honest.
    crops = engine.get_crop_img_list(bgr, boxes)
    row["detector_box_count"] = len(crops)
    target_x = _target_x_centre(condition, font_info["path"])
    if target_x is None:
        row["outcome"] = "NO_TARGET_TOKEN"
        row["reason"] = "target glyph x position not derivable"
        return row
    chosen = None
    for index, box in enumerate(boxes):
        xs = [point[0] for point in box]
        if min(xs) <= target_x <= max(xs):
            chosen = index
            break
    if chosen is None:
        row["outcome"] = "WRONG_LINE_SELECTED"
        row["reason"] = f"target x={target_x:.1f} in none of {len(crops)} boxes"
        return row
    crop = crops[chosen]
    row["target_line_matched"] = True
    row["chosen_box_index"] = chosen

    crop_height, crop_width = crop.shape[:2]
    geometry = measure_line_geometry(crop, rec_height)
    row["crop_width"], row["crop_height"] = int(crop_width), int(crop_height)
    if geometry is not None:
        row["runtime_ink_height"] = geometry.ink_height
        row["recognizer_resize_scale"] = round(geometry.recognizer_resize_scale, 6)
        row["horizontal_padding_ratio"] = round(geometry.horizontal_padding_ratio, 6)
        row["clipped"] = bool(geometry.ink_top <= 0
                              or geometry.ink_bottom >= crop_height - 1)

    max_wh_ratio = max(rec_width / rec_height, crop_width / float(crop_height))
    try:
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_width / float(crop_height)],
            max_wh_ratio=max_wh_ratio)[0][0]
    except Exception as error:                       # pragma: no cover
        row["outcome"] = "RECOGNIZER_FAILURE"
        row["reason"] = type(error).__name__
        return row
    row["recognizer_decoded"] = True
    row["decoded_text"] = decoded
    normalized = unicodedata.normalize("NFC", decoded)
    row["decoded_length"] = len(normalized)

    emitted = collapse_ctc(logits[0].argmax(axis=-1).tolist(), labels)
    if "".join(i["char"] for i in emitted) != decoded:
        row["outcome"] = "ALIGNMENT_AMBIGUITY"
        row["reason"] = "ctc collapse disagrees with postprocess"
        return row
    row["sequence_aligned"] = True

    # When the detector split the line, this crop holds only part of the text.
    # Comparing it against the whole drawn string would score every split as a
    # deletion, so the expected substring is recomputed from the renderer's own
    # layout for the box that was chosen -- again without consulting the decode.
    expected, position = _expected_substring(condition, font_info["path"],
                                             boxes[chosen])
    row["expected_substring"] = expected
    row["target_position_in_crop"] = position
    if position < 0:
        row["outcome"] = "NO_TARGET_TOKEN"
        row["reason"] = "target fell outside the chosen box"
        return row
    drawn = expected

    if len(normalized) != len(drawn):
        row["outcome"] = ("INSERTION" if len(normalized) > len(drawn)
                          else "DELETION")
        return row

    differing = [i for i, (a, b) in enumerate(zip(drawn, normalized)) if a != b]
    row["clean_eligible"] = True
    if not differing:
        if condition.target_character == "é":
            row["clean_preservation"] = True
            row["outcome"] = "CLEAN_PRESERVATION"
        else:
            row["outcome"] = "CLEAN_CORRECT_BARE_E"
        return row
    if len(differing) > 1:
        row["outcome"] = "MULTIPLE_CHANGES"
        return row
    index = differing[0]
    if index != position:
        row["outcome"] = "CHANGE_ELSEWHERE"
        return row
    if drawn[index] == "e" and normalized[index] == "é":
        row["clean_hallucination"] = True
        row["outcome"] = "CLEAN_HALLUCINATION"
    elif drawn[index] == "é" and normalized[index] == "e":
        row["outcome"] = "ACCENT_LOST"
    else:
        row["outcome"] = "OTHER_SUBSTITUTION"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sealed_fonts = seal_fonts(args.font_dir)
    budget = {
        "base_conditions": BASE_CONDITIONS,
        "fonts": len(FONTS),
        "cells": CELLS,
        "renderings_per_cell": RENDERINGS_PER_CELL,
        "total_renderings": TOTAL_RENDERINGS,
        "eligible_occurrences_expected": TOTAL_RENDERINGS,
        "termination": ("the full matrix is always completed; detector output "
                        "and hallucination counts are results and may not end "
                        "the run"),
    }
    recipe = {
        "diagnostic": "matched_pixel_geometry_v1",
        "seed": DIAGNOSTIC_SEED, "budget": budget, "fonts": sealed_fonts,
        "sizes": SIZES, "padding_buckets": PADDING_BUCKETS,
        "upscale_buckets": UPSCALE_BUCKETS, "polarity_buckets": POLARITY_BUCKETS,
        "contrast_buckets": CONTRAST_BUCKETS, "blur_buckets": BLUR_BUCKETS,
        "words": WORDS, "templates": TEMPLATES,
        "bins": {"ink_height": INK_HEIGHT_BINS, "glyph_height": GLYPH_HEIGHT_BINS,
                 "occupancy": OCCUPANCY_BINS, "resize": RESIZE_BINS},
        "inference": "frozen detector + French baseline, ONNX Runtime CPU only",
        "model_predictions_used": False,
        "eligibility": ("decided from renderer ground truth before OCR; whether "
                        "the baseline emitted é has no bearing on it"),
    }
    recipe_path = args.out_dir / "recipe.json"
    if not recipe_path.is_file():
        digest = atomic_write_json(recipe_path, recipe)
        log_line(f"sealed recipe sha256 {digest}")
    else:
        digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
        log_line(f"existing recipe sha256 {digest}")
    log_line(f"budget: {BASE_CONDITIONS} base conditions x {len(FONTS)} fonts "
             f"= {TOTAL_RENDERINGS} renderings ({CELLS} cells x "
             f"{RENDERINGS_PER_CELL})")
    if args.seal_only:
        return 0

    checkpoint = args.out_dir / "checkpoint.jsonl"
    state = load_checkpoint(checkpoint, unit_field="base_condition")
    log_line(f"resuming with {len(state.rows)} rows, "
             f"{len(state.completed_units)} base conditions done "
             f"(resume #{state.resume_count}), pid {os.getpid()}")

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    started = time.time()
    exit_code = 0
    with CheckpointWriter(checkpoint, state.digests, flush_every=len(FONTS)) as writer:
        try:
            for index, condition in resumable_units(
                    BASE_CONDITIONS, BaseCondition, state.completed_units):
                for font_name, font_info in sealed_fonts.items():
                    writer.append(evaluate(
                        condition, font_name, font_info, engine, recognizer,
                        labels, rec_height, rec_width))
                if (index + 1) % args.progress_every == 0:
                    writer.sync()
                    elapsed = time.time() - started
                    log_line(f"  base condition {index + 1}/{BASE_CONDITIONS} "
                             f"rows={writer.written} dup={writer.duplicates_rejected} "
                             f"elapsed={elapsed:.0f}s")
        except KeyboardInterrupt:                    # pragma: no cover
            log_line("interrupted; checkpoint retains completed work")
            exit_code = 130
    log_line(f"writer finished: {writer.written} rows, "
             f"{writer.duplicates_rejected} duplicates rejected, exit {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
