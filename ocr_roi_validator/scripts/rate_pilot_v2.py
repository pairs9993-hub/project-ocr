"""Measure the hallucination rate per rendering, by font and macro stratum.

The main budget was going to be sized from cells holding two and three events.
That is not an estimate anyone should commit twenty hours to, so this pilot
buys a firmer one: a fixed 60,000 renderings, 20,000 aimed at each macro
stratum, with the fonts each split will actually use balanced deterministically
inside every stratum.

Font matters here. The font main effect is confirmed, so a rate measured on one
typeface does not transfer to another, and every planned font is measured in
every stratum. The interaction is INCONCLUSIVE, so nothing here assumes the
stratum effect has the same shape for each font -- the cells are reported
separately and left that way.

The full 20,000 per stratum runs even after enough events accumulate. Stopping
when the count looks sufficient would make the stopping rule depend on the
outcome and bias the very rate being measured.

Output is development_rate_diagnostic_only. It sizes a budget and nothing else:
not training, not calibration, not any quota, not a safety gate.
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

# Fonts each v2 split will use. Disjoint across splits, as required, and all
# measured here because the rate differs by font.
SPLIT_FONTS = {
    "supplement": ["arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf",
                   "corbel.ttf", "Candara.ttf"],
    "calibration": ["trebuc.ttf", "georgia.ttf", "palab.ttf"],
    "preflight": ["times.ttf", "framd.ttf", "tahoma.ttf", "consola.ttf"],
}
PILOT_FONTS = [f for fonts in SPLIT_FONTS.values() for f in fonts]

# Targeting settings measured in D2. Each stratum gets the (size, upscale)
# combination that landed there most often; these are fixed now and are not
# revised after seeing pilot counts.
STRATUM_TARGETS = {
    "SMALL": {"sizes": (11, 12, 13), "upscale": (1.0, 1.0)},
    "MEDIUM": {"sizes": (12, 13, 14), "upscale": (1.45, 1.75)},
    "LARGE": {"sizes": (17, 18, 20), "upscale": (1.55, 2.05)},
}
RENDERINGS_PER_STRATUM = 20000
TOTAL_RENDERINGS = RENDERINGS_PER_STRATUM * len(STRATUM_TARGETS)
PILOT_SEED = 20260817

# Words and templates disjoint from every v1/v2 cohort, so the pilot cannot
# contaminate the datasets it is sizing.
PILOT_WORDS = [
    ("bequille", "béquillé"), ("chevre", "chévré"), ("presage", "présagé"),
    ("tremble", "trémblé"), ("liberte", "libérté"),
]
PILOT_TEMPLATES = ["{} plt", "Un {} zt", "{} 4,2 kb", "Zx {} md", "{}"]


class PilotCondition:
    """One rendering, fully determined by its index."""

    __slots__ = ("index", "stratum", "font", "size", "upscale", "word_index",
                 "accented", "template_index", "pad_x", "pad_y", "background",
                 "foreground", "contrast", "blur", "jitter_x", "jitter_y",
                 "resample")

    def __init__(self, index: int) -> None:
        self.index = index
        stratum_index = index // RENDERINGS_PER_STRATUM
        self.stratum = list(STRATUM_TARGETS)[stratum_index]
        within = index % RENDERINGS_PER_STRATUM
        # Deterministic round-robin: every font gets the same share of every
        # stratum, so no cell is thin by accident.
        self.font = PILOT_FONTS[within % len(PILOT_FONTS)]

        target = STRATUM_TARGETS[self.stratum]
        rng = random.Random(PILOT_SEED + index)
        self.size = target["sizes"][within % len(target["sizes"])]
        self.upscale = round(rng.uniform(*target["upscale"]), 4)
        self.word_index = rng.randrange(len(PILOT_WORDS))
        self.accented = rng.random() < 0.5
        self.template_index = rng.randrange(len(PILOT_TEMPLATES))
        self.pad_x = rng.randint(12, 30)
        self.pad_y = 20                      # fixed; see diagnose_pixel_geometry
        dark = rng.random() < 0.75
        self.background = rng.randint(8, 34) if dark else rng.randint(226, 250)
        self.foreground = rng.randint(220, 252) if dark else rng.randint(8, 40)
        self.contrast = round(rng.uniform(0.85, 1.20), 4)
        self.blur = round(rng.uniform(0.0, 0.5), 4)
        self.jitter_x = round(rng.uniform(-0.5, 0.5), 4)
        self.jitter_y = round(rng.uniform(-0.5, 0.5), 4)
        self.resample = rng.choice(["bicubic", "lanczos"])

    @property
    def template(self) -> str:
        return PILOT_TEMPLATES[self.template_index]

    @property
    def text(self) -> str:
        plain, accented = PILOT_WORDS[self.word_index]
        return self.template.format(accented if self.accented else plain)

    @property
    def target_character(self) -> str:
        return "é" if self.accented else "e"

    def target_position(self) -> int:
        text = unicodedata.normalize("NFC", self.text)
        offset = self.template.index("{}")
        for position in range(offset, len(text)):
            if text[position] in {"e", "é"}:
                return position
        return -1


def render(condition: PilotCondition, font_dir: Path) -> Image.Image:
    font = ImageFont.truetype(str(font_dir / condition.font), condition.size)
    origin_x = condition.pad_x + condition.jitter_x
    origin_y = condition.pad_y + condition.jitter_y
    box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox(
        (origin_x, origin_y), condition.text, font=font)
    image = Image.new("RGB", (int(box[2] + condition.pad_x),
                              int(box[3] + condition.pad_y)),
                      (condition.background,) * 3)
    ImageDraw.Draw(image).text((origin_x, origin_y), condition.text, font=font,
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


def evaluate(condition, font_dir, engine, recognizer, labels, rec_height,
             rec_width) -> dict:
    drawn = unicodedata.normalize("NFC", condition.text)
    position = condition.target_position()
    row = {
        "index": condition.index,
        "row_digest": hashlib.sha256(f"pilot|{condition.index}".encode()).hexdigest(),
        "target_stratum": condition.stratum,
        "font": condition.font,
        "nominal_size": condition.size,
        "upscale": condition.upscale,
        "visual_target": condition.target_character,
        "terminal_reason": "NOT_EVALUATED",
        "runtime_ink_height": None,
        "measured_stratum": None,
        "clean_hallucination": False,
        "clean_preservation": False,
    }
    if position < 0:
        row["terminal_reason"] = "NOT_ELIGIBLE_NO_TARGET"
        return row
    try:
        page = render(condition, font_dir)
        bgr = np.asarray(page)[:, :, ::-1].copy()
        boxes, _ = engine.auto_text_det(bgr)
    except Exception as error:                        # pragma: no cover
        row["terminal_reason"] = "RENDER_ERROR"
        return row
    if boxes is None or len(boxes) == 0:
        row["terminal_reason"] = "DETECTOR_MISS"
        return row

    crops = engine.get_crop_img_list(bgr, boxes)
    font = ImageFont.truetype(str(font_dir / condition.font), condition.size)
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    before = draw.textlength(condition.text[:position], font=font)
    width = draw.textlength(condition.text[position], font=font)
    target_x = ((condition.pad_x + condition.jitter_x + before + width / 2.0)
                * condition.upscale)
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
    except Exception:                                 # pragma: no cover
        row["terminal_reason"] = "RECOGNIZER_FAILURE"
        return row
    emitted = collapse_ctc(logits[0].argmax(axis=-1).tolist(), labels)
    if "".join(i["char"] for i in emitted) != decoded:
        row["terminal_reason"] = "ALIGNMENT_AMBIGUITY"
        return row

    kept, target_index = [], -1
    xs = [point[0] for point in boxes[chosen]]
    low, high = min(xs), max(xs)
    for order, character in enumerate(drawn):
        offset = draw.textlength(drawn[:order], font=font)
        glyph_width = draw.textlength(character, font=font)
        centre = ((condition.pad_x + condition.jitter_x + offset
                   + glyph_width / 2.0) * condition.upscale)
        if low <= centre <= high:
            if order == position:
                target_index = len(kept)
            kept.append(character)
    expected = "".join(kept)
    if target_index < 0:
        row["terminal_reason"] = "NO_TARGET_TOKEN"
        return row

    normalized = unicodedata.normalize("NFC", decoded)
    if len(normalized) != len(expected):
        row["terminal_reason"] = ("INSERTION" if len(normalized) > len(expected)
                                  else "DELETION")
        return row
    differing = [i for i, (a, b) in enumerate(zip(expected, normalized)) if a != b]
    if not differing:
        if condition.target_character == "é":
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fonts = {}
    for name in PILOT_FONTS:
        path = (args.font_dir / name).resolve()
        if not path.is_file():
            print(f"missing font {name}", file=sys.stderr)
            return 1
        fonts[name] = {
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "family": ImageFont.truetype(str(path), 20).getname()[0],
        }

    recipe = {
        "pilot": "rate_confirmation_v2",
        "role": "development_rate_diagnostic_only",
        "prohibited_uses": [
            "train/calibration/preflight quota", "model training",
            "threshold determination", "safety gate evidence",
        ],
        "seed": PILOT_SEED,
        "renderings_per_stratum": RENDERINGS_PER_STRATUM,
        "total_renderings": TOTAL_RENDERINGS,
        "stratum_targets": STRATUM_TARGETS,
        "split_fonts": SPLIT_FONTS,
        "fonts": fonts,
        "words": PILOT_WORDS,
        "templates": PILOT_TEMPLATES,
        "font_balance": "deterministic round-robin within each stratum",
        "stopping_rule": ("the full 20,000 per stratum always completes; "
                          "outcome-dependent stopping would bias the rate"),
        "independence": ("words and templates are disjoint from every v1 and "
                         "planned v2 cohort"),
    }
    recipe_path = args.out_dir / "recipe.json"
    if not recipe_path.is_file():
        atomic_write_json(recipe_path, recipe)
    digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    log_line(f"pilot recipe sha256 {digest}")
    log_line(f"budget {TOTAL_RENDERINGS} renderings "
             f"({RENDERINGS_PER_STRATUM} per stratum x {len(STRATUM_TARGETS)})")
    if args.seal_only:
        return 0

    total = args.limit or TOTAL_RENDERINGS
    checkpoint = args.out_dir / "checkpoint.jsonl"
    state = load_checkpoint(checkpoint, unit_field="index")
    log_line(f"resuming with {len(state.rows)} rows (resume #{state.resume_count}), "
             f"pid {os.getpid()}")

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    counts = Counter()
    started = time.time()
    with CheckpointWriter(checkpoint, state.digests, flush_every=100) as writer:
        for index, condition in resumable_units(
                total, PilotCondition, state.completed_units):
            row = evaluate(condition, args.font_dir, engine, recognizer, labels,
                           rec_height, rec_width)
            writer.append(row)
            counts[row["terminal_reason"]] += 1
            if (index + 1) % args.progress_every == 0:
                writer.sync()
                log_line(f"  {index + 1}/{total} "
                         f"halluc={counts['CLEAN_HALLUCINATION']} "
                         f"elapsed={time.time() - started:.0f}s")
    log_line(f"finished {writer.written} rows, "
             f"{writer.duplicates_rejected} duplicates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
