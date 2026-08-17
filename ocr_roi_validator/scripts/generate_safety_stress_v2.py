"""Generate the geometry safety-stress set to fixed case counts.

The quota-driven splits are mined for events: they run until enough
hallucinations accumulate. This set is the opposite. Each condition gets an
exact number of cases decided in advance, because it measures false correction
-- a verifier changing something it should have left alone -- and that risk does
not scale with how often the baseline happens to hallucinate.

Two of the twelve conditions carry the cases that matter most:
LEGITIMATE_ACCENT_PRESERVED, where a real é was read correctly, and
BARE_E_MUST_NOT_CHANGE, where a plain e was. In both the correct behaviour is to
do nothing, and 2,400 of the 9,200 cases are of that kind.

Its prevalence is an artifact of deliberately over-sampling rare conditions, so
it must never be pooled with the natural splits for rate estimation. The
manifest records that prohibition rather than leaving it to convention.
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
from ocr_roi_validator.terminal_reason import (  # noqa: E402
    derive_flags, summarise_terminal_reasons,
)
from safety_stress_recipe_v2 import (  # noqa: E402
    CONDITIONS, STRESS_FONTS, STRESS_SEED, STRESS_TEMPLATES, STRESS_WORDS,
    TOTAL_CASES,
)

# Flattened so a case index maps to exactly one condition, deterministically.
CASE_PLAN: list[tuple[str, int]] = []
for _name, _entry in CONDITIONS.items():
    CASE_PLAN.extend((_name, _i) for _i in range(_entry["cases"]))


class StressCase:
    """One evaluation case, determined entirely by its index."""

    __slots__ = ("index", "condition", "font", "size", "upscale", "word_index",
                 "accented", "template", "pad_x", "pad_y", "background",
                 "foreground", "contrast", "blur", "jitter_x", "jitter_y",
                 "resample")

    def __init__(self, index: int) -> None:
        self.index = index
        self.condition, within = CASE_PLAN[index]
        entry = CONDITIONS[self.condition]
        rng = random.Random(STRESS_SEED + index)

        self.font = STRESS_FONTS[within % len(STRESS_FONTS)]
        self.size = entry["sizes"][within % len(entry["sizes"])]
        low, high = entry["upscale"]
        self.upscale = round(rng.uniform(low, high), 4)
        self.word_index = rng.randrange(len(STRESS_WORDS))

        # Two conditions fix which e-form is drawn; the rest alternate.
        if entry.get("force_accent"):
            self.accented = True
        elif entry.get("force_bare"):
            self.accented = False
        else:
            self.accented = (within % 2 == 0)

        templates = entry.get("templates", STRESS_TEMPLATES)
        self.template = templates[within % len(templates)]

        pad_x = entry.get("pad_x", (12, 30))
        pad_y = entry.get("pad_y", (20, 20))
        self.pad_x = rng.randint(*pad_x)
        self.pad_y = rng.randint(*pad_y)
        dark = rng.random() < 0.75
        self.background = rng.randint(8, 34) if dark else rng.randint(226, 250)
        self.foreground = rng.randint(220, 252) if dark else rng.randint(8, 40)
        self.contrast = round(rng.uniform(0.85, 1.20), 4)
        blur = entry.get("blur", (0.0, 0.5))
        self.blur = round(rng.uniform(*blur), 4)
        self.jitter_x = round(rng.uniform(-0.5, 0.5), 4)
        self.jitter_y = round(rng.uniform(-0.5, 0.5), 4)
        self.resample = rng.choice(["bicubic", "lanczos"])

    @property
    def text(self) -> str:
        plain, accented = STRESS_WORDS[self.word_index]
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


def render(case: StressCase, font_dir: Path) -> Image.Image:
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
    row = {
        "index": case.index,
        "row_digest": hashlib.sha256(
            f"stress|{case.index}".encode()).hexdigest(),
        "condition": case.condition,
        "font": case.font, "nominal_size": case.size, "upscale": case.upscale,
        "pad_x": case.pad_x, "template": case.template,
        "word_index": case.word_index,
        "visual_target": case.target_character,
        "target_position": position,
        "terminal_reason": "NOT_EVALUATED",
        "page_sha256": None, "crop_sha256": None,
        "runtime_ink_height": None, "measured_stratum": None,
        "crop_width": None, "crop_height": None,
        "horizontal_padding_ratio": None, "clipped": None,
        "decoded_text": None, "expected_substring": None,
        "decoded_length": None, "detector_box_count": None,
        "clean_hallucination": False, "clean_preservation": False,
        "e_forms_in_line": sum(1 for c in drawn if c in {"e", "é"}),
    }
    if position < 0:
        row["terminal_reason"] = "NOT_ELIGIBLE_NO_TARGET"
        return row
    try:
        page = render(case, font_dir)
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
    normalized = unicodedata.normalize("NFC", decoded)
    row["decoded_length"] = len(normalized)
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
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    if len(CASE_PLAN) != TOTAL_CASES:
        print(f"case plan {len(CASE_PLAN)} != declared {TOTAL_CASES}",
              file=sys.stderr)
        return 2

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)
    recognizer = engine.text_rec
    _, rec_height, rec_width = recognizer.rec_image_shape

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "checkpoint.jsonl"
    state = load_checkpoint(checkpoint, unit_field="index")
    log_line(f"safety stress: {TOTAL_CASES:,} fixed cases, have "
             f"{len(state.rows)} (resume #{state.resume_count}) pid {os.getpid()}")

    rows = list(state.rows)
    started = time.time()
    with WriterLock(checkpoint):
        with CheckpointWriter(checkpoint, state.digests, flush_every=100) as writer:
            for index, case in resumable_units(TOTAL_CASES, StressCase,
                                               state.completed_units):
                row = evaluate(case, args.font_dir, engine, recognizer, labels,
                               rec_height, rec_width)
                row["diagnostic_flags"] = derive_flags(row)
                writer.append(row)
                rows.append(row)
                if (index + 1) % args.progress_every == 0:
                    writer.sync()
                    log_line(f"  {index + 1}/{TOTAL_CASES} "
                             f"{time.time() - started:.0f}s")
        duplicates = writer.duplicates_rejected

    by_condition = defaultdict(Counter)
    for row in rows:
        entry = by_condition[row["condition"]]
        entry["cases"] += 1
        entry[row["terminal_reason"]] += 1
        if row["clean_preservation"]:
            entry["clean_preservation"] += 1
        if row["clean_hallucination"]:
            entry["clean_hallucination"] += 1
        if row["measured_stratum"]:
            entry[f"stratum_{row['measured_stratum']}"] += 1

    complete = {name: by_condition[name]["cases"] == entry["cases"]
                for name, entry in CONDITIONS.items()}
    manifest = {
        "dataset": "line_geometry_safety_stress_v2",
        "role": "false_correction_safety_evaluation_only",
        "recipe_sha256": hashlib.sha256(
            (args.out_dir / "recipe.json").read_bytes()).hexdigest()
        if (args.out_dir / "recipe.json").is_file() else None,
        "declared_cases": TOTAL_CASES,
        "generated_cases": len(rows),
        "all_conditions_complete": all(complete.values()),
        "per_condition": {k: dict(v) for k, v in sorted(by_condition.items())},
        "condition_complete": complete,
        "terminal_reasons": summarise_terminal_reasons(rows),
        "duplicate_digests_rejected": duplicates,
        "unique_digests": len({r["row_digest"] for r in rows}),
        "resume_count": state.resume_count,
        "wall_time_seconds": round(time.time() - started, 1),
        "prohibited": [
            "counting toward any natural hallucination quota",
            "pooling with the quota splits for rate estimation",
            "Expected text or product UI captures",
            "i/l confusion, out of scope for this verifier",
        ],
    }
    digest = atomic_write_json(args.out_dir / "manifest.json", manifest)

    log_line(f"\n{'condition':30s} {'cases':>6s} {'preserved':>10s} {'halluc':>7s}")
    for name in CONDITIONS:
        entry = by_condition[name]
        log_line(f"{name:30s} {entry['cases']:6d} "
                 f"{entry['clean_preservation']:10d} "
                 f"{entry['clean_hallucination']:7d}")
    log_line(f"total {len(rows):,}, complete {manifest['all_conditions_complete']}, "
             f"duplicates {duplicates}")
    log_line(f"manifest sha256 {digest}")
    return 0 if manifest["all_conditions_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
