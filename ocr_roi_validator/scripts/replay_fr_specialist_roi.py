"""Replay a saved ROI through the French baseline and specialist recognizers.

Uses the same runtime path as ``run.bat``: the same detector, the same
dictionary, the same ``_pad_long_roi`` padding and the same RapidOCR
preprocessing, then routes the two outputs through
``route_specialist_text``.

Expected text is never given to OCR or to the router. It is read from the
failure metadata only so the report can show what a post-OCR exact comparison
would have concluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_roi_validator.fr_specialist_router import route_specialist_text  # noqa: E402
from ocr_roi_validator.gui import _pad_long_roi  # noqa: E402
from ocr_roi_validator.model_package import load_model_package  # noqa: E402
from ocr_roi_validator.ocr_engine import OCREngine  # noqa: E402


def perturb(image: Image.Image, kind: str) -> Image.Image:
    """Small rendering perturbations that must not flip a glyph decision."""
    width, height = image.size
    if kind == "none":
        return image
    if kind == "crop_left_1":
        return image.crop((1, 0, width, height))
    if kind == "crop_right_1":
        return image.crop((0, 0, width - 1, height))
    if kind == "crop_top_1":
        return image.crop((0, 1, width, height))
    if kind == "crop_bottom_1":
        return image.crop((0, 0, width, height - 1))
    if kind == "pad_1":
        padded = Image.new(image.mode, (width + 2, height + 2), image.getpixel((0, 0)))
        padded.paste(image, (1, 1))
        return padded
    if kind == "pad_minus_1":
        return image.crop((1, 1, width - 1, height - 1))
    if kind == "bicubic_2x":
        return image.resize((width * 2, height * 2), Image.Resampling.BICUBIC)
    if kind == "lanczos_2x":
        return image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=0.4))
    if kind == "contrast_down":
        return ImageEnhance.Contrast(image).enhance(0.85)
    if kind == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(1.15)
    raise ValueError(f"unknown perturbation: {kind}")


PERTURBATIONS = (
    "none",
    "crop_left_1",
    "crop_right_1",
    "crop_top_1",
    "crop_bottom_1",
    "pad_1",
    "pad_minus_1",
    "bicubic_2x",
    "lanczos_2x",
    "blur",
    "contrast_down",
    "contrast_up",
)


def build_engine(package_dir: Path, language: str, recognizer: Path) -> OCREngine:
    package = load_model_package(package_dir)
    recognizers = dict(package.recognizers)
    recognizers[language] = recognizer.resolve()
    return OCREngine(package=replace(package, recognizers=recognizers), backend="rapid")


def run_one(engine: OCREngine, image: Image.Image, language: str) -> tuple[str, float, int]:
    result = engine.run(image, language)
    return result.text, result.mean_score, result.n_boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, required=True, help="path to roi.png")
    parser.add_argument("--metadata", type=Path, help="optional metadata.json for reporting")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--specialist-model", type=Path, required=True)
    parser.add_argument("--no-pad-long-roi", action="store_true")
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    expected = None
    if args.metadata and args.metadata.is_file():
        expected = json.loads(args.metadata.read_text(encoding="utf-8")).get("expected")

    with Image.open(args.roi) as handle:
        original = handle.convert("RGB")
    input_size = list(original.size)

    baseline_engine = build_engine(args.package, args.language, args.baseline_model)
    specialist_engine = build_engine(args.package, args.language, args.specialist_model)

    kinds = PERTURBATIONS if args.perturbations else ("none",)
    records = []
    for kind in kinds:
        image = perturb(original, kind)
        if not args.no_pad_long_roi:
            image = _pad_long_roi(image)
        padded_size = list(image.size)

        baseline_text, baseline_score, baseline_boxes = run_one(
            baseline_engine, image, args.language
        )
        specialist_text, specialist_score, specialist_boxes = run_one(
            specialist_engine, image, args.language
        )
        # Expected text is deliberately NOT passed here.
        decision = route_specialist_text(baseline_text, specialist_text)

        record = {
            "label": args.label,
            "roi_path": str(args.roi),
            "perturbation": kind,
            "input_size": input_size,
            "padded_size": padded_size,
            "baseline": baseline_text,
            "specialist": specialist_text,
            "final": decision.final_text,
            "route": decision.route,
            "specialist_applied": decision.specialist_applied,
            "baseline_score": baseline_score,
            "specialist_score": specialist_score,
            "baseline_boxes": baseline_boxes,
            "specialist_boxes": specialist_boxes,
        }
        if expected is not None:
            # Post-OCR comparison only; played no part in the routing above.
            record["expected"] = expected
            record["final_matches_expected"] = decision.final_text == expected
        records.append(record)

        print(f"--- perturbation={kind}")
        print(f"ROI_PATH={args.roi}")
        print(f"INPUT_SIZE={input_size[0]}x{input_size[1]}")
        print(f"PADDED_SIZE={padded_size[0]}x{padded_size[1]}")
        print(f"BASELINE={baseline_text!r}")
        print(f"SPECIALIST={specialist_text!r}")
        print(f"FINAL={decision.final_text!r}")
        print(f"ROUTE={decision.route}")
        print(f"SPECIALIST_APPLIED={'Y' if decision.specialist_applied else 'N'}")
        print(f"BASELINE_SCORE={baseline_score}")
        print(f"SPECIALIST_SCORE={specialist_score}")
        print()

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
