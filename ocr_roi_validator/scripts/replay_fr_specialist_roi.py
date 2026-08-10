"""Replay a saved ROI through the French baseline and specialist recognizers.

Fidelity
--------
Three input modes, in descending order of trustworthiness:

``--ocr-input roi_ocr_input.png``
    The pixels the GUI recorded at the moment it called ``OCREngine.run()``.
    Fed to the recognizers verbatim -- no margin, padding or upscaling is
    applied, because all of that already happened before the image was
    recorded. Tagged ``exact_recorded_ocr_input``. This is the only mode that
    reproduces a real run.

``--source-image`` + ``--rect``
    Recomputes the crop from a full screen image through the shared
    :mod:`ocr_roi_validator.roi_preprocess` helpers. Faithful to the current
    preprocessing settings, but it is a reconstruction: nothing proves the
    original run used these same settings. Tagged
    ``reconstructed_runtime_input``.

``--roi roi.png``
    A saved failure diagnostic stores ``source_image.crop(roi.rect)`` -- the
    ROI rect only. The GUI feeds the recognizer a *margin-expanded* crop
    (default 8px per side), and those margin pixels are outside the saved PNG
    and unrecoverable from it. Tagged ``approximate_saved_roi``.

This tool never guesses or synthesises the missing margin.

Perturbations
-------------
Perturbations are split into two classes and must not be pooled into one gate:

* ``runtime_plausible`` -- 1px crops and small padding changes, which a real
  capture can produce through rounding and window movement.
* ``exploratory_stress`` -- 2x resampling, blur and contrast changes, which
  probe robustness but do not correspond to what the runtime actually feeds
  the recognizer.

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
from ocr_roi_validator.model_package import load_model_package  # noqa: E402
from ocr_roi_validator.ocr_engine import OCREngine  # noqa: E402
from ocr_roi_validator.roi_preprocess import (  # noqa: E402
    RoiPreprocessConfig,
    crop_roi,
    pad_long_roi,
    resize_short_side,
)

RUNTIME_PLAUSIBLE = (
    "none",
    "crop_left_1px",
    "crop_right_1px",
    "crop_top_1px",
    "crop_bottom_1px",
    "pad_all_1px",
    "crop_all_1px",
)
EXPLORATORY_STRESS = (
    "resize_bicubic_2x",
    "resize_lanczos_2x",
    "blur_gaussian_0p4",
    "contrast_0p85",
    "contrast_1p15",
)
PERTURBATION_CLASS = {
    **{name: "runtime_plausible" for name in RUNTIME_PLAUSIBLE},
    **{name: "exploratory_stress" for name in EXPLORATORY_STRESS},
}


def perturb(image: Image.Image, kind: str) -> Image.Image:
    """Apply one perturbation. Names describe the actual operation."""
    width, height = image.size
    if kind == "none":
        return image
    if kind == "crop_left_1px":
        return image.crop((1, 0, width, height))
    if kind == "crop_right_1px":
        return image.crop((0, 0, width - 1, height))
    if kind == "crop_top_1px":
        return image.crop((0, 1, width, height))
    if kind == "crop_bottom_1px":
        return image.crop((0, 0, width, height - 1))
    if kind == "pad_all_1px":
        padded = Image.new(image.mode, (width + 2, height + 2), image.getpixel((0, 0)))
        padded.paste(image, (1, 1))
        return padded
    if kind == "crop_all_1px":
        # Previously mislabelled "pad_minus_1": this removes a 1px border.
        return image.crop((1, 1, width - 1, height - 1))
    if kind == "resize_bicubic_2x":
        return image.resize((width * 2, height * 2), Image.Resampling.BICUBIC)
    if kind == "resize_lanczos_2x":
        return image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
    if kind == "blur_gaussian_0p4":
        return image.filter(ImageFilter.GaussianBlur(radius=0.4))
    if kind == "contrast_0p85":
        return ImageEnhance.Contrast(image).enhance(0.85)
    if kind == "contrast_1p15":
        return ImageEnhance.Contrast(image).enhance(1.15)
    raise ValueError(f"unknown perturbation: {kind}")


def build_engine(package_dir: Path, language: str, recognizer: Path) -> OCREngine:
    package = load_model_package(package_dir)
    recognizers = dict(package.recognizers)
    recognizers[language] = recognizer.resolve()
    return OCREngine(package=replace(package, recognizers=recognizers), backend="rapid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("input (choose one)")
    source.add_argument(
        "--ocr-input",
        type=Path,
        help="roi_ocr_input.png recorded at the OCR call site (exact replay)",
    )
    source.add_argument("--roi", type=Path, help="saved roi.png (approximate replay)")
    source.add_argument(
        "--source-image",
        type=Path,
        help="full screen image (reconstructed replay, with --rect)",
    )
    source.add_argument(
        "--rect", help="x1,y1,x2,y2 ROI rect within --source-image"
    )
    parser.add_argument("--metadata", type=Path, help="optional metadata.json for reporting")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--specialist-model", type=Path, required=True)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--min-side", type=int, default=160)
    parser.add_argument("--no-pad-long-roi", action="store_true")
    parser.add_argument("--no-auto-upscale", action="store_true")
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument(
        "--stress",
        action="store_true",
        help="also run exploratory stress perturbations (not runtime-representative)",
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if args.ocr_input:
        fidelity = "exact_recorded_ocr_input"
    elif args.source_image and args.rect:
        fidelity = "reconstructed_runtime_input"
    elif args.roi:
        fidelity = "approximate_saved_roi"
    else:
        parser.error(
            "provide --ocr-input, or --roi, or --source-image together with --rect"
        )

    config = RoiPreprocessConfig(
        margin=args.margin,
        min_side=args.min_side,
        pad_long_roi=not args.no_pad_long_roi,
        auto_upscale=not args.no_auto_upscale,
    )

    expected = None
    if args.metadata and args.metadata.is_file():
        expected = json.loads(args.metadata.read_text(encoding="utf-8")).get("expected")

    if fidelity == "exact_recorded_ocr_input":
        with Image.open(args.ocr_input) as handle:
            base_image = handle.convert("RGB")
        rect = None
        roi_path = str(args.ocr_input)
    elif fidelity == "reconstructed_runtime_input":
        with Image.open(args.source_image) as handle:
            full_image = handle.convert("RGB")
        rect = tuple(int(v) for v in args.rect.split(","))
        if len(rect) != 4:
            parser.error("--rect must be x1,y1,x2,y2")
        base_image = full_image
        roi_path = f"{args.source_image}#{args.rect}"
    else:
        with Image.open(args.roi) as handle:
            base_image = handle.convert("RGB")
        rect = None
        roi_path = str(args.roi)
        print(
            "NOTE: replaying a saved roi.png. The original OCR input included "
            f"{config.margin}px of margin outside this crop, which the saved file "
            "does not contain. Results are approximate, not an exact runtime replay.",
            file=sys.stderr,
        )

    original_size = list(base_image.size)

    baseline_engine = build_engine(args.package, args.language, args.baseline_model)
    specialist_engine = build_engine(args.package, args.language, args.specialist_model)

    kinds: tuple[str, ...] = ("none",)
    if args.perturbations:
        kinds = RUNTIME_PLAUSIBLE + (EXPLORATORY_STRESS if args.stress else ())

    records = []
    for kind in kinds:
        perturbed = perturb(base_image, kind)
        perturbed_size = list(perturbed.size)

        if fidelity == "exact_recorded_ocr_input":
            # Already the recognizer's input: margin, padding and upscaling all
            # happened before it was recorded. Re-applying them would corrupt it.
            ocr_input = perturbed
        elif fidelity == "reconstructed_runtime_input":
            ocr_input = crop_roi(perturbed, rect, config)
        else:
            # No surrounding pixels exist, so the margin step cannot be applied.
            ocr_input = perturbed
            if config.pad_long_roi:
                ocr_input = pad_long_roi(ocr_input)
            if config.auto_upscale:
                ocr_input = resize_short_side(ocr_input, config.min_side)
        ocr_input_size = list(ocr_input.size)

        baseline_result = baseline_engine.run(ocr_input, args.language)
        specialist_result = specialist_engine.run(ocr_input, args.language)
        # Expected text is deliberately NOT passed here.
        decision = route_specialist_text(baseline_result.text, specialist_result.text)

        record = {
            "label": args.label,
            "roi_path": roi_path,
            "replay_fidelity": fidelity,
            "perturbation": kind,
            "perturbation_class": PERTURBATION_CLASS[kind],
            "original_size": original_size,
            "perturbed_size": perturbed_size,
            "ocr_input_size": ocr_input_size,
            "preprocess": (
                None if fidelity == "exact_recorded_ocr_input" else config.as_dict()
            ),
            "preprocess_reapplied": fidelity != "exact_recorded_ocr_input",
            "margin_applied": fidelity == "reconstructed_runtime_input",
            "baseline": baseline_result.text,
            "specialist": specialist_result.text,
            "final": decision.final_text,
            "route": decision.route,
            "specialist_applied": decision.specialist_applied,
            "baseline_score": baseline_result.mean_score,
            "specialist_score": specialist_result.mean_score,
            "baseline_boxes": baseline_result.n_boxes,
            "specialist_boxes": specialist_result.n_boxes,
        }
        if expected is not None:
            # Post-OCR comparison only; played no part in the routing above.
            record["expected"] = expected
            record["final_matches_expected"] = decision.final_text == expected
        records.append(record)

        print(f"--- perturbation={kind} [{PERTURBATION_CLASS[kind]}]")
        print(f"ROI_PATH={roi_path}")
        print(f"REPLAY_FIDELITY={fidelity}")
        print(f"ORIGINAL_SIZE={original_size[0]}x{original_size[1]}")
        print(f"PERTURBED_SIZE={perturbed_size[0]}x{perturbed_size[1]}")
        print(f"OCR_INPUT_SIZE={ocr_input_size[0]}x{ocr_input_size[1]}")
        print(f"BASELINE={baseline_result.text!r}")
        print(f"SPECIALIST={specialist_result.text!r}")
        print(f"FINAL={decision.final_text!r}")
        print(f"ROUTE={decision.route}")
        print(f"SPECIALIST_APPLIED={'Y' if decision.specialist_applied else 'N'}")
        print(f"BASELINE_SCORE={baseline_result.mean_score}")
        print(f"SPECIALIST_SCORE={specialist_result.mean_score}")
        print()

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "replay_fidelity": fidelity,
                    "preprocess": config.as_dict(),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
