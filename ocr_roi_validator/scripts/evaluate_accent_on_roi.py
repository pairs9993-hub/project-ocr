"""Apply the frozen accent verifier to a real ROI (Gates F0 and F2).

Runs the product's own pipeline over an image, finds every ``é`` the baseline
predicted via CTC alignment, asks the verifier about each one from pixels, and
applies a correction only where the verifier is confidently sure there is no
accent.

The correction is a single-codepoint substitution in place, so the string length
cannot change and no character other than the judged ``é`` can move. Expected
text is read only afterwards, to score the result.

The frozen ``ACCENT_PREPROCESS_BASELINE`` batching is used throughout: this
measures the accent question against the product's current behaviour, and does
not attempt to fix the separate padding-context defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.accent_verifier import load_model, verify_accent_glyph  # noqa: E402
from ocr_roi_validator.model_package import load_model_package  # noqa: E402

PERTURBATIONS = (
    "none", "crop_left_1px", "crop_right_1px", "crop_top_1px",
    "crop_bottom_1px", "pad_all_1px", "crop_all_1px",
)


def perturb(image: Image.Image, kind: str) -> Image.Image:
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
        return image.crop((1, 1, width - 1, height - 1))
    raise ValueError(kind)


def build_engine(package_dir: Path, language: str):
    from rapidocr_onnxruntime import RapidOCR

    package = load_model_package(package_dir)
    p = package.preprocess
    engine = RapidOCR(
        det_model_path=str(package.detector_model),
        rec_model_path=str(package.recognizers[language]),
        rec_keys_path=str(package.dictionary),
        det_limit_type=str(p["det_limit_type"]),
        det_limit_side_len=int(p["det_limit_side_len"]),
        det_box_thresh=float(p["det_box_thresh"]),
        det_unclip_ratio=float(p["det_unclip_ratio"]),
        det_donot_use_dilation=bool(p["det_donot_use_dilation"]),
        use_cls=bool(p["use_cls"]),
    )
    return engine, package


def load_labels(dictionary: Path) -> list[str]:
    characters = dictionary.read_text(encoding="utf-8").split("\n")
    if characters and characters[-1] == "":
        characters = characters[:-1]
    return ["<blank>"] + characters + [" "]


def collapse_ctc(argmax, labels):
    emitted, previous = [], 0
    for timestep, index in enumerate(argmax):
        if index != 0 and index != previous:
            emitted.append(
                {"char": labels[index] if index < len(labels) else "?",
                 "start": timestep, "end": timestep}
            )
        elif emitted and index == previous and index != 0:
            emitted[-1]["end"] = timestep
        previous = index
    return emitted


def verify_lines(engine, package, labels, model, bgr: np.ndarray) -> dict:
    """Recognize an image and apply verifier-licensed accent removals."""
    boxes, _ = engine.auto_text_det(bgr)
    if boxes is None or len(boxes) == 0:
        return {"baseline_text": "", "final_text": "", "glyphs": []}

    crops = engine.get_crop_img_list(bgr, boxes)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape
    # ACCENT_PREPROCESS_BASELINE: the product's batched normalization.
    ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
    max_wh_ratio = max([width / height] + ratios)

    baseline_lines, final_lines, glyph_records = [], [], []
    for line_index, crop in enumerate(crops):
        crop_h, crop_w = crop.shape[:2]
        tensor = recognizer.resize_norm_img(crop, max_wh_ratio)[np.newaxis, :]
        logits = np.asarray(recognizer.session(tensor.astype(np.float32))[0])
        probabilities = logits[0]
        argmax = probabilities.argmax(axis=-1).tolist()
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[crop_w / float(crop_h)],
            max_wh_ratio=max_wh_ratio,
        )[0][0]
        baseline_lines.append(decoded)

        emitted = collapse_ctc(argmax, labels)
        if "".join(item["char"] for item in emitted) != decoded:
            # Alignment untrustworthy: keep the baseline line untouched.
            final_lines.append(decoded)
            continue

        padded_w = int(height * max_wh_ratio)
        resized_w = min(padded_w, int(np.ceil(height * (crop_w / crop_h))))
        timesteps = probabilities.shape[0]
        scale = (padded_w / timesteps) * (crop_w / resized_w)

        characters = list(decoded)
        for position, item in enumerate(emitted):
            if unicodedata.normalize("NFC", item["char"]) != "é":
                continue
            x0 = max(0, int(np.floor((item["start"] + 0.5) * scale)) - 4)
            x1 = min(crop_w, int(np.ceil((item["end"] + 1 + 0.5) * scale)) + 4)
            glyph = crop[:, x0:x1]
            result = verify_accent_glyph(glyph, model)
            applied = result.is_accent_absent and position < len(characters)
            if applied:
                # Single-codepoint substitution: length is preserved and no
                # other character is touched.
                characters[position] = "e"
            glyph_records.append(
                {
                    "line_index": line_index,
                    "position": position,
                    "x0": x0, "x1": x1,
                    "verdict": result.verdict,
                    "probability_absent": result.probability_absent,
                    "reason": result.reason,
                    "applied": applied,
                }
            )
        final_lines.append("".join(characters))

    return {
        "baseline_text": "\n".join(baseline_lines),
        "final_text": "\n".join(final_lines),
        "glyphs": glyph_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    model = load_model(args.model)
    if model is None:
        print(f"no model at {args.model}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()

    engine, package = build_engine(args.package, args.language)
    labels = load_labels(package.dictionary)

    expected = None
    if args.metadata and args.metadata.is_file():
        expected = json.loads(args.metadata.read_text(encoding="utf-8")).get("expected")

    with Image.open(args.image) as handle:
        base = handle.convert("RGB")

    kinds = PERTURBATIONS if args.perturbations else ("none",)
    records = []
    for kind in kinds:
        image = perturb(base, kind)
        bgr = np.asarray(image)[:, :, ::-1].copy()
        outcome = verify_lines(engine, package, labels, model, bgr)
        record = {"perturbation": kind, **outcome}
        if expected is not None:
            record["baseline_matches_expected"] = outcome["baseline_text"] == expected
            record["final_matches_expected"] = outcome["final_text"] == expected
        records.append(record)

        print(f"--- {kind}")
        print(f"  baseline : {outcome['baseline_text']!r}")
        print(f"  final    : {outcome['final_text']!r}")
        for glyph in outcome["glyphs"]:
            print(f"    line {glyph['line_index']} pos {glyph['position']:>3d} "
                  f"{glyph['verdict']:>8s} p={glyph['probability_absent']:.4f} "
                  f"applied={glyph['applied']}")

    print(f"\nmodel {model.version} sha256 {digest[:16]}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {"model_sha256": digest, "image": str(args.image),
                 "records": records},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
