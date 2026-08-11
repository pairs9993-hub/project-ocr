"""Determine why a line decodes differently alone than inside a batch.

RapidOCR normalizes every line in a batch to ``max_wh_ratio``, the widest
aspect ratio in that batch. Two mechanisms could follow from that, and they
call for different fixes:

* the line's pixels are horizontally squeezed (a resize difference), or
* the line's pixels are untouched and only the zero padding to its right
  grows (a padding-context effect).

``resize_norm_img`` computes ``resized_w`` from the line's own aspect ratio and
only clips it when it would exceed the padded width, so the second mechanism is
the one to expect -- but that has to be measured, not assumed. This tool
compares the content region of the tensors byte for byte across batch
compositions and reports which mechanism is actually at work.

Diagnostic only. Ground truth is never read; nothing here runs at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.model_package import load_model_package  # noqa: E402

CAUSE_RESIZE = "HORIZONTAL_RESIZE_DIFFERENCE"
CAUSE_PADDING = "RIGHT_PADDING_CONTEXT_EFFECT"
CAUSE_ORDER = "BATCH_ORDER_OR_POSTPROCESS_BUG"
CAUSE_NUMERIC = "NUMERICAL_BATCH_DEPENDENCE"
CAUSE_OTHER = "OTHER"
CAUSE_UNPROVEN = "UNPROVEN"


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build(package_dir: Path, language: str):
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


def run_one(recognizer, crop: np.ndarray, max_wh_ratio: float) -> dict:
    """Normalize and decode a single crop at a chosen max_wh_ratio."""
    channels, height, width = recognizer.rec_image_shape
    padded_width = int(height * max_wh_ratio)
    crop_h, crop_w = crop.shape[:2]
    ratio = crop_w / float(crop_h)
    resized_w = (
        padded_width
        if math.ceil(height * ratio) > padded_width
        else int(math.ceil(height * ratio))
    )

    tensor = recognizer.resize_norm_img(crop, max_wh_ratio)
    batch = tensor[np.newaxis, :].astype(np.float32)
    logits = np.asarray(recognizer.session(batch)[0])
    decoded = recognizer.postprocess_op(
        logits, False, wh_ratio_list=[ratio], max_wh_ratio=max_wh_ratio
    )
    content = tensor[:, :, :resized_w]
    return {
        "max_wh_ratio": max_wh_ratio,
        "padded_width": padded_width,
        "resized_w": resized_w,
        "clipped": math.ceil(height * ratio) > padded_width,
        "tensor_shape": list(tensor.shape),
        "tensor_sha256": digest(tensor),
        "content_shape": list(content.shape),
        "content_sha256": digest(content),
        "logits_shape": list(logits.shape),
        "decoded": decoded[0][0],
        "score": float(decoded[0][1]),
        "_content": content,
        "_logits": logits,
    }


def controlled_padding_sweep(recognizer, crop: np.ndarray) -> list[dict]:
    """Decode one line at several padding widths with identical content pixels.

    ``resize_norm_img`` couples resize width and padding width, so this bypasses
    it: resize once, then place those exact pixels into canvases of different
    widths. Any change in the decode is attributable to padding alone.
    """
    import cv2

    channels, height, _ = recognizer.rec_image_shape
    crop_h, crop_w = crop.shape[:2]
    ratio = crop_w / float(crop_h)
    resized_w = int(math.ceil(height * ratio))

    resized = cv2.resize(crop, (resized_w, height)).astype("float32")
    resized = resized.transpose((2, 0, 1)) / 255
    resized = (resized - 0.5) / 0.5

    rows = []
    for padded_width in (resized_w, resized_w + 11, resized_w + 59, resized_w + 60,
                         resized_w + 71, resized_w + 101, resized_w + 288):
        tensor = np.zeros((channels, height, padded_width), dtype=np.float32)
        tensor[:, :, :resized_w] = resized
        logits = np.asarray(recognizer.session(tensor[np.newaxis, :])[0])
        decoded = recognizer.postprocess_op(
            logits, False, wh_ratio_list=[ratio],
            max_wh_ratio=padded_width / height,
        )
        rows.append(
            {
                "padded_width": padded_width,
                "resized_w": resized_w,
                "content_sha256": digest(tensor[:, :, :resized_w]),
                "timesteps": int(logits.shape[1]),
                "decoded": decoded[0][0],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-input", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--line-index", type=int, default=1)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    with Image.open(args.ocr_input) as handle:
        bgr = np.asarray(handle.convert("RGB"))[:, :, ::-1].copy()

    engine, package = build(args.package, args.language)
    boxes, _ = engine.auto_text_det(bgr)
    crops = engine.get_crop_img_list(bgr, boxes)
    recognizer = engine.text_rec
    channels, height, width = recognizer.rec_image_shape
    default_ratio = width / height

    target = crops[args.line_index]
    target_ratio = target.shape[1] / float(target.shape[0])
    ratios = [c.shape[1] / float(c.shape[0]) for c in crops]

    print(f"rapidocr rec_image_shape : {recognizer.rec_image_shape}")
    print(f"rec_batch_num            : {recognizer.rec_batch_num}")
    print(f"line ratios              : {[round(r, 3) for r in ratios]}")
    print(f"target line {args.line_index} ratio     : {target_ratio:.3f}\n")

    # Conditions differ only in the max_wh_ratio the target line is given.
    conditions = {
        "alone": max(default_ratio, target_ratio),
        "original_batch": max([default_ratio] + ratios),
        "reordered_batch": max([default_ratio] + list(reversed(ratios))),
        "with_shorter_line": max(default_ratio, target_ratio, min(ratios)),
        "with_wider_line": max(default_ratio, target_ratio, max(ratios) * 1.5),
        "padding_plus_one": max([default_ratio] + ratios) + (1.0 / height),
    }

    results = {}
    for name, ratio in conditions.items():
        results[name] = run_one(recognizer, target, ratio)
        r = results[name]
        print(
            f"{name:20s} maxR={r['max_wh_ratio']:7.3f} "
            f"padW={r['padded_width']:4d} resW={r['resized_w']:4d} "
            f"clip={str(r['clipped']):5s} content={r['content_sha256'][:10]} "
            f"-> {r['decoded']!r}"
        )

    alone = results["alone"]
    batch = results["original_batch"]

    same_content = alone["content_sha256"] == batch["content_sha256"]
    same_resized = alone["resized_w"] == batch["resized_w"]
    same_decode = alone["decoded"] == batch["decoded"]
    padding_differs = alone["padded_width"] != batch["padded_width"]

    # Compare the shared content region numerically, in case digests differ
    # only by a rounding hair.
    overlap = min(alone["_content"].shape[2], batch["_content"].shape[2])
    max_abs_diff = float(
        np.abs(
            alone["_content"][:, :, :overlap] - batch["_content"][:, :, :overlap]
        ).max()
    )

    print(f"\ncontent identical      : {same_content}")
    print(f"resized_w identical    : {same_resized} "
          f"({alone['resized_w']} vs {batch['resized_w']})")
    print(f"padded width differs   : {padding_differs} "
          f"({alone['padded_width']} vs {batch['padded_width']})")
    print(f"max |content diff|     : {max_abs_diff:.3e}")
    print(f"decoded identical      : {same_decode}")
    print(f"  alone : {alone['decoded']!r}")
    print(f"  batch : {batch['decoded']!r}")

    by_padding: dict[int, set[str]] = {}
    for name, r in results.items():
        by_padding.setdefault(r["padded_width"], set()).add(r["decoded"])
    padding_determines = all(len(v) == 1 for v in by_padding.values())
    print(f"\ndecode determined by padded width alone: {padding_determines}")
    for pad_width, decodes in sorted(by_padding.items()):
        print(f"  padW={pad_width:4d} -> {sorted(decodes)}")

    # The decisive experiment: hold the content region byte-identical and vary
    # only the zero padding. If the decode still changes, the cause is padding
    # context, no matter what the resize widths happened to be.
    controlled = controlled_padding_sweep(recognizer, target)
    content_digests = {r["content_sha256"] for r in controlled}
    decodes = {r["decoded"] for r in controlled}
    print("\ncontrolled sweep (identical content, padding varied):")
    for r in controlled:
        print(f"  padW={r['padded_width']:4d} T={r['timesteps']:3d} -> {r['decoded']!r}")
    print(f"  distinct content digests: {len(content_digests)}")
    print(f"  distinct decodes        : {len(decodes)}")

    if same_decode:
        cause = CAUSE_UNPROVEN
        note = "the line decodes the same in both conditions here"
    elif len(content_digests) == 1 and len(decodes) > 1:
        cause = CAUSE_PADDING
        note = (
            "with the content region held byte-identical, varying only the "
            "zero padding changes the decode"
        )
    elif not same_resized or (not same_content and max_abs_diff > 1e-6):
        cause = CAUSE_RESIZE
        note = "the line's own pixels differ between conditions"
    elif not padding_determines:
        cause = CAUSE_NUMERIC
        note = "same padded width produced different decodes"
    else:
        cause = CAUSE_OTHER
        note = "content and padding both identical but decode differs"

    print(f"\nL_BATCH_ROOT_CAUSE = {cause}")
    print(f"  {note}")

    if args.out_json:
        payload = {
            "rapidocr_rec_image_shape": recognizer.rec_image_shape,
            "line_index": args.line_index,
            "line_ratios": ratios,
            "conditions": {
                name: {k: v for k, v in r.items() if not k.startswith("_")}
                for name, r in results.items()
            },
            "content_identical_alone_vs_batch": same_content,
            "max_abs_content_diff": max_abs_diff,
            "decode_determined_by_padding_width": padding_determines,
            "root_cause": cause,
            "note": note,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
