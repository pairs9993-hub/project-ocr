"""Dump the exact per-line recognizer inputs and CTC logits for one ROI.

The saved ``roi_ocr_input.png`` is what ``OCREngine.run()`` received, but glyph
verification needs a level deeper: the rectified line crop each recognizer call
actually saw, the normalized tensor fed to the ONNX session, and the CTC
probabilities that produced the decoded text.

This reproduces the RapidOCR pipeline stage by stage against the same detector,
recognizer and dictionary, then checks that the reconstructed text matches what
the engine reported. If it does not, the dump is not trustworthy and the tool
fails rather than emitting misleading artifacts.

Diagnostic only. Nothing here runs during a normal run.bat session, and no
output is committed. Ground truth is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.model_package import load_model_package  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_rapidocr(package_dir: Path, language: str, recognizer: Path | None = None):
    """Construct RapidOCR with the same settings the runtime engine uses."""
    from rapidocr_onnxruntime import RapidOCR

    package = load_model_package(package_dir)
    preprocess = package.preprocess
    rec_path = recognizer or package.recognizers[language]
    engine = RapidOCR(
        det_model_path=str(package.detector_model),
        rec_model_path=str(rec_path),
        rec_keys_path=str(package.dictionary),
        det_limit_type=str(preprocess["det_limit_type"]),
        det_limit_side_len=int(preprocess["det_limit_side_len"]),
        det_box_thresh=float(preprocess["det_box_thresh"]),
        det_unclip_ratio=float(preprocess["det_unclip_ratio"]),
        det_donot_use_dilation=bool(preprocess["det_donot_use_dilation"]),
        use_cls=bool(preprocess["use_cls"]),
    )
    return engine, package, rec_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--recognizer", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    with Image.open(args.ocr_input) as handle:
        pil_image = handle.convert("RGB")
    bgr = np.asarray(pil_image)[:, :, ::-1].copy()

    engine, package, rec_path = build_rapidocr(
        args.package, args.language, args.recognizer
    )

    # Stage 1: detection, in the same sorted order the pipeline uses.
    boxes, _det_elapse = engine.auto_text_det(bgr)
    if boxes is None or len(boxes) == 0:
        print("detector found no text lines", file=sys.stderr)
        return 1

    # Stage 2: rectified line crops.
    crops = engine.get_crop_img_list(bgr, boxes)

    recognizer_module = engine.text_rec
    channels, height, _width = recognizer_module.rec_image_shape

    # Stage 3+4: normalize and run each line.
    #
    # The batch width comes from the widest line in the batch, not from each
    # line individually, so a line's tensor depends on its neighbours. To dump
    # what the recognizer really saw, use the batch-wide ratio; dumping each
    # line in isolation would produce different pixels and different text.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_ratio = _width / height
    batch_max_wh_ratio = max(
        [default_ratio] + [c.shape[1] / float(c.shape[0]) for c in crops]
    )

    lines = []
    for index, (box, crop) in enumerate(zip(boxes, crops)):
        crop_h, crop_w = crop.shape[:2]
        wh_ratio = crop_w / float(crop_h)
        max_wh_ratio = batch_max_wh_ratio
        tensor = recognizer_module.resize_norm_img(crop, max_wh_ratio)
        batch = tensor[np.newaxis, :].astype(np.float32)

        logits = recognizer_module.session(batch)[0]
        decoded = recognizer_module.postprocess_op(
            logits, False, wh_ratio_list=[wh_ratio], max_wh_ratio=max_wh_ratio
        )
        text, score = decoded[0][0], decoded[0][1]

        probabilities = np.asarray(logits)[0]
        argmax_indices = probabilities.argmax(axis=-1)
        argmax_confidence = probabilities.max(axis=-1)

        # Same line decoded on its own, to expose batch-width side effects.
        isolated_ratio = max(default_ratio, wh_ratio)
        isolated_tensor = recognizer_module.resize_norm_img(crop, isolated_ratio)[
            np.newaxis, :
        ].astype(np.float32)
        isolated_logits = recognizer_module.session(isolated_tensor)[0]
        isolated_decoded = recognizer_module.postprocess_op(
            isolated_logits, False, wh_ratio_list=[wh_ratio],
            max_wh_ratio=isolated_ratio,
        )

        record = {
            "line_index": index,
            "polygon": np.asarray(box).tolist(),
            "crop_size": [int(crop_w), int(crop_h)],
            "wh_ratio": wh_ratio,
            "batch_max_wh_ratio": batch_max_wh_ratio,
            "isolated_decoded_text": isolated_decoded[0][0],
            "isolated_differs_from_batch": isolated_decoded[0][0] != text,
            "crop_sha256": sha256_bytes(np.ascontiguousarray(crop).tobytes()),
            "tensor_shape": list(batch.shape),
            "tensor_sha256": sha256_bytes(np.ascontiguousarray(batch).tobytes()),
            "logits_shape": list(probabilities.shape),
            "logits_sha256": sha256_bytes(
                np.ascontiguousarray(probabilities).tobytes()
            ),
            "decoded_text": text,
            "decoded_score": float(score),
            "ctc_timesteps": int(probabilities.shape[0]),
            "ctc_argmax": argmax_indices.tolist(),
            "ctc_argmax_confidence": [float(v) for v in argmax_confidence],
        }
        lines.append(record)

        np.save(args.out_dir / f"line{index}_tensor.npy", batch)
        np.save(args.out_dir / f"line{index}_logits.npy", probabilities)
        if args.save_images:
            cv2.imwrite(str(args.out_dir / f"line{index}_crop.png"), crop)

    # Stage 5: reproduce the engine's own text and verify we match it.
    engine_result, _ = engine(bgr)
    engine_lines = [str(item[1]) for item in (engine_result or [])]
    dumped_lines = [line["decoded_text"] for line in lines]

    matches = engine_lines == dumped_lines
    print(f"detected lines: {len(lines)}")
    for line in lines:
        print(f"  line {line['line_index']}: {line['decoded_text']!r} "
              f"(score {line['decoded_score']:.4f}, "
              f"{line['ctc_timesteps']} timesteps)")
    print(f"\nengine lines : {engine_lines}")
    print(f"dumped lines : {dumped_lines}")
    print(f"per-line reproduction matches engine: {matches}")

    expected_full = None
    if args.metadata and args.metadata.is_file():
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        expected_full = metadata.get("ocr_raw_output")

    summary = {
        "ocr_input": str(args.ocr_input),
        "ocr_input_sha256": sha256_bytes(args.ocr_input.read_bytes()),
        "package": str(args.package),
        "detector_sha256": sha256_bytes(package.detector_model.read_bytes()),
        "dictionary_sha256": sha256_bytes(package.dictionary.read_bytes()),
        "recognizer_path": str(rec_path),
        "recognizer_sha256": sha256_bytes(Path(rec_path).read_bytes()),
        "rec_image_shape": [channels, height, _width],
        "lines": lines,
        "engine_lines": engine_lines,
        "per_line_matches_engine": matches,
        "recorded_ocr_raw_output": expected_full,
    }
    (args.out_dir / "line_dump.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out_dir / 'line_dump.json'}")

    if not matches:
        print(
            "ABORT: per-line reproduction does not match the engine output",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
