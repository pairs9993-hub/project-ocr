from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR model package")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--rec-en-es", type=Path, required=True)
    parser.add_argument("--rec-fr", type=Path, required=True)
    parser.add_argument("--rec-zh", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--package-name", type=str, default="ocr_shared_det_multirec")
    parser.add_argument("--det-limit-type", type=str, default="min")
    parser.add_argument("--det-limit-side-len", type=int, default=640)
    parser.add_argument("--det-box-thresh", type=float, default=0.5)
    parser.add_argument("--det-unclip-ratio", type=float, default=3.5)
    parser.add_argument("--det-no-dilation", action="store_true", default=True)
    parser.add_argument("--use-cls", action="store_true", default=False)
    return parser.parse_args()


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    detector_dst = out / "detector" / "det.onnx"
    rec_en_es_dst = out / "recognizers" / "rec_en_es.onnx"
    rec_fr_dst = out / "recognizers" / "rec_fr.onnx"
    rec_zh_dst = out / "recognizers" / "rec_zh.onnx"
    dict_dst = out / "dict" / "ppocr_keys.txt"

    _copy(args.detector.resolve(), detector_dst)
    _copy(args.rec_en_es.resolve(), rec_en_es_dst)
    _copy(args.rec_fr.resolve(), rec_fr_dst)
    _copy(args.rec_zh.resolve(), rec_zh_dst)
    _copy(args.dict.resolve(), dict_dst)

    manifest = {
        "package_name": args.package_name,
        "version": args.version,
        "detector_model": "detector/det.onnx",
        "dictionary": "dict/ppocr_keys.txt",
        "recognizers": {
            "en_es": "recognizers/rec_en_es.onnx",
            "fr": "recognizers/rec_fr.onnx",
            "zh": "recognizers/rec_zh.onnx",
        },
        "preprocess": {
            "det_limit_type": args.det_limit_type,
            "det_limit_side_len": args.det_limit_side_len,
            "det_mean": [0.485, 0.456, 0.406],
            "det_std": [0.229, 0.224, 0.225],
            "det_box_thresh": args.det_box_thresh,
            "det_unclip_ratio": args.det_unclip_ratio,
            "det_donot_use_dilation": args.det_no_dilation,
            "use_cls": args.use_cls,
        },
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Model package created at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
