from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Paddle OCR model package")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--det-infer", type=Path, required=True)
    parser.add_argument("--rec-en-es-infer", type=Path, required=True)
    parser.add_argument("--rec-fr-infer", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--rec-zh-infer", type=Path, default=None)
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--package-name", type=str, default="paddle_shared_det_multirec")
    return parser.parse_args()


def _copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _validate_infer_dir(path: Path, name: str):
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not (path / "inference.pdmodel").exists() or not (path / "inference.pdiparams").exists():
        raise FileNotFoundError(f"{name} must contain inference.pdmodel and inference.pdiparams: {path}")


def main() -> int:
    args = parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    det_src = args.det_infer.resolve()
    rec_en_es_src = args.rec_en_es_infer.resolve()
    rec_fr_src = args.rec_fr_infer.resolve()
    rec_zh_src = args.rec_zh_infer.resolve() if args.rec_zh_infer else None
    dict_src = args.dict.resolve()

    _validate_infer_dir(det_src, "det-infer")
    _validate_infer_dir(rec_en_es_src, "rec-en-es-infer")
    _validate_infer_dir(rec_fr_src, "rec-fr-infer")
    if rec_zh_src is not None:
        _validate_infer_dir(rec_zh_src, "rec-zh-infer")
    if not dict_src.exists() or not dict_src.is_file():
        raise FileNotFoundError(f"dict not found: {dict_src}")

    det_dst = out / "detector_infer"
    rec_en_es_dst = out / "recognizers" / "rec_en_es_infer"
    rec_fr_dst = out / "recognizers" / "rec_fr_infer"
    dict_dst = out / "dict" / "ppocr_keys.txt"

    _copytree(det_src, det_dst)
    _copytree(rec_en_es_src, rec_en_es_dst)
    _copytree(rec_fr_src, rec_fr_dst)
    dict_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dict_src, dict_dst)

    recognizer_model_dirs = {
        "en_es": "recognizers/rec_en_es_infer",
        "fr": "recognizers/rec_fr_infer",
    }
    if rec_zh_src is not None:
        rec_zh_dst = out / "recognizers" / "rec_zh_infer"
        _copytree(rec_zh_src, rec_zh_dst)
        recognizer_model_dirs["zh"] = "recognizers/rec_zh_infer"

    manifest = {
        "package_name": args.package_name,
        "version": args.version,
        "detector_model_dir": "detector_infer",
        "dictionary": "dict/ppocr_keys.txt",
        "recognizer_model_dirs": recognizer_model_dirs,
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Paddle model package created at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
