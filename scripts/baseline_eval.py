"""Run RapidOCR pretrained baseline on the test set, optionally with preprocessing.

Outputs a per-sample JSONL at artifacts/baseline/<mode>_results.jsonl with
fields: image_path, language, pattern, background, ref, hyp, cer, wer, elapsed_ms.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_validator.evaluate import cer, wer, normalize_text
from ocr_validator.ocr_engine import OCREngine
from ocr_validator.preprocess import PreprocessConfig, preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["raw", "preprocessed"], required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None,
                   help="optional cap on number of samples (for quick smoke runs)")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="directory containing rec.onnx and ppocr_keys.txt")
    args = p.parse_args()

    project = Path(__file__).resolve().parent.parent
    split_dir = project / "dataset" / args.split
    out_dir = project / "artifacts/baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = None
    rec_model_path = None
    rec_keys_path = None
    if args.model_dir is not None:
        model_dir = args.model_dir if args.model_dir.is_absolute() else project / args.model_dir
        model_dir = model_dir.resolve()
        rec_model_path = model_dir / "rec.onnx"
        rec_keys_path = model_dir / "ppocr_keys.txt"
        if not rec_model_path.exists():
            sys.exit(f"missing {rec_model_path}")
        if not rec_keys_path.exists():
            sys.exit(f"missing {rec_keys_path}")
        model_tag = model_dir.name
    out_name = f"{model_tag}_{args.mode}_results.jsonl" if model_tag else f"{args.mode}_results.jsonl"
    out_path = out_dir / out_name

    engine = OCREngine(rec_model_path=rec_model_path, rec_keys_path=rec_keys_path)
    cfg = PreprocessConfig() if args.mode == "preprocessed" else None

    n = 0
    total_t = 0.0
    start_global = time.time()
    with (split_dir / "labels.jsonl").open(encoding="utf-8") as f, \
         out_path.open("w", encoding="utf-8") as out_f:
        for line in f:
            d = json.loads(line)
            if args.limit and n >= args.limit:
                break
            img_path = split_dir / d["image_path"]
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if cfg is not None:
                proc = preprocess(bgr, cfg)
                # OCR engine expects BGR; if grayscale, expand
                if proc.ndim == 2:
                    proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
                ocr_input = proc
            else:
                ocr_input = bgr

            t0 = time.time()
            boxes = engine.recognize(ocr_input)
            elapsed = (time.time() - t0) * 1000.0
            total_t += elapsed

            hyp = OCREngine.boxes_to_text(boxes)
            ref = d["raw_text"]
            row = {
                "image_path": d["image_path"],
                "model": model_tag or "pretrained",
                "language": d["language"],
                "pattern": d["pattern"],
                "background": d["background"],
                "ref": ref,
                "hyp": hyp,
                "cer": cer(ref, hyp),
                "wer": wer(ref, hyp),
                "elapsed_ms": elapsed,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                wall = time.time() - start_global
                eta = wall / n * (1500 - n) if n < 1500 else 0
                print(f"  {n} done, mean {total_t/n:.0f}ms/img, wall={wall:.0f}s, eta={eta:.0f}s",
                      flush=True)

    print(f"\nmodel={model_tag or 'pretrained'} mode={args.mode}: {n} samples -> {out_path}")
    print(f"mean inference time: {total_t/max(n,1):.1f} ms/img")
    print(f"total wall: {time.time()-start_global:.1f} s")


if __name__ == "__main__":
    main()
