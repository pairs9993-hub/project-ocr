"""Compare batched vs per-line recognition on real screenshots.

``rec_batch_num=1`` removes the padding-context dependence by giving every line
its own tensor width. That fixes the known ``l'eau`` loss, but the same padding
sensitivity that causes the loss could equally well flip other lines the other
way, so the change has to be measured on real screens before it can be
considered a fix rather than a coincidence.

Both modes are run against the same images with the same detector, so detector
output is identical by construction and only the recognizer batching differs.

Ground truth is read only after both modes have produced their text, purely to
score them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VALIDATOR_ROOT.parent
for path in (str(VALIDATOR_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ocr_roi_validator.model_package import load_model_package  # noqa: E402
from ocr_validator.promotion_gate import cer as canonical_cer  # noqa: E402
from ocr_validator.promotion_gate import normalize_text, verdict  # noqa: E402


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
    return engine


def recognize(engine, bgr: np.ndarray, per_line: bool) -> tuple[list[str], int, float]:
    """Return (line texts in detector order, box count, elapsed seconds)."""
    recognizer = engine.text_rec
    previous = recognizer.rec_batch_num
    recognizer.rec_batch_num = 1 if per_line else previous
    try:
        started = time.perf_counter()
        boxes, _ = engine.auto_text_det(bgr)
        if boxes is None or len(boxes) == 0:
            return [], 0, time.perf_counter() - started
        crops = engine.get_crop_img_list(bgr, boxes)
        results, _ = recognizer(crops)
        elapsed = time.perf_counter() - started
        return [str(text) for text, _score in results], len(boxes), elapsed
    finally:
        recognizer.rec_batch_num = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True,
                        help="reference run providing image_path and reference")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    engine = build(args.package, args.language)

    records = []
    missing = 0
    batch_times: list[float] = []
    per_line_times: list[float] = []

    for index, row in enumerate(rows, 1):
        image_path = Path(row.get("image_path", ""))
        if not image_path.is_absolute():
            image_path = args.image_root / image_path
        if not image_path.is_file():
            missing += 1
            continue

        with Image.open(image_path) as handle:
            bgr = np.asarray(handle.convert("RGB"))[:, :, ::-1].copy()

        batch_lines, batch_boxes, batch_elapsed = recognize(engine, bgr, per_line=False)
        line_lines, line_boxes, line_elapsed = recognize(engine, bgr, per_line=True)
        batch_times.append(batch_elapsed)
        per_line_times.append(line_elapsed)

        reference = str(row.get("reference", ""))
        batch_text = "\n".join(batch_lines)
        line_text = "\n".join(line_lines)

        records.append(
            {
                "filename": row.get("filename"),
                "reference": reference,
                "batch_text": batch_text,
                "per_line_text": line_text,
                "batch_boxes": batch_boxes,
                "per_line_boxes": line_boxes,
                "detector_identical": batch_boxes == line_boxes,
                "batch_raw_exact": batch_text == reference,
                "per_line_raw_exact": line_text == reference,
                "batch_canonical_exact": normalize_text(batch_text)
                == normalize_text(reference),
                "per_line_canonical_exact": normalize_text(line_text)
                == normalize_text(reference),
                "batch_cer": canonical_cer(reference, batch_text),
                "per_line_cer": canonical_cer(reference, line_text),
                "batch_verdict": verdict(canonical_cer(reference, batch_text)),
                "per_line_verdict": verdict(canonical_cer(reference, line_text)),
                "changed": batch_text != line_text,
                "batch_seconds": batch_elapsed,
                "per_line_seconds": line_elapsed,
            }
        )
        if index % 25 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summarize(prefix: str) -> dict:
        return {
            "raw_exact": sum(r[f"{prefix}_raw_exact"] for r in records),
            "canonical_exact": sum(r[f"{prefix}_canonical_exact"] for r in records),
            "mean_cer": statistics.fmean(r[f"{prefix}_cer"] for r in records)
            if records else 0.0,
            "pass": sum(r[f"{prefix}_verdict"] == "PASS" for r in records),
        }

    batch = summarize("batch")
    per_line = summarize("per_line")

    raw_regressions = [
        r["filename"] for r in records
        if r["batch_raw_exact"] and not r["per_line_raw_exact"]
    ]
    canonical_regressions = [
        r["filename"] for r in records
        if r["batch_canonical_exact"] and not r["per_line_canonical_exact"]
    ]
    pass_regressions = [
        r["filename"] for r in records
        if r["batch_verdict"] == "PASS" and r["per_line_verdict"] != "PASS"
    ]
    detector_changed = [r["filename"] for r in records if not r["detector_identical"]]

    print(f"\nrows evaluated: {len(records)}  (missing images: {missing})")
    print(f"{'':12s} {'rawEx':>7s} {'canEx':>7s} {'PASS':>6s} {'meanCER':>10s}")
    print(f"{'batch':12s} {batch['raw_exact']:>7d} {batch['canonical_exact']:>7d} "
          f"{batch['pass']:>6d} {batch['mean_cer']:>10.6f}")
    print(f"{'per-line':12s} {per_line['raw_exact']:>7d} "
          f"{per_line['canonical_exact']:>7d} {per_line['pass']:>6d} "
          f"{per_line['mean_cer']:>10.6f}")
    print(f"\nrows whose text changed : {sum(r['changed'] for r in records)}")
    print(f"raw exact regressions   : {len(raw_regressions)} {raw_regressions[:8]}")
    print(f"canonical regressions   : {len(canonical_regressions)} "
          f"{canonical_regressions[:8]}")
    print(f"PASS regressions        : {len(pass_regressions)} {pass_regressions[:8]}")
    print(f"detector output changed : {len(detector_changed)}")

    if batch_times:
        def p95(values: list[float]) -> float:
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

        print(f"\nCPU timing over {len(batch_times)} images")
        print(f"  batch    mean {statistics.fmean(batch_times) * 1000:8.1f} ms  "
              f"p95 {p95(batch_times) * 1000:8.1f} ms")
        print(f"  per-line mean {statistics.fmean(per_line_times) * 1000:8.1f} ms  "
              f"p95 {p95(per_line_times) * 1000:8.1f} ms")
        print(f"  slowdown factor (mean): "
              f"{statistics.fmean(per_line_times) / statistics.fmean(batch_times):.2f}x")

    gates = {
        "no_raw_exact_regressions": not raw_regressions,
        "no_canonical_exact_regressions": not canonical_regressions,
        "no_pass_regressions": not pass_regressions,
        "mean_cer_not_worse": per_line["mean_cer"] <= batch["mean_cer"],
        "detector_unchanged": not detector_changed,
    }
    for gate, ok in gates.items():
        print(f"  gate {gate}: {'PASS' if ok else 'FAIL'}")
    passed = all(gates.values())
    print(f"L_BATCH_FIX_STATUS = {'PASS' if passed else 'FAIL'}")
    print(f"wrote {args.out_jsonl}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
