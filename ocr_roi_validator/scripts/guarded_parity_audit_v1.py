"""Re-check runtime parity on the guarded deployment verdict.

Raw parity failed on one sample of 1,398: it sat 5.96e-08 from the frozen
threshold, and the runtimes disagree by up to 5.7e-06. The base threshold is
not moved. Instead the guard withholds any correction within 1e-4 of the
boundary, which is wider than the disagreement, so no numerical difference can
change a deployment decision.

Raw probabilities and raw verdicts are still reported -- the earlier failure is
preserved, not overwritten. What changes is which verdict the parity gate is
applied to.

The audit runs over every model-input row, not just the calibration slice, so a
boundary sample anywhere in the corpus would surface here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.line_verifier_model import LineVerifier
from ocr_roi_validator.runtime_guard import (
    ACCENT_PRESENT, BARE_E, RUNTIME_EPSILON, UNKNOWN, assert_monotonic,
    guarded_verdict, in_uncertainty_band, ungated_verdict,
)

PROBE = """
import json, sys, time
import numpy as np, onnxruntime as ort
model, out = sys.argv[1], sys.argv[2]
npz_paths = sys.argv[3:]
session = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
names = [i.name for i in session.get_inputs()]
chunks, lat = [], []
for path in npz_paths:
    data = np.load(path)
    planes = data['planes'].astype(np.float32)
    query = data['query'].astype(np.float32)
    for s in range(0, len(planes), 64):
        feed = {names[0]: planes[s:s+64], names[1]: query[s:s+64]}
        t0 = time.perf_counter()
        r = session.run(None, feed)[0]
        lat.append((time.perf_counter() - t0) / len(r) * 1000.0)
        chunks.append(r)
np.save(out, np.concatenate(chunks))
print(json.dumps({'ort': ort.__version__, 'latencies': lat}))
"""


def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def run_training_onnx(model_path, planes, query, batch=64):
    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path),
                                   providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    chunks, latencies = [], []
    for start in range(0, len(planes), batch):
        feed = {names[0]: planes[start:start + batch],
                names[1]: query[start:start + batch]}
        began = time.perf_counter()
        result = session.run(None, feed)[0]
        latencies.append((time.perf_counter() - began) / len(result) * 1000.0)
        chunks.append(result)
    return np.concatenate(chunks), latencies


def counts(verdict, labels):
    return {
        "legitimate_accent_false_correction": int(
            ((verdict == BARE_E) & (labels == ACCENT_PRESENT)).sum()),
        "wrong_direction_e_to_accent": int(
            ((verdict == ACCENT_PRESENT) & (labels == BARE_E)).sum()),
        "non_accent_change": int(
            ((verdict != UNKNOWN) & (labels == UNKNOWN)).sum()),
        "bare_covered": int(((verdict == BARE_E) & (labels == BARE_E)).sum()),
        "bare_total": int((labels == BARE_E).sum()),
        "unknown_verdicts": int((verdict == UNKNOWN).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, required=True)
    parser.add_argument("--npz", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--product-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    threshold = np.float32(json.loads(
        args.threshold_config.read_text(encoding="utf-8"))["threshold"])

    planes_parts, query_parts, label_parts, origin = [], [], [], []
    for name, path in zip(args.names, args.npz):
        data = np.load(path)
        planes_parts.append(data["planes"].astype(np.float32))
        query_parts.append(data["query"].astype(np.float32))
        label_parts.append(data["label"])
        origin.extend([name] * len(data["label"]))
    planes = np.concatenate(planes_parts)
    query = np.concatenate(query_parts)
    labels = np.concatenate(label_parts)
    origin = np.array(origin)

    model = LineVerifier()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    torch_logits = []
    with torch.no_grad():
        for start in range(0, len(planes), 256):
            logits, _ = model(torch.from_numpy(planes[start:start + 256]),
                              torch.from_numpy(query[start:start + 256]))
            torch_logits.append(logits.numpy())
    torch_probs = softmax(np.concatenate(torch_logits))

    training_logits, training_latencies = run_training_onnx(
        args.onnx, planes, query)
    training_probs = softmax(training_logits)

    probe = VALIDATOR_ROOT / "scripts" / "_guarded_probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    product_out = args.onnx.with_suffix(".guarded.npy")
    completed = subprocess.run(
        [str(args.product_python), str(probe), str(args.onnx), str(product_out)]
        + [str(p) for p in args.npz], capture_output=True, text=True)
    if completed.returncode != 0:
        print("product venv failed:\n" + completed.stderr[-800:], file=sys.stderr)
        return 1
    product_info = json.loads(completed.stdout.strip().splitlines()[-1])
    product_probs = softmax(np.load(product_out))
    probe.unlink(missing_ok=True)
    product_out.unlink(missing_ok=True)

    raw = {name: ungated_verdict(p, threshold) for name, p in
           (("torch", torch_probs), ("training", training_probs),
            ("product", product_probs))}
    guarded = {name: guarded_verdict(p, threshold, RUNTIME_EPSILON) for name, p in
               (("torch", torch_probs), ("training", training_probs),
                ("product", product_probs))}
    classes = {name: p.argmax(axis=1) for name, p in
               (("torch", torch_probs), ("training", training_probs),
                ("product", product_probs))}

    monotonic = {name: assert_monotonic(p, threshold, RUNTIME_EPSILON)
                 for name, p in (("torch", torch_probs),
                                 ("training", training_probs),
                                 ("product", product_probs))}

    pairs = (("torch", "training"), ("torch", "product"),
             ("training", "product"))
    comparisons = {}
    for first, second in pairs:
        key = "%s_vs_%s" % (first, second)
        probs_first = {"torch": torch_probs, "training": training_probs,
                       "product": product_probs}[first]
        probs_second = {"torch": torch_probs, "training": training_probs,
                        "product": product_probs}[second]
        comparisons[key] = {
            "class_mismatches": int((classes[first] != classes[second]).sum()),
            "raw_verdict_mismatches": int((raw[first] != raw[second]).sum()),
            "guarded_verdict_mismatches": int(
                (guarded[first] != guarded[second]).sum()),
            "max_probability_error": float(
                np.abs(probs_first - probs_second).max()),
        }

    per_runtime = {}
    for name in ("torch", "training", "product"):
        band = in_uncertainty_band(
            {"torch": torch_probs, "training": training_probs,
             "product": product_probs}[name], threshold, RUNTIME_EPSILON)
        per_runtime[name] = {
            "raw": counts(raw[name], labels),
            "guarded": counts(guarded[name], labels),
            "rows_in_uncertainty_band": int(band.sum()),
            "monotonic": monotonic[name],
        }

    per_source = {}
    for name in args.names:
        mask = origin == name
        per_source[name] = {
            "rows": int(mask.sum()),
            "raw": counts(raw["torch"][mask], labels[mask]),
            "guarded": counts(guarded["torch"][mask], labels[mask]),
            "withheld_by_guard": int(
                ((raw["torch"][mask] == BARE_E)
                 & (guarded["torch"][mask] == UNKNOWN)).sum()),
        }

    guarded_mismatches = sum(c["guarded_verdict_mismatches"]
                             for c in comparisons.values())
    raw_mismatches = sum(c["raw_verdict_mismatches"] for c in comparisons.values())
    safety_ok = all(
        entry["guarded"]["legitimate_accent_false_correction"] == 0
        and entry["guarded"]["wrong_direction_e_to_accent"] == 0
        and entry["guarded"]["non_accent_change"] == 0
        for entry in per_runtime.values())

    report = {
        "audit": "guarded_parity_audit_v1",
        "run_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision_rule_version": "line-verifier-v1-runtime-guard",
        "parent_threshold": float(threshold),
        "runtime_epsilon": float(RUNTIME_EPSILON),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "total_rows": int(len(planes)),
        "sources": {name: int((origin == name).sum()) for name in args.names},
        "environments": {
            "torch": torch.__version__,
            "training_onnxruntime": __import__("onnxruntime").__version__,
            "product_onnxruntime": product_info["ort"],
        },
        "comparisons": comparisons,
        "per_runtime": per_runtime,
        "per_source": per_source,
        "latency_ms_per_sample": {
            "training_mean": round(float(np.mean(training_latencies)), 4),
            "training_p95": round(float(np.percentile(training_latencies, 95)), 4),
            "product_mean": round(float(np.mean(product_info["latencies"])), 4),
            "product_p95": round(float(np.percentile(product_info["latencies"], 95)), 4),
        },
        "RAW_ONNX_VERDICT_PARITY_STATUS": "FAIL" if raw_mismatches else "PASS",
        "raw_verdict_mismatches_total": raw_mismatches,
        "RUNTIME_GUARD_STATUS": "PASS" if safety_ok else "FAIL",
        "GUARDED_ONNX_VERDICT_PARITY_STATUS": (
            "PASS" if guarded_mismatches == 0 else "FAIL"),
        "guarded_verdict_mismatches_total": guarded_mismatches,
        "note": ("the raw failure is preserved, not overwritten; only the "
                 "guarded deployment verdict is subject to the new gate"),
    }
    payload = json.dumps(report, indent=2)
    args.out.write_text(payload, encoding="utf-8")

    print("rows %d from %s" % (len(planes), report["sources"]))
    print("\n%-22s %7s %11s %15s %14s"
          % ("comparison", "class", "raw verdict", "guarded verdict", "max prob err"))
    for key, entry in comparisons.items():
        print("%-22s %7d %11d %15d %14.3e"
              % (key, entry["class_mismatches"], entry["raw_verdict_mismatches"],
                 entry["guarded_verdict_mismatches"],
                 entry["max_probability_error"]))
    print("\n%-10s %9s %9s %9s %8s %8s"
          % ("runtime", "raw bare", "grd bare", "in band", "acc fc", "unknown"))
    for name, entry in per_runtime.items():
        print("%-10s %9d %9d %9d %8d %8d"
              % (name, entry["raw"]["bare_covered"], entry["guarded"]["bare_covered"],
                 entry["rows_in_uncertainty_band"],
                 entry["guarded"]["legitimate_accent_false_correction"],
                 entry["guarded"]["unknown_verdicts"]))
    print("\nRAW_ONNX_VERDICT_PARITY_STATUS     %s (%d mismatches)"
          % (report["RAW_ONNX_VERDICT_PARITY_STATUS"], raw_mismatches))
    print("RUNTIME_GUARD_STATUS               %s" % report["RUNTIME_GUARD_STATUS"])
    print("GUARDED_ONNX_VERDICT_PARITY_STATUS %s (%d mismatches)"
          % (report["GUARDED_ONNX_VERDICT_PARITY_STATUS"], guarded_mismatches))
    print("report sha256 %s" % hashlib.sha256(payload.encode()).hexdigest())
    return 0 if guarded_mismatches == 0 and safety_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
