"""Export the frozen verifier to ONNX and prove three runtimes agree.

Nothing about the model or the threshold changes here. The export exists so the
same decision can be reproduced in the product's ONNX Runtime, which is a
different version from the one used for training, and version skew has broken
numerics in this project before.

Parity is checked on decisions, not just on tensors. Two runtimes can differ in
the tenth decimal place and still disagree on a verdict if a probability sits
next to the threshold, so the class, the thresholded verdict, and the behaviour
of samples near the boundary are each compared separately.

The artifact is diagnostic_only. It is not placed in the runtime package.
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

UNKNOWN = 2
BOUNDARY_BAND = 0.02


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def decide(probs, threshold):
    predicted = probs.argmax(axis=1)
    return np.where(probs.max(axis=1) >= threshold, predicted, UNKNOWN)


def run_onnx(model_path, planes, query, batch=64):
    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path),
                                   providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    outputs = []
    latencies = []
    for start in range(0, len(planes), batch):
        chunk = {names[0]: planes[start:start + batch],
                 names[1]: query[start:start + batch]}
        began = time.perf_counter()
        result = session.run(None, chunk)[0]
        latencies.append((time.perf_counter() - began) / len(result) * 1000.0)
        outputs.append(result)
    return np.concatenate(outputs), latencies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, required=True)
    parser.add_argument("--sample-npz", type=Path, required=True)
    parser.add_argument("--onnx-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--product-python", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    threshold = json.loads(
        args.threshold_config.read_text(encoding="utf-8"))["threshold"]
    model = LineVerifier()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    data = np.load(args.sample_npz)
    planes = data["planes"].astype(np.float32)
    query = data["query"].astype(np.float32)

    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (torch.from_numpy(planes[:1]), torch.from_numpy(query[:1])),
        str(args.onnx_out), opset_version=args.opset,
        input_names=["planes", "query"], output_names=["logits", "attention"],
        dynamic_axes={"planes": {0: "batch"}, "query": {0: "batch"},
                      "logits": {0: "batch"}, "attention": {0: "batch"}},
        do_constant_folding=True)

    # PyTorch reference
    began = time.perf_counter()
    with torch.no_grad():
        torch_logits, _ = model(torch.from_numpy(planes),
                                torch.from_numpy(query))
    torch_elapsed = (time.perf_counter() - began) / len(planes) * 1000.0
    torch_probs = torch.softmax(torch_logits, dim=1).numpy().astype(np.float32)

    # Training-environment ONNX Runtime
    training_logits, training_latencies = run_onnx(args.onnx_out, planes, query)
    training_probs = softmax(training_logits.astype(np.float32))

    # Product venv ONNX Runtime, in its own interpreter
    probe = VALIDATOR_ROOT / "scripts" / "_onnx_probe.py"
    probe.write_text(
        "import json, sys, time\n"
        "import numpy as np, onnxruntime as ort\n"
        "model, npz, out = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "data = np.load(npz)\n"
        "planes = data['planes'].astype(np.float32)\n"
        "query = data['query'].astype(np.float32)\n"
        "session = ort.InferenceSession(model, providers=['CPUExecutionProvider'])\n"
        "names = [i.name for i in session.get_inputs()]\n"
        "chunks, lat = [], []\n"
        "for s in range(0, len(planes), 64):\n"
        "    feed = {names[0]: planes[s:s+64], names[1]: query[s:s+64]}\n"
        "    t0 = time.perf_counter()\n"
        "    r = session.run(None, feed)[0]\n"
        "    lat.append((time.perf_counter()-t0)/len(r)*1000.0)\n"
        "    chunks.append(r)\n"
        "np.save(out, np.concatenate(chunks))\n"
        "print(json.dumps({'ort': ort.__version__, 'latencies': lat}))\n",
        encoding="utf-8")
    product_out = args.onnx_out.with_suffix(".product.npy")
    completed = subprocess.run(
        [str(args.product_python), str(probe), str(args.onnx_out),
         str(args.sample_npz), str(product_out)],
        capture_output=True, text=True)
    product_loaded = completed.returncode == 0
    if not product_loaded:
        print("product venv load FAILED:\n" + completed.stderr[-800:],
              file=sys.stderr)
        report = {"STATUS": "FAIL", "product_venv_load": False,
                  "stderr": completed.stderr[-2000:]}
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1
    product_info = json.loads(completed.stdout.strip().splitlines()[-1])
    product_probs = softmax(np.load(product_out).astype(np.float32))

    torch_class = torch_probs.argmax(axis=1)
    training_class = training_probs.argmax(axis=1)
    product_class = product_probs.argmax(axis=1)
    torch_verdict = decide(torch_probs, threshold)
    training_verdict = decide(training_probs, threshold)
    product_verdict = decide(product_probs, threshold)

    near = np.abs(torch_probs.max(axis=1) - threshold) <= BOUNDARY_BAND
    comparisons = {
        "torch_vs_training": {
            "class_mismatches": int((torch_class != training_class).sum()),
            "verdict_mismatches": int((torch_verdict != training_verdict).sum()),
            "max_probability_error": float(np.abs(torch_probs - training_probs).max()),
        },
        "torch_vs_product": {
            "class_mismatches": int((torch_class != product_class).sum()),
            "verdict_mismatches": int((torch_verdict != product_verdict).sum()),
            "max_probability_error": float(np.abs(torch_probs - product_probs).max()),
        },
        "training_vs_product": {
            "class_mismatches": int((training_class != product_class).sum()),
            "verdict_mismatches": int((training_verdict != product_verdict).sum()),
            "max_probability_error": float(np.abs(training_probs - product_probs).max()),
        },
    }
    boundary = {
        "band": BOUNDARY_BAND,
        "samples_near_threshold": int(near.sum()),
        "verdict_mismatches_near_threshold": int(
            ((torch_verdict != product_verdict) & near).sum()
            + ((torch_verdict != training_verdict) & near).sum()),
    }
    total_mismatch = sum(c["class_mismatches"] + c["verdict_mismatches"]
                         for c in comparisons.values())

    report = {
        "export": "line_verifier_diagnostic_onnx_v1",
        "role": "diagnostic_only -- NOT placed in the runtime package",
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opset": args.opset,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.onnx_out.read_bytes()).hexdigest(),
        "threshold": threshold,
        "samples": int(len(planes)),
        "environments": {
            "torch": torch.__version__,
            "training_onnxruntime": __import__("onnxruntime").__version__,
            "product_onnxruntime": product_info["ort"],
        },
        "product_venv_load": product_loaded,
        "comparisons": comparisons,
        "threshold_boundary": boundary,
        "latency_ms_per_sample": {
            "torch_mean": round(torch_elapsed, 4),
            "training_mean": round(float(np.mean(training_latencies)), 4),
            "training_p95": round(float(np.percentile(training_latencies, 95)), 4),
            "product_mean": round(float(np.mean(product_info["latencies"])), 4),
            "product_p95": round(float(np.percentile(product_info["latencies"], 95)), 4),
        },
        "STATUS": "PASS" if total_mismatch == 0 else "FAIL",
    }
    payload = json.dumps(report, indent=2)
    args.report.write_text(payload, encoding="utf-8")
    probe.unlink(missing_ok=True)
    product_out.unlink(missing_ok=True)

    print("samples %d, opset %d" % (len(planes), args.opset))
    for name, entry in comparisons.items():
        print("  %-22s class %d  verdict %d  max prob err %.3e"
              % (name, entry["class_mismatches"], entry["verdict_mismatches"],
                 entry["max_probability_error"]))
    print("  near threshold (+-%.2f): %d samples, %d verdict mismatches"
          % (BOUNDARY_BAND, boundary["samples_near_threshold"],
             boundary["verdict_mismatches_near_threshold"]))
    print("  latency ms/sample: training %.3f (p95 %.3f), product %.3f (p95 %.3f)"
          % (report["latency_ms_per_sample"]["training_mean"],
             report["latency_ms_per_sample"]["training_p95"],
             report["latency_ms_per_sample"]["product_mean"],
             report["latency_ms_per_sample"]["product_p95"]))
    print("  onnx sha256 %s" % report["onnx_sha256"])
    print("STATUS %s" % report["STATUS"])
    return 0 if total_mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
