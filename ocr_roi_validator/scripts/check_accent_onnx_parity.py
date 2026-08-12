"""Verify the exported accent CNN behaves identically everywhere it will run.

Exporting at a conservative opset is not evidence of compatibility. The model
is trained under PyTorch in one environment and executed under ONNX Runtime in
another, on a different Python and a newer runtime version, so all three have
to be compared on the same inputs.

What must match is not the floating-point output but the decisions taken from
it: the predicted class, and -- more importantly -- whether the pipeline would
correct a character or abstain. A probability difference of 1e-7 is
irrelevant; a single flipped correction is not.

Run the PyTorch half in the training environment and pass ``--torch-npz`` to
compare it here, or run without it to compare the two ONNX Runtimes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.accent_cnn_input import (  # noqa: E402
    AccentInputConfig,
    prepare_cnn_input,
)


def softmax_bare(logits: np.ndarray) -> np.ndarray:
    """P(no accent) for a (N, 2) logit array."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return (exponentiated / exponentiated.sum(axis=1, keepdims=True))[:, 1]


def load_inputs(split_dir: Path, config: AccentInputConfig, limit: int):
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    tensors, meta = [], []
    for sample in manifest["samples"]:
        if limit and len(tensors) >= limit:
            break
        prepared = prepare_cnn_input(
            cv2.imread(str(split_dir / sample["file"])), config
        )
        if prepared is None:
            continue
        tensors.append(prepared[0])
        meta.append(sample)
    return np.stack(tensors).astype(np.float32), meta


def run_onnx(model_path: Path, batch: np.ndarray) -> tuple[np.ndarray, dict]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: batch})[0]

    # Per-sample latency, which is what the runtime actually pays.
    single = batch[:1]
    for _ in range(5):
        session.run(None, {name: single})
    timings = []
    for _ in range(60):
        started = time.perf_counter()
        session.run(None, {name: single})
        timings.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(timings)
    return np.asarray(logits), {
        "onnxruntime": ort.__version__,
        "python": sys.version.split()[0],
        "mean_ms": statistics.fmean(timings),
        "p95_ms": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
    }


def decisions(probabilities: np.ndarray, absent: float, present: float) -> np.ndarray:
    """0 = accent present, 1 = absent (correct), 2 = unknown (abstain)."""
    verdicts = np.full(probabilities.shape, 2, dtype=np.int64)
    verdicts[probabilities >= absent] = 1
    verdicts[probabilities <= present] = 0
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--torch-npz", type=Path,
                        help="logits saved from the training environment")
    parser.add_argument("--save-inputs", type=Path,
                        help="write the prepared batch for the torch run")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = AccentInputConfig(**settings["input_config"])
    absent = float(settings["absent_threshold"])
    present = float(settings["present_threshold"])

    batch, meta = load_inputs(args.split_dir, config, args.limit)
    print(f"inputs      : {batch.shape}")
    if args.save_inputs:
        args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_inputs, batch=batch)
        print(f"wrote {args.save_inputs}")

    logits, runtime_info = run_onnx(args.onnx, batch)
    probabilities = softmax_bare(logits)
    verdicts = decisions(probabilities, absent, present)
    print(f"onnxruntime : {runtime_info['onnxruntime']} "
          f"(python {runtime_info['python']})")
    print(f"latency     : mean {runtime_info['mean_ms']:.3f} ms  "
          f"p95 {runtime_info['p95_ms']:.3f} ms")
    print(f"decisions   : accent={int((verdicts == 0).sum())} "
          f"absent={int((verdicts == 1).sum())} unknown={int((verdicts == 2).sum())}")

    report = {
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "samples": int(batch.shape[0]),
        "runtime": runtime_info,
        "decision_counts": {
            "accent": int((verdicts == 0).sum()),
            "absent": int((verdicts == 1).sum()),
            "unknown": int((verdicts == 2).sum()),
        },
    }

    if args.torch_npz and args.torch_npz.is_file():
        reference = np.load(args.torch_npz)
        reference_logits = reference["logits"]
        if reference_logits.shape != logits.shape:
            print("torch logits shape mismatch", file=sys.stderr)
            return 1
        reference_probabilities = softmax_bare(reference_logits)
        reference_verdicts = decisions(reference_probabilities, absent, present)

        class_match = int(
            ((reference_logits.argmax(1)) == (logits.argmax(1))).sum()
        )
        decision_match = int((reference_verdicts == verdicts).sum())
        max_difference = float(np.abs(reference_probabilities - probabilities).max())
        mean_difference = float(np.abs(reference_probabilities - probabilities).mean())

        print(f"\nvs PyTorch  : class {class_match}/{len(verdicts)}  "
              f"decision {decision_match}/{len(verdicts)}")
        print(f"  max |Δp|  : {max_difference:.3e}")
        print(f"  mean |Δp| : {mean_difference:.3e}")
        report["vs_torch"] = {
            "class_match": class_match,
            "decision_match": decision_match,
            "total": int(len(verdicts)),
            "max_probability_difference": max_difference,
            "mean_probability_difference": mean_difference,
            "class_parity": class_match == len(verdicts),
            "decision_parity": decision_match == len(verdicts),
        }
        if class_match != len(verdicts) or decision_match != len(verdicts):
            print("PARITY FAILED", file=sys.stderr)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
