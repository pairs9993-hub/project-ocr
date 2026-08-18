"""Freeze the decision threshold for the selected model.

Argmax is not the decision rule. A correction only fires when the model is
confident enough, and "enough" is set here against data the model was never
selected on: the held-out half of the counterfactual calibration contexts, plus
the legitimate-accent preservation rows from the failed calibration-v2 run,
which are real baseline outputs rather than synthetic pairs.

The rule is ordered. No threshold is acceptable if it mis-corrects a genuine
accent, flips a bare e to an accented one, or changes anything other than the
queried accent; among those that satisfy all three, the one covering the most
bare-e cases wins. Anything below threshold returns UNKNOWN and the baseline
string stands.

The search steps in float32 with ``nextafter``. An earlier accent model was
frozen at a threshold chosen in float64 that rounded back below a real
probability once cast, silently admitting twenty forbidden corrections.

Zero observed errors on a finite sample is not a zero error rate. Every count
is reported with its denominator and a one-sided 95% upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.line_verifier_model import CLASS_INDEX, LineVerifier

ACCENT = CLASS_INDEX["ACCENT_PRESENT"]
BARE = CLASS_INDEX["BARE_E"]
UNKNOWN = CLASS_INDEX["UNKNOWN"]


def upper_bound_zero(trials: int, confidence: float = 0.95) -> float:
    """One-sided 95% upper bound on a rate after observing no events."""
    return 1.0 - (1.0 - confidence) ** (1.0 / trials) if trials else 1.0


def wilson_upper(successes: int, trials: int, z: float = 1.6449) -> float:
    if trials == 0:
        return 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return min(1.0, centre + spread)


def probabilities(model, npz_path):
    data = np.load(npz_path)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(data["planes"]),
                          torch.from_numpy(data["query"]))
        return (torch.softmax(logits, dim=1).numpy().astype(np.float32),
                data["label"])


def decide(probs, threshold):
    """Correct only when the winning class clears the threshold."""
    predicted = probs.argmax(axis=1)
    confident = probs.max(axis=1) >= threshold
    return np.where(confident, predicted, UNKNOWN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--threshold-npz", type=Path, required=True)
    parser.add_argument("--anchor-npz", type=Path,
                        help="legitimate-accent preservation rows")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    model = LineVerifier()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    probs, labels = probabilities(model, args.threshold_npz)
    anchor_probs = anchor_labels = None
    if args.anchor_npz and args.anchor_npz.is_file():
        anchor_probs, anchor_labels = probabilities(model, args.anchor_npz)

    def violations(threshold):
        """The three prohibited outcomes, counted at one threshold."""
        decided = decide(probs, threshold)
        accent_rows = labels == ACCENT
        bare_rows = labels == BARE
        # 1. a real accent corrected away
        false_correction = int(((decided == BARE) & accent_rows).sum())
        # 2. a bare e turned into an accent
        wrong_direction = int(((decided == ACCENT) & bare_rows).sum())
        # 3. an unanswerable query answered as if it were not
        non_accent_change = int(((decided != UNKNOWN) & (labels == UNKNOWN)).sum())
        covered = int(((decided == BARE) & bare_rows).sum())
        if anchor_probs is not None:
            anchor_decided = decide(anchor_probs, threshold)
            false_correction += int(((anchor_decided == BARE)
                                     & (anchor_labels == ACCENT)).sum())
        return false_correction, wrong_direction, non_accent_change, covered

    # Candidate thresholds are the observed confidences themselves, stepped up
    # by one float32 ulp so a value exactly at a sample's confidence does not
    # admit it. Stepping in float64 would round back below the float32 value.
    observed = np.unique(probs.max(axis=1).astype(np.float32))
    candidates = [np.nextafter(np.float32(v), np.float32(1.0)) for v in observed]
    candidates.append(np.float32(1.0))
    if anchor_probs is not None:
        for value in np.unique(anchor_probs.max(axis=1).astype(np.float32)):
            candidates.append(np.nextafter(np.float32(value), np.float32(1.0)))
    candidates = sorted({float(c) for c in candidates})

    acceptable = []
    for threshold in candidates:
        false_correction, wrong_direction, other, covered = violations(threshold)
        if false_correction == 0 and wrong_direction == 0 and other == 0:
            acceptable.append((covered, -threshold, threshold))
    if not acceptable:
        report = {"STATUS": "THRESHOLD_NOT_FOUND",
                  "reason": "no threshold satisfied all three prohibitions"}
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("THRESHOLD_NOT_FOUND")
        return 1

    covered, _, chosen = max(acceptable)
    false_correction, wrong_direction, other, _ = violations(chosen)

    accent_rows = int((labels == ACCENT).sum())
    bare_rows = int((labels == BARE).sum())
    unknown_rows = int((labels == UNKNOWN).sum())
    anchor_rows = int((anchor_labels == ACCENT).sum()) if anchor_probs is not None else 0
    accent_denominator = accent_rows + anchor_rows

    report = {
        "config": "line_verifier_threshold_v1",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "threshold": float(chosen),
        "threshold_float32_hex": float(np.float32(chosen)).hex(),
        "rule": ("apply the predicted class only when its probability >= "
                 "threshold; otherwise return UNKNOWN and leave the baseline "
                 "string unchanged"),
        "data": {
            "threshold_calibration_v1": str(args.threshold_npz),
            "preservation_anchor": str(args.anchor_npz) if anchor_probs is not None else None,
        },
        "counts": {
            "accent_false_correction": false_correction,
            "accent_denominator": accent_denominator,
            "wrong_direction_e_to_accent": wrong_direction,
            "bare_denominator": bare_rows,
            "non_accent_change": other,
            "unknown_denominator": unknown_rows,
            "bare_covered": covered,
            "bare_coverage": round(covered / bare_rows, 6) if bare_rows else None,
        },
        "finite_sample_bounds": {
            "note": ("zero observed errors is not a zero error rate; these are "
                     "one-sided 95% upper bounds at the observed denominators"),
            "accent_false_correction_upper95": (
                upper_bound_zero(accent_denominator) if false_correction == 0
                else wilson_upper(false_correction, accent_denominator)),
            "wrong_direction_upper95": (
                upper_bound_zero(bare_rows) if wrong_direction == 0
                else wilson_upper(wrong_direction, bare_rows)),
            "non_accent_change_upper95": (
                upper_bound_zero(unknown_rows) if other == 0
                else wilson_upper(other, unknown_rows)),
        },
        "float32_boundary": (
            "candidates are observed confidences stepped by one float32 ulp via "
            "nextafter; an earlier model was frozen at a float64 value that "
            "rounded back below a real probability and admitted 20 forbidden "
            "corrections"),
        "immutable_after_freeze": (
            "natural support, target F0 and F2 results may not motivate a change"),
        "STATUS": "FROZEN",
    }
    payload = json.dumps(report, indent=2)
    args.out.write_text(payload, encoding="utf-8")

    print("selected seed %d" % args.seed)
    print("threshold %.10f" % chosen)
    print("  accent false correction %d/%d (upper95 %.5f)"
          % (false_correction, accent_denominator,
             report["finite_sample_bounds"]["accent_false_correction_upper95"]))
    print("  wrong direction e->accent %d/%d" % (wrong_direction, bare_rows))
    print("  non-accent change %d/%d" % (other, unknown_rows))
    print("  bare coverage %d/%d = %.4f" % (covered, bare_rows, covered / bare_rows))
    print("config sha256 %s" % hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
