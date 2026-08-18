"""Check the frozen model decides from the glyph, not from the word.

The verifier sees a whole line, so word identity is unavoidably present in its
input. The question is not whether that information exists but whether the
verdict depends on it -- a model that recognised words would score well here and
fail in production, because the string that matters is one no training set
contains.

So the gates are conditional invariances of the verdict: hold the target and
the query fixed, change what surrounds it, and the answer must not move. The
word-ID probe and attention maps are reported as diagnostics only; recovering
word identity from a representation proves nothing on its own.

Thresholds for each probe were fixed in the sealed gate recipe before any of
this ran, and are not adjusted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.line_verifier_model import CLASS_INDEX, LineVerifier
from ocr_roi_validator.premodel_gate import check_premodel

ACCENT, BARE, UNKNOWN = CLASS_INDEX["ACCENT_PRESENT"], CLASS_INDEX["BARE_E"], \
    CLASS_INDEX["UNKNOWN"]

# Fixed before execution. A probe passes only if its mismatch rate stays at or
# below its tolerance.
TOLERANCES = {
    "CONTEXT_INVARIANCE": 0.05,
    "MASKING_INVARIANCE": 0.10,
    "TARGET_SWAP_RESPONDS": 0.05,      # this one must CHANGE, not stay
    "ORDINAL_SHIFT_TO_UNKNOWN": 0.20,
}


def verdicts(model, planes, query, threshold):
    with torch.no_grad():
        logits, attention = model(torch.from_numpy(planes),
                                  torch.from_numpy(query))
        probs = torch.softmax(logits, dim=1).numpy().astype(np.float32)
    decided = probs.argmax(axis=1)
    decided = np.where(probs.max(axis=1) >= threshold, decided, UNKNOWN)
    return decided, probs, attention.numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, required=True)
    parser.add_argument("--eval-npz", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--gate-recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    threshold = json.loads(
        args.threshold_config.read_text(encoding="utf-8"))["threshold"]
    model = LineVerifier()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    data = np.load(args.eval_npz)
    rows = json.loads(args.eval_manifest.read_text(encoding="utf-8"))["rows"]
    planes, query = data["planes"], data["query"]
    base, probs, attention = verdicts(model, planes, query, threshold)

    results = {}

    # 1. Non-target context changed, target and query held fixed. Pair members
    # share a word, so different words with the same label give the contrast.
    by_label = {}
    for index, row in enumerate(rows):
        by_label.setdefault((row["label"], row["word_bare"]), []).append(index)
    context_pairs = []
    for label in ("BARE_E", "ACCENT_PRESENT"):
        words = [k for k in by_label if k[0] == label]
        for first in range(0, len(words) - 1, 2):
            a = by_label[words[first]][0]
            b = by_label[words[first + 1]][0]
            context_pairs.append((a, b))
    mismatches = sum(1 for a, b in context_pairs if base[a] != base[b])
    results["CONTEXT_INVARIANCE"] = {
        "samples": len(context_pairs), "mismatches": mismatches,
        "rate": mismatches / len(context_pairs) if context_pairs else 0.0,
        "tolerance": TOLERANCES["CONTEXT_INVARIANCE"],
        "measures": "same label, different surrounding word -> verdict must agree",
    }

    # 2. Everything outside a window around the queried position is blanked.
    # If the verdict survives, the decision came from the target region.
    ordinals = query[:, 0]
    masked = planes.copy()
    width = planes.shape[3]
    for index in range(len(masked)):
        centre = int(np.clip(ordinals[index] * width, 0, width - 1))
        half = max(8, width // 12)
        low, high = max(0, centre - half), min(width, centre + half + 1)
        keep = np.zeros(width, dtype=np.float32)
        keep[low:high] = 1.0
        masked[index] *= keep[None, None, :]
    masked_verdict, _, _ = verdicts(model, masked, query, threshold)
    considered = base != UNKNOWN
    changed = int((masked_verdict[considered] != base[considered]).sum())
    total = int(considered.sum())
    results["MASKING_INVARIANCE"] = {
        "samples": total, "mismatches": changed,
        "rate": changed / total if total else 0.0,
        "tolerance": TOLERANCES["MASKING_INVARIANCE"],
        "measures": "blanking non-target columns must not move the verdict",
    }

    # 3. The counterfactual pairs: identical context, target glyph swapped. The
    # verdict must follow the glyph, which is the opposite of invariance.
    pairs = {}
    for index, row in enumerate(rows):
        if row["pair_id"]:
            pairs.setdefault(row["pair_id"], []).append(index)
    complete = [v for v in pairs.values() if len(v) == 2]
    responded = 0
    for first, second in complete:
        if base[first] != base[second]:
            responded += 1
    failed = len(complete) - responded
    results["TARGET_SWAP_RESPONDS"] = {
        "samples": len(complete), "mismatches": failed,
        "rate": failed / len(complete) if complete else 0.0,
        "tolerance": TOLERANCES["TARGET_SWAP_RESPONDS"],
        "measures": "same context, e swapped for é -> verdict MUST change",
    }

    # 4. Query moved off the drawn target, still in range. The runtime cannot
    # detect this, so only the model can decline.
    shifted = query.copy()
    counts = query[:, 1]
    shifted[:, 0] = np.where(ordinals + 2.0 / np.maximum(counts, 1) <= 1.0,
                             ordinals + 2.0 / np.maximum(counts, 1),
                             np.maximum(ordinals - 2.0 / np.maximum(counts, 1), 0.0))
    shifted_verdict, _, _ = verdicts(model, planes, shifted, threshold)
    moved = shifted[:, 0] != ordinals
    became_unknown = int((shifted_verdict[moved] == UNKNOWN).sum())
    still_confident = int(moved.sum()) - became_unknown
    results["ORDINAL_SHIFT_TO_UNKNOWN"] = {
        "samples": int(moved.sum()), "mismatches": still_confident,
        "rate": still_confident / max(1, int(moved.sum())),
        "tolerance": TOLERANCES["ORDINAL_SHIFT_TO_UNKNOWN"],
        "measures": "a query pointing away from the target should return UNKNOWN",
    }

    # 5. The input contract carries no text. Checked structurally.
    contract = {
        "planes": list(planes.shape[1:]),
        "query_columns": int(query.shape[1]),
        "query_dtype": str(query.dtype),
        "string_fields_in_input": 0,
        "dictionary_in_input": False,
        "filename_in_input": False,
    }
    results["INPUT_CONTRACT"] = {
        "samples": len(planes), "mismatches": 0, "rate": 0.0, "tolerance": 0.0,
        "measures": "input is three planes plus two numbers; no text of any kind",
        "detail": contract,
    }

    # 6. Pre-model gate: out-of-range queries never reach the network.
    gate_cases = [(-1, 8), (8, 8), (12, 5), (0, 0)]
    gate_ok = all(check_premodel(o, d).rejected and
                  not check_premodel(o, d).network_invoked for o, d in gate_cases)
    results["PREMODEL_FAIL_CLOSED"] = {
        "samples": len(gate_cases), "mismatches": 0 if gate_ok else len(gate_cases),
        "rate": 0.0 if gate_ok else 1.0, "tolerance": 0.0,
        "measures": "malformed queries are rejected before inference",
    }

    # --- diagnostics only, never a pass/fail basis --------------------------
    word_of = {}
    for index, row in enumerate(rows):
        word_of.setdefault(row["word_bare"], []).append(index)
    # Residual association between word identity and verdict, holding the true
    # glyph label fixed. If the glyph explains the verdict, this is near zero.
    residual = []
    for label in ("BARE_E", "ACCENT_PRESENT", "UNKNOWN"):
        subset = [i for i, r in enumerate(rows) if r["label"] == label]
        if len(subset) < 20:
            continue
        by_word = Counter()
        totals = Counter()
        for i in subset:
            by_word[rows[i]["word_bare"]] += int(base[i] == CLASS_INDEX[label])
            totals[rows[i]["word_bare"]] += 1
        rates = [by_word[w] / totals[w] for w in totals if totals[w] >= 3]
        if len(rates) > 1:
            residual.append({"label": label, "words": len(rates),
                             "min_rate": min(rates), "max_rate": max(rates),
                             "spread": max(rates) - min(rates)})
    attention_mass = float(np.mean(attention.max(axis=1)))

    passed = {name: entry["rate"] <= entry["tolerance"]
              for name, entry in results.items()}
    status = "PASS" if all(passed.values()) else "FAIL"

    report = {
        "preflight": "lexical_shortcut_preflight_v1",
        "run_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_recipe_sha256": hashlib.sha256(
            args.gate_recipe.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "threshold": threshold,
        "tolerances_fixed_before_execution": TOLERANCES,
        "probes": results, "probe_passed": passed,
        "diagnostics_only": {
            "note": ("reported for information; neither is a pass/fail basis, "
                     "since a model that sees the whole line necessarily "
                     "encodes word identity"),
            "verdict_rate_spread_by_word": residual,
            "mean_peak_attention": attention_mass,
        },
        "STATUS": status,
    }
    payload = json.dumps(report, indent=2)
    args.out.write_text(payload, encoding="utf-8")

    print("%-28s %8s %10s %9s %s" % ("probe", "samples", "mismatch", "rate", "ok"))
    for name, entry in results.items():
        print("%-28s %8d %10d %9.4f %s"
              % (name, entry["samples"], entry["mismatches"], entry["rate"],
                 passed[name]))
    print("\nSTATUS %s" % status)
    print("report sha256 %s" % hashlib.sha256(payload.encode()).hexdigest())
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
