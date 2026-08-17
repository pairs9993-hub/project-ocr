"""Size budgets on the probability that ALL quotas are met in one run.

The previous gate checked each quota separately and found every one at or above
0.95. That is not the same claim. A split has to satisfy a dozen quotas in a
single generation, and if one of them sits at exactly 0.95 while the rest are
near certainty, the run still fails five times in a hundred -- the joint
probability is bounded above by the weakest member, never higher.

So the budget is set on P(all quotas met), estimated by resampling the pilot's
own rows. Each draw takes N renderings with replacement from the pilot, keeping
each row's (font, stratum, outcome) tuple intact, and counts every quota on the
resulting sample. That preserves the real joint structure: a rendering is one
font in one stratum with one terminal reason, so a row that supplies a
preservation event cannot also supply a hallucination, and the fonts compete for
the same fixed exposure. Multiplying marginals would miss both constraints.

Resampling rows rather than drawing from fitted rates also carries the pilot's
own sampling error into the answer, which is the conservative direction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402
from size_v2_budget_v2 import MACRO_STRATA, QUOTAS  # noqa: E402

DRAWS = 4000
SEED = 20260817
TARGET = 0.95
UNKNOWN_REASONS = {"DELETION", "INSERTION", "MULTIPLE_CHANGES",
                   "CHANGE_ELSEWHERE", "OTHER_SUBSTITUTION", "ACCENT_LOST"}


def load_pilot(pilot: Path):
    rows = [json.loads(line) for line
            in (pilot / "checkpoint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    recipe = json.loads((pilot / "recipe.json").read_text(encoding="utf-8"))
    return rows, recipe


def build_arrays(rows, fonts):
    """Encode the pilot rows a split can draw from, as parallel arrays."""
    subset = [r for r in rows if r["font"] in fonts]
    font_index = {f: i for i, f in enumerate(sorted(fonts))}
    stratum_index = {s: i for i, s in enumerate(MACRO_STRATA)}
    return {
        "font": np.array([font_index[r["font"]] for r in subset], dtype=np.int8),
        "stratum": np.array([stratum_index[r["target_stratum"]] for r in subset],
                            dtype=np.int8),
        "hallucination": np.array([r["clean_hallucination"] for r in subset],
                                  dtype=bool),
        "preservation": np.array([r["clean_preservation"] for r in subset],
                                 dtype=bool),
        "unknown": np.array([r["terminal_reason"] in UNKNOWN_REASONS
                             for r in subset], dtype=bool),
        "font_names": sorted(fonts),
        "rows": len(subset),
    }


def joint_success(arrays, quota, n, draws=DRAWS, seed=SEED,
                  two_stage=True):
    """Fraction of resampled runs of size ``n`` meeting every quota at once.

    Resampling the pilot rows directly reproduces the pilot's *point* rates,
    which makes the estimate less conservative than the marginal calculation
    that used one-sided lower bounds -- the two disagreed by a wide margin for
    exactly that reason. With ``two_stage``, each draw first resamples the
    pilot itself (a bootstrap of the 60,000 rows) and only then draws the run
    from that resampled pilot, so the uncertainty in the pilot's own rates is
    carried into the answer rather than treated as known.
    """
    rng = np.random.default_rng(seed)
    total = arrays["rows"]
    credit = quota.get("credit", {})
    successes = 0
    per_quota = defaultdict(int)

    for _ in range(draws):
        if two_stage:
            # Stage one: a plausible pilot, given the pilot we observed.
            pilot_draw = rng.integers(0, total, size=total)
            picked = pilot_draw[rng.integers(0, total, size=n)]
        else:
            picked = rng.integers(0, total, size=n)
        font = arrays["font"][picked]
        stratum = arrays["stratum"][picked]
        hallucination = arrays["hallucination"][picked]
        preservation = arrays["preservation"][picked]
        unknown = arrays["unknown"][picked]

        checks = {}
        credited = sum(credit.values())
        checks["hallucination_total"] = (
            int(hallucination.sum()) + credited >= quota["hallucination_total"])
        for index, name in enumerate(MACRO_STRATA):
            here = int(hallucination[stratum == index].sum()) + credit.get(name, 0)
            checks[f"hallucination_{name}"] = (
                here >= quota["hallucination_per_stratum"])
        checks["preservation_total"] = (
            int(preservation.sum()) >= quota["preservation_total"])
        if quota["preservation_per_stratum"]:
            for index, name in enumerate(MACRO_STRATA):
                checks[f"preservation_{name}"] = (
                    int(preservation[stratum == index].sum())
                    >= quota["preservation_per_stratum"])
        if quota["preservation_per_font"]:
            for index, name in enumerate(arrays["font_names"]):
                checks[f"preservation_font_{name}"] = (
                    int(preservation[font == index].sum())
                    >= quota["preservation_per_font"])
        checks["unknown_total"] = int(unknown.sum()) >= quota["unknown_total"]

        for name, passed in checks.items():
            per_quota[name] += int(passed)
        successes += int(all(checks.values()))

    probability = successes / draws
    # Binomial standard error on the Monte Carlo estimate itself.
    error = math.sqrt(max(probability * (1 - probability), 1e-12) / draws)
    return {
        "n": n,
        "joint_probability": probability,
        "monte_carlo_stderr": error,
        "monte_carlo_ci95": [max(0.0, probability - 1.96 * error),
                             min(1.0, probability + 1.96 * error)],
        "marginal_probabilities": {k: v / draws for k, v in sorted(per_quota.items())},
        "weakest_quota": (min(per_quota, key=lambda k: per_quota[k])
                          if per_quota else None),
        "draws": draws, "seed": seed,
    }


def find_budget(arrays, quota, start, ceiling, draws, seed):
    """Smallest n (on a coarse grid, then refined) reaching the joint target."""
    n = start
    history = []
    while n <= ceiling:
        result = joint_success(arrays, quota, n, draws=max(600, draws // 4),
                               seed=seed)
        history.append((n, result["joint_probability"]))
        # Require the lower confidence bound to clear the target, so Monte
        # Carlo noise cannot pass a budget that is actually short.
        if result["joint_probability"] - 1.96 * result["monte_carlo_stderr"] >= TARGET:
            break
        n = int(n * 1.35) + 1
    else:
        return None, history

    low = history[-2][0] if len(history) > 1 else start
    high = n
    while high - low > max(250, high // 100):
        middle = (low + high) // 2
        result = joint_success(arrays, quota, middle, draws=max(600, draws // 4),
                               seed=seed)
        if result["joint_probability"] - 1.96 * result["monte_carlo_stderr"] >= TARGET:
            high = middle
        else:
            low = middle + 1
    return high, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--ceiling", type=int, default=2_000_000)
    args = parser.parse_args()

    rows, recipe = load_pilot(args.pilot)
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    split_fonts = recipe["split_fonts"]

    results = {}
    for split, quota in QUOTAS.items():
        arrays = build_arrays(rows, split_fonts[split])
        marginal_budget = previous["budgets"][split]["total_renderings"]
        at_marginal = joint_success(arrays, quota, marginal_budget,
                                    draws=args.draws, seed=SEED)
        budget, history = find_budget(arrays, quota, marginal_budget,
                                      args.ceiling, args.draws, SEED)
        confirmed = (joint_success(arrays, quota, budget, draws=args.draws,
                                   seed=SEED + 1) if budget else None)
        results[split] = {
            "fonts": split_fonts[split],
            "pilot_rows_available": arrays["rows"],
            "previous_marginal_budget": marginal_budget,
            "joint_probability_at_marginal_budget": at_marginal,
            "search_history": history,
            "joint_budget": budget,
            "confirmation_independent_seed": confirmed,
            "feasible": budget is not None,
        }

    feasible = all(v["feasible"] for v in results.values())
    total = (sum(v["joint_budget"] for v in results.values()) if feasible else None)

    # Wall time from the pilot's own measurement, not an assumed throughput.
    pilot_seconds = 10705.0
    per_rendering = pilot_seconds / 60000.0
    report = {
        "analysis": "joint_quota_gate_v2",
        "criterion": "B -- P(all quotas met in one run) >= 0.95",
        "previous_criterion_was": (
            "A -- each quota individually >= 0.95, which does not imply the "
            "joint claim: with one binding quota at 0.95 and the rest near 1.0, "
            "the joint probability is bounded above by the weakest member"),
        "method": (
            "resample pilot rows with replacement, preserving each row's "
            "(font, stratum, outcome) tuple, so mutually exclusive outcomes and "
            "shared exposure across fonts are represented rather than assumed "
            "independent"),
        "draws": args.draws, "seed": SEED, "target": TARGET,
        "budget_accepted_on": "Monte Carlo lower confidence bound, not the point estimate",
        "results": results,
        "SUPPLEMENT_JOINT_SUCCESS_PROB": (
            results["supplement"]["confirmation_independent_seed"]["joint_probability"]
            if results["supplement"]["feasible"] else None),
        "CALIBRATION_JOINT_SUCCESS_PROB": (
            results["calibration"]["confirmation_independent_seed"]["joint_probability"]
            if results["calibration"]["feasible"] else None),
        "PREFLIGHT_JOINT_SUCCESS_PROB": (
            results["preflight"]["confirmation_independent_seed"]["joint_probability"]
            if results["preflight"]["feasible"] else None),
        "ALL_JOINT_QUOTAS_FEASIBLE": feasible,
        "TOTAL_RENDERINGS": total,
        "ESTIMATED_RUNTIME_HOURS": (round(total * per_rendering / 3600, 2)
                                    if total else None),
    }
    digest = atomic_write_json(args.out, report)

    for split, entry in results.items():
        at = entry["joint_probability_at_marginal_budget"]
        print(f"\n{split}")
        print(f"  marginal budget {entry['previous_marginal_budget']:,} "
              f"-> joint P = {at['joint_probability']:.4f} "
              f"(weakest: {at['weakest_quota']})")
        if entry["feasible"]:
            confirmed = entry["confirmation_independent_seed"]
            print(f"  joint budget    {entry['joint_budget']:,} "
                  f"-> joint P = {confirmed['joint_probability']:.4f} "
                  f"+/- {1.96 * confirmed['monte_carlo_stderr']:.4f}")
            worst = sorted(confirmed["marginal_probabilities"].items(),
                           key=lambda item: item[1])[:3]
            for name, value in worst:
                print(f"      {name:34s} {value:.4f}")
        else:
            print("  INFEASIBLE within ceiling")
    print(f"\nALL_JOINT_QUOTAS_FEASIBLE {feasible}")
    if total:
        print(f"TOTAL_RENDERINGS {total:,}")
        print(f"ESTIMATED_RUNTIME {report['ESTIMATED_RUNTIME_HOURS']} h")
    print(f"report sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
