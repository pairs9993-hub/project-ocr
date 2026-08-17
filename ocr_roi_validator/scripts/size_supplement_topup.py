"""Size the supplement top-up from the supplement's own measured rates.

The base run fell short because rates measured on the pilot did not transfer to
the supplement's cohort: with fonts held equal, accent preservation ran at 0.142
against the pilot's 0.184, and per-word preservation inside the supplement
spans 0.052 to 0.225. Independence required the two cohorts to use different
words, and I treated a rate measured on one as if it applied to the other.

That failure mode cannot recur here, because the top-up continues the *same*
cohort. Its rates come from the 23,095 renderings already produced, which is a
direct measurement of the population being extended rather than a transfer from
a different one.

The requirement is joint: the cumulative preservation count and the cumulative
LARGE hallucination count must both clear their quotas, and 349/0.0715 is only
the N at which the expected preservation count reaches the target -- roughly a
coin flip. The number returned is the smallest N where both clear together with
probability 0.95, estimated by resampling the base rows so the two events keep
their observed joint structure instead of being assumed independent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402

DRAWS = 4000
SEED = 20260819
TARGET = 0.95
HARD_CAP = 200_000


def wilson_lower(successes: int, trials: int, z: float = 1.6449) -> float:
    """One-sided 95% lower bound on a proportion."""
    if trials == 0 or successes == 0:
        return 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - spread)


def joint_success(preservation, large, credit_preservation, credit_large,
                  need_preservation, need_large, n, draws=DRAWS, seed=SEED):
    """P(both cumulative quotas met) over ``n`` further renderings.

    Rows are resampled in two stages: first the base run itself is
    bootstrapped, then the top-up is drawn from that resampled base. The outer
    stage carries the uncertainty in the base rates rather than treating the
    23,095 renderings as if they pinned the rate exactly -- which is close to
    the mistake that produced this shortfall in the first place.
    """
    rng = np.random.default_rng(seed)
    total = len(preservation)
    successes = 0
    both = [0, 0]
    for _ in range(draws):
        resampled = rng.integers(0, total, size=total)
        picked = resampled[rng.integers(0, total, size=n)]
        got_preservation = int(preservation[picked].sum())
        got_large = int(large[picked].sum())
        ok_preservation = got_preservation + credit_preservation >= need_preservation
        ok_large = got_large + credit_large >= need_large
        both[0] += int(ok_preservation)
        both[1] += int(ok_large)
        successes += int(ok_preservation and ok_large)
    probability = successes / draws
    error = math.sqrt(max(probability * (1 - probability), 1e-12) / draws)
    return {
        "n": n, "joint_probability": probability,
        "monte_carlo_stderr": error,
        "lower_confidence_bound": probability - 1.96 * error,
        "marginal_preservation": both[0] / draws,
        "marginal_large": both[1] / draws,
        "draws": draws, "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in (args.base / "checkpoint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((args.base / "manifest.json").read_text(encoding="utf-8"))
    state = manifest["quota_state"]

    preservation = np.array([bool(r["clean_preservation"]) for r in rows])
    large = np.array([bool(r["clean_hallucination"]
                           and r.get("measured_stratum") == "LARGE")
                      for r in rows])

    credit_preservation = state["preservation_total"]["observed"]
    credit_large = state["hallucination_LARGE"]["observed"]
    need_preservation = state["preservation_total"]["required"]
    need_large = state["hallucination_LARGE"]["required"]

    rate_preservation = float(preservation.mean())
    rate_large = float(large.mean())
    lower_preservation = wilson_lower(int(preservation.sum()), len(rows))
    lower_large = wilson_lower(int(large.sum()), len(rows))

    shortfall_preservation = max(0, need_preservation - credit_preservation)
    shortfall_large = max(0, need_large - credit_large)
    naive = (math.ceil(shortfall_preservation / rate_preservation)
             if rate_preservation else None)

    # Grow until the Monte Carlo lower bound clears the target, then bisect.
    n = max(1000, naive or 1000)
    history = []
    while n <= HARD_CAP:
        result = joint_success(preservation, large, credit_preservation,
                               credit_large, need_preservation, need_large, n,
                               draws=max(800, args.draws // 4), seed=SEED)
        history.append((n, round(result["joint_probability"], 4)))
        if result["lower_confidence_bound"] >= TARGET:
            break
        n = int(n * 1.4) + 1
    else:
        report = {"analysis": "supplement_topup_budget",
                  "status": "INFEASIBLE_WITHIN_CAP", "hard_cap": HARD_CAP,
                  "search_history": history}
        atomic_write_json(args.out, report)
        print("INFEASIBLE within hard cap")
        return 1

    low = history[-2][0] if len(history) > 1 else 1000
    high = n
    while high - low > 100:
        middle = (low + high) // 2
        result = joint_success(preservation, large, credit_preservation,
                               credit_large, need_preservation, need_large,
                               middle, draws=max(800, args.draws // 4), seed=SEED)
        if result["lower_confidence_bound"] >= TARGET:
            high = middle
        else:
            low = middle + 1
    confirmed = joint_success(preservation, large, credit_preservation,
                              credit_large, need_preservation, need_large, high,
                              draws=args.draws, seed=SEED + 1)

    report = {
        "analysis": "supplement_topup_budget",
        "source": "supplement base run only -- same cohort, no pilot transfer",
        "why_not_pilot": (
            "the base shortfall came from applying pilot rates to a different "
            "word cohort; preservation ran at 0.142 against the pilot's 0.184 "
            "with fonts held equal, and per-word preservation inside the "
            "supplement spans 0.052 to 0.225"),
        "base_renderings": len(rows),
        "base_manifest_sha256": manifest["recipe_sha256"],
        "measured_rates": {
            "preservation_per_rendering": rate_preservation,
            "preservation_lower95": lower_preservation,
            "large_hallucination_per_rendering": rate_large,
            "large_hallucination_lower95": lower_large,
        },
        "credits": {"preservation": credit_preservation, "large": credit_large},
        "requirements": {"preservation": need_preservation, "large": need_large},
        "shortfalls": {"preservation": shortfall_preservation,
                       "large": shortfall_large},
        "naive_expectation_n": naive,
        "why_naive_is_wrong": (
            "349/0.0715 is the N where the expected preservation count reaches "
            "the quota, which succeeds about half the time, and it ignores the "
            "LARGE requirement entirely"),
        "search_history": history,
        "topup_max_renderings": high,
        "joint_probability_at_max": confirmed,
        "method": ("two-stage bootstrap of the base rows: resample the base, "
                   "then draw the top-up from it, so uncertainty in the base "
                   "rates is carried and the two events keep their observed "
                   "joint structure"),
        "hard_cap": HARD_CAP,
    }
    digest = atomic_write_json(args.out, report)

    print(f"base {len(rows):,} renderings")
    print(f"  preservation {int(preservation.sum())} "
          f"rate {rate_preservation:.6f} lower95 {lower_preservation:.6f}")
    print(f"  LARGE        {int(large.sum())} "
          f"rate {rate_large:.6f} lower95 {lower_large:.6f}")
    print(f"  shortfalls: preservation {shortfall_preservation}, "
          f"LARGE {shortfall_large}")
    print(f"\nnaive expectation N {naive:,} (insufficient: ~50% success, "
          f"ignores LARGE)")
    print(f"top-up max {high:,}")
    print(f"  joint P {confirmed['joint_probability']:.4f} "
          f"+/- {1.96 * confirmed['monte_carlo_stderr']:.4f}")
    print(f"  marginal preservation {confirmed['marginal_preservation']:.4f}, "
          f"LARGE {confirmed['marginal_large']:.4f}")
    print(f"report sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
