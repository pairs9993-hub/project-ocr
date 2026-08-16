"""Size the v2 generation budget on quota-success probability, not expectation.

Dividing the required event count by a rate lower bound gives the N at which the
*expected* number of events equals the quota. At that N the quota is met about
half the time, because roughly half of all binomial draws land below their mean.
Sizing that way would have produced a budget that fails as often as it succeeds.

What is computed instead is the smallest N with

    P[Binomial(N, p_lower) >= required] >= 0.95

using a one-sided 95% lower bound for p, so the rate uncertainty and the
sampling variability are both carried. Two sources of uncertainty compose here:
a rendering aimed at a stratum does not always land in it, and a rendering that
lands there does not always hallucinate. Both are estimated from the pilot and
multiplied, with the lower bound taken on the product.

If a cell's lower bound is zero -- no events observed -- no budget is emitted
for it. An N cannot be derived from a rate that might be zero, and inventing one
would be worse than reporting that the pilot was too small.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402

MACRO_STRATA = ("SMALL", "MEDIUM", "LARGE")
SUCCESS_PROBABILITY = 0.95
MAX_BUDGET_PER_CELL = 5_000_000          # refuse rather than emit a silly N

# Quotas as specified. train_v1's backfilled hallucinations count toward the
# supplement's stratum quotas, so the shortfall is what must still be generated.
QUOTAS = {
    "supplement": {"SMALL": 60, "MEDIUM": 60, "LARGE": 60},
    "calibration": {"SMALL": 20, "MEDIUM": 20, "LARGE": 20},
    "preflight": {"SMALL": 20, "MEDIUM": 20, "LARGE": 20},
}


def wilson_lower(successes: int, trials: int, z: float = 1.6449) -> float:
    """One-sided 95% lower bound on a proportion."""
    if trials == 0:
        return 0.0
    if successes == 0:
        return 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - spread)


def binomial_at_least(n: int, p: float, required: int) -> float:
    """P[Binomial(n, p) >= required], summed from the tail that is shorter."""
    if required <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0 if n >= required else 0.0
    if n < required:
        return 0.0
    # Sum the lower tail and subtract; required is small relative to n here.
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for k in range(required):
        log_term = (math.lgamma(n + 1) - math.lgamma(k + 1)
                    - math.lgamma(n - k + 1) + k * log_p + (n - k) * log_q)
        total += math.exp(log_term)
        if total >= 1.0:
            return 0.0
    return max(0.0, 1.0 - total)


def renderings_for_quota(rate_lower: float, required: int,
                         confidence: float = SUCCESS_PROBABILITY) -> int | None:
    """Smallest N whose quota-success probability reaches ``confidence``."""
    if rate_lower <= 0.0 or required <= 0:
        return None
    # Start from the expectation-matching N and grow until the probability
    # requirement is actually met.
    n = max(required, int(math.ceil(required / rate_lower)))
    # The ceiling has to be checked before the search, not only inside the
    # doubling loop: a rate small enough to need billions of renderings starts
    # above the cap and would otherwise be returned as a real budget.
    if n > MAX_BUDGET_PER_CELL:
        return None
    if binomial_at_least(n, rate_lower, required) >= confidence:
        low, high = required, n
    else:
        low = n
        high = n * 2
        while binomial_at_least(high, rate_lower, required) < confidence:
            low = high
            high *= 2
            if high > MAX_BUDGET_PER_CELL:
                return None
    while low < high:
        middle = (low + high) // 2
        if binomial_at_least(middle, rate_lower, required) >= confidence:
            high = middle
        else:
            low = middle + 1
    return low


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supplement-shortfall", type=Path,
                        help="sidecar summary giving train_v1 stratum counts")
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in (args.pilot / "checkpoint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    recipe = json.loads((args.pilot / "recipe.json").read_text(encoding="utf-8"))
    split_fonts = recipe["split_fonts"]

    # Per (font, target stratum): how often a rendering both lands in the
    # intended stratum and hallucinates. That joint rate is what a budget
    # actually consumes, so it is measured directly rather than as a product of
    # two separately estimated factors.
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        key = (row["font"], row["target_stratum"])
        cells[key][0] += 1
        landed = row["measured_stratum"] == row["target_stratum"]
        if landed:
            cells[key][1] += 1
            if row["clean_hallucination"]:
                cells[key][2] += 1

    cell_report = {}
    for (font, stratum), (attempted, landed, events) in sorted(cells.items()):
        joint = events / attempted if attempted else 0.0
        cell_report[f"{font}|{stratum}"] = {
            "renderings": attempted,
            "landed_in_stratum": landed,
            "targeting_rate": landed / attempted if attempted else 0.0,
            "hallucinations": events,
            "joint_rate": joint,
            "joint_rate_lower95": wilson_lower(events, attempted),
        }

    # Each split only uses its own fonts, so its rate is pooled over those.
    budgets = {}
    unidentifiable = []
    for split, quota in QUOTAS.items():
        fonts = split_fonts[split]
        split_entry = {}
        for stratum in MACRO_STRATA:
            attempted = sum(cells[(f, stratum)][0] for f in fonts)
            events = sum(cells[(f, stratum)][2] for f in fonts)
            landed = sum(cells[(f, stratum)][1] for f in fonts)
            lower = wilson_lower(events, attempted)
            required = quota[stratum]
            needed = renderings_for_quota(lower, required)
            if needed is None:
                unidentifiable.append(f"{split}|{stratum}")
            split_entry[stratum] = {
                "required_events": required,
                "pilot_renderings": attempted,
                "pilot_landed": landed,
                "pilot_events": events,
                "joint_rate_point": events / attempted if attempted else 0.0,
                "joint_rate_lower95": lower,
                "renderings_for_95pct_quota": needed,
                "success_probability_at_budget": (
                    binomial_at_least(needed, lower, required)
                    if needed else None),
                "fonts": fonts,
            }
        split_entry["total_renderings"] = (
            None if any(v.get("renderings_for_95pct_quota") is None
                        for k, v in split_entry.items() if k in MACRO_STRATA)
            else sum(v["renderings_for_95pct_quota"] for k, v in split_entry.items()
                     if k in MACRO_STRATA))
        budgets[split] = split_entry

    identifiable = not unidentifiable
    grand_total = (sum(b["total_renderings"] for b in budgets.values())
                   if identifiable else None)

    # Throughput and size measured from the pilot itself rather than guessed.
    checkpoint = args.pilot / "checkpoint.jsonl"
    bytes_per_row = checkpoint.stat().st_size / max(1, len(rows))

    report = {
        "analysis": "v2_budget_from_quota_success_probability",
        "method": (
            "smallest N with P[Binomial(N, p_lower) >= required] >= 0.95, where "
            "p_lower is the one-sided 95% Wilson lower bound on the joint rate "
            "of landing in the target stratum AND hallucinating"),
        "why_not_expectation": (
            "required/rate gives the N where the expected count equals the "
            "quota, which succeeds only about half the time"),
        "success_probability_target": SUCCESS_PROBABILITY,
        "pilot_rows": len(rows),
        "pilot_recipe_sha256": recipe.get("recipe_sha256"),
        "per_font_stratum": cell_report,
        "budgets": budgets,
        "BUDGET_STATUS": ("IDENTIFIED" if identifiable else "BUDGET_UNIDENTIFIABLE"),
        "unidentifiable_cells": unidentifiable,
        "total_renderings": grand_total,
        "bytes_per_row_measured": round(bytes_per_row, 1),
        "estimated_storage_bytes": (int(grand_total * bytes_per_row)
                                    if grand_total else None),
    }
    digest = atomic_write_json(args.out, report)

    print(f"pilot rows {len(rows)}")
    print(f"\n{'split':12s} {'stratum':8s} {'rend':>7s} {'land':>6s} {'ev':>4s} "
          f"{'rate':>9s} {'lower95':>9s} {'N for 95%':>11s}")
    for split, entry in budgets.items():
        for stratum in MACRO_STRATA:
            cell = entry[stratum]
            needed = cell["renderings_for_95pct_quota"]
            shown = f"{needed:,}" if needed else "UNIDENT"
            print(f"{split:12s} {stratum:8s} {cell['pilot_renderings']:7d} "
                  f"{cell['pilot_landed']:6d} {cell['pilot_events']:4d} "
                  f"{cell['joint_rate_point']:9.6f} "
                  f"{cell['joint_rate_lower95']:9.6f} {shown:>11s}")
        total = entry["total_renderings"]
        print(f"{split:12s} {'TOTAL':8s} {'':7s} {'':6s} {'':4s} {'':9s} {'':9s} "
              f"{(f'{total:,}' if total else 'UNIDENT'):>11s}")
    print(f"\nBUDGET_STATUS {report['BUDGET_STATUS']}")
    if grand_total:
        print(f"total renderings {grand_total:,}")
        print(f"estimated storage {report['estimated_storage_bytes'] / 1024**2:.1f} MB")
    print(f"report sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
