"""Budget the v2 datasets under the redefined quotas.

Two changes from the previous sizing, both consequential.

Per-font clean-hallucination minima are gone. Requiring twenty hallucinations
from consola, whose baseline produces three in 3,854 renderings, measured how
reliably that font triggers the defect rather than how safely the verifier
handles it -- and it put the preflight budget at 429,108 renderings or made it
unsatisfiable outright.

The font-level safety gate moves to legitimate-accent preservation: cases where
an accent was drawn and the baseline read it correctly. Those are where a
verifier could do real harm by "correcting" a genuine é to a bare e, and they
are plentiful in exactly the fonts that hallucinate least. consola has the worst
hallucination rate of the thirteen and the best preservation rate, so the gate
that matters is well supported precisely where the old one failed.

Budgets are also no longer summed across quotas. One rendering can satisfy a
stratum quota, a font preservation quota and an UNKNOWN quota at once, so each
split's budget is the largest single requirement, not the total of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402
from size_v2_budget import (  # noqa: E402
    binomial_at_least, renderings_for_quota, wilson_lower,
)

MACRO_STRATA = ("SMALL", "MEDIUM", "LARGE")
LOW_SUPPORT_THRESHOLD = 5

# train_v1's backfilled hallucinations, verified by re-render parity, count
# toward the supplement's stratum quotas.
TRAIN_V1_BACKFILL = {"SMALL": 107, "MEDIUM": 82, "LARGE": 11}

QUOTAS = {
    "supplement": {
        "hallucination_total": 300,
        "hallucination_per_stratum": 60,
        "preservation_total": 2000,
        "preservation_per_font": 0,
        "preservation_per_stratum": 0,
        "unknown_total": 2000,
        "credit": TRAIN_V1_BACKFILL,
    },
    "calibration": {
        "hallucination_total": 100,
        "hallucination_per_stratum": 20,
        "preservation_total": 500,
        "preservation_per_font": 100,
        "preservation_per_stratum": 100,
        "unknown_total": 500,
        "credit": {},
    },
    "preflight": {
        "hallucination_total": 100,
        "hallucination_per_stratum": 20,
        "preservation_total": 1000,
        "preservation_per_font": 200,
        "preservation_per_stratum": 200,
        "unknown_total": 1000,
        "credit": {},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in (args.pilot / "checkpoint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    recipe = json.loads((args.pilot / "recipe.json").read_text(encoding="utf-8"))
    split_fonts = recipe["split_fonts"]

    unknown_reasons = {"DELETION", "INSERTION", "MULTIPLE_CHANGES",
                       "CHANGE_ELSEWHERE", "OTHER_SUBSTITUTION", "ACCENT_LOST"}

    # (font, stratum) -> exposures and the three event types.
    cells: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"rendered": 0, "landed": 0, "hallucination": 0,
                 "preservation": 0, "unknown": 0})
    for row in rows:
        cell = cells[(row["font"], row["target_stratum"])]
        cell["rendered"] += 1
        if row["measured_stratum"] == row["target_stratum"]:
            cell["landed"] += 1
        cell["hallucination"] += int(row["clean_hallucination"])
        cell["preservation"] += int(row["clean_preservation"])
        cell["unknown"] += int(row["terminal_reason"] in unknown_reasons)

    cell_report = {}
    low_support = []
    for (font, stratum), cell in sorted(cells.items()):
        key = f"{font}|{stratum}"
        entry = dict(cell)
        entry["targeting_rate"] = cell["landed"] / cell["rendered"]
        for kind in ("hallucination", "preservation", "unknown"):
            entry[f"{kind}_rate"] = cell[kind] / cell["rendered"]
            entry[f"{kind}_lower95"] = wilson_lower(cell[kind], cell["rendered"])
        if cell["preservation"] < LOW_SUPPORT_THRESHOLD:
            entry["support"] = "LOW_SUPPORT"
            low_support.append(key)
        else:
            entry["support"] = "OK"
        cell_report[key] = entry

    budgets = {}
    infeasible = []
    for split, quota in QUOTAS.items():
        fonts = split_fonts[split]
        share = 1.0 / len(fonts)          # exposure is split evenly by design
        requirements: dict[str, dict] = {}

        def rate_over(kind: str, subset) -> tuple[int, int]:
            attempted = sum(cells[key]["rendered"] for key in subset)
            events = sum(cells[key][kind] for key in subset)
            return events, attempted

        # Aggregate hallucination, net of any credited backfill.
        credit = quota["credit"]
        needed = max(0, quota["hallucination_total"] - sum(credit.values()))
        events, attempted = rate_over(
            "hallucination", [(f, s) for f in fonts for s in MACRO_STRATA])
        lower = wilson_lower(events, attempted)
        requirements["hallucination_total"] = {
            "required": needed, "pilot_events": events,
            "pilot_renderings": attempted, "rate_lower95": lower,
            "renderings": renderings_for_quota(lower, needed) if needed else 0,
        }

        # Per-stratum hallucination. A rendering aimed at one stratum does not
        # help another, so the requirement is scaled by that stratum's share.
        for stratum in MACRO_STRATA:
            still = max(0, quota["hallucination_per_stratum"]
                        - credit.get(stratum, 0))
            events, attempted = rate_over("hallucination",
                                          [(f, stratum) for f in fonts])
            lower = wilson_lower(events, attempted)
            needed_here = (renderings_for_quota(lower, still) if still else 0)
            requirements[f"hallucination_{stratum}"] = {
                "required": still, "pilot_events": events,
                "pilot_renderings": attempted, "rate_lower95": lower,
                # scaled to the whole split, since only a third of renderings
                # target this stratum
                "renderings": (needed_here * len(MACRO_STRATA)
                               if needed_here else 0),
            }

        # Per-font preservation: the safety gate.
        if quota["preservation_per_font"]:
            for font in fonts:
                events, attempted = rate_over(
                    "preservation", [(font, s) for s in MACRO_STRATA])
                lower = wilson_lower(events, attempted)
                needed_here = renderings_for_quota(
                    lower, quota["preservation_per_font"])
                requirements[f"preservation_font_{font}"] = {
                    "required": quota["preservation_per_font"],
                    "pilot_events": events, "pilot_renderings": attempted,
                    "rate_lower95": lower,
                    "renderings": (int(needed_here / share) if needed_here
                                   else None),
                }

        # Per-stratum preservation.
        if quota["preservation_per_stratum"]:
            for stratum in MACRO_STRATA:
                events, attempted = rate_over("preservation",
                                              [(f, stratum) for f in fonts])
                lower = wilson_lower(events, attempted)
                needed_here = renderings_for_quota(
                    lower, quota["preservation_per_stratum"])
                requirements[f"preservation_{stratum}"] = {
                    "required": quota["preservation_per_stratum"],
                    "pilot_events": events, "pilot_renderings": attempted,
                    "rate_lower95": lower,
                    "renderings": (needed_here * len(MACRO_STRATA)
                                   if needed_here else None),
                }

        for name, kind in (("preservation_total", "preservation"),
                           ("unknown_total", "unknown")):
            events, attempted = rate_over(
                kind, [(f, s) for f in fonts for s in MACRO_STRATA])
            lower = wilson_lower(events, attempted)
            requirements[name] = {
                "required": quota[name], "pilot_events": events,
                "pilot_renderings": attempted, "rate_lower95": lower,
                "renderings": renderings_for_quota(lower, quota[name]),
            }

        unmet = [k for k, v in requirements.items() if v["renderings"] is None]
        if unmet:
            infeasible.extend(f"{split}|{k}" for k in unmet)
            total = None
            binding = None
        else:
            # One rendering can satisfy several quotas at once, so the budget
            # is the largest single requirement rather than their sum.
            total = max(v["renderings"] for v in requirements.values())
            binding = max(requirements, key=lambda k: requirements[k]["renderings"])
        budgets[split] = {
            "fonts": fonts, "requirements": requirements,
            "binding_constraint": binding, "total_renderings": total,
            "credit_applied": credit,
        }

    feasible = not infeasible
    grand_total = (sum(b["total_renderings"] for b in budgets.values())
                   if feasible else None)
    bytes_per_row = ((args.pilot / "checkpoint.jsonl").stat().st_size
                     / max(1, len(rows)))
    seconds_per_rendering = 10705 / 60000        # measured on this pilot

    report = {
        "analysis": "v2_budget_redefined_quotas",
        "FONT_HALLUCINATION_QUOTA_POLICY": "AGGREGATE_ONLY",
        "FONT_SAFETY_QUOTA_POLICY": "LEGITIMATE_ACCENT_PRESERVATION",
        "rationale": (
            "per-font hallucination minima measured how reliably a font "
            "triggers the defect, not how safely the verifier treats it. The "
            "font-level risk is mis-correcting a genuine é, so the gate moves "
            "to legitimate-accent preservation, which is best supported in the "
            "fonts that hallucinate least."),
        "rate_caveat": (
            "SMALL_RATE, MEDIUM_RATE and LARGE_RATE are yields of a recipe "
            "chosen to collect quota, not the product's natural hallucination "
            "prevalence, and must not be read as such."),
        "budget_composition": (
            "per-split budget is the largest single requirement, not the sum: "
            "one rendering can satisfy stratum, font and UNKNOWN quotas at once"),
        "pilot_rows": len(rows),
        "train_v1_credit": TRAIN_V1_BACKFILL,
        "font_stratum_cells": cell_report,
        "low_support_cells": low_support,
        "budgets": budgets,
        "infeasible_requirements": infeasible,
        "ALL_QUOTAS_95P_FEASIBLE": feasible,
        "TOTAL_RENDERINGS": grand_total,
        "ESTIMATED_RUNTIME_HOURS": (round(grand_total * seconds_per_rendering
                                          / 3600, 2) if grand_total else None),
        "ESTIMATED_STORAGE_MB": (round(grand_total * bytes_per_row / 1024 ** 2, 1)
                                 if grand_total else None),
    }
    digest = atomic_write_json(args.out, report)

    for split, entry in budgets.items():
        print(f"\n{split} (fonts: {len(entry['fonts'])})")
        for name, requirement in sorted(
                entry["requirements"].items(),
                key=lambda item: -(item[1]["renderings"] or 0)):
            shown = (f"{requirement['renderings']:,}"
                     if requirement["renderings"] is not None else "INFEASIBLE")
            print(f"  {name:34s} need {requirement['required']:5d} "
                  f"ev {requirement['pilot_events']:5d} "
                  f"lower {requirement['rate_lower95']:.6f} -> {shown:>12s}")
        total = entry["total_renderings"]
        print(f"  {'BUDGET (max, not sum)':34s} "
              f"{(f'{total:,}' if total else 'INFEASIBLE'):>44s}")
        print(f"  binding: {entry['binding_constraint']}")
    print(f"\nlow-support cells: {len(low_support)}")
    print(f"ALL_QUOTAS_95P_FEASIBLE {feasible}")
    if grand_total:
        print(f"TOTAL_RENDERINGS {grand_total:,}")
        print(f"ESTIMATED_RUNTIME {report['ESTIMATED_RUNTIME_HOURS']} h")
        print(f"ESTIMATED_STORAGE {report['ESTIMATED_STORAGE_MB']} MB")
    print(f"report sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
