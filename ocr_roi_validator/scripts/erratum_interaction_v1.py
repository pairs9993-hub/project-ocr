"""Separate main effects from interaction, and correct the domain verdict.

The completed diagnostic showed geometry bins that differ and fonts that
differ. The earlier verdict logic took those two facts together as an
interaction, which does not follow: two main effects can coexist without the
geometry effect varying by font. Interaction is a distinct claim and is tested
here on its own.

Fifty events across eighteen font-by-stratum cells is thin, with five cells
empty, so the analysis is deliberately small: three macro strata rather than
seven bins, a main-effects logistic fit, and a likelihood-ratio comparison
against the same fit plus interaction terms. If the evidence cannot support a
conclusion, the answer is INCONCLUSIVE rather than a manufactured PASS or FAIL.

The original report's bytes are not touched. This writes a sidecar.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402
from ocr_roi_validator.interaction_stats import (  # noqa: E402
    detect_separation, fisher_exact, fit_logistic, holm_adjust,
    likelihood_ratio_test,
)

MACRO_STRATA = ("SMALL", "MEDIUM", "LARGE")


def macro_stratum(ink_height: int | None) -> str | None:
    """Pre-registered collapse of the seven bins, fixed before testing."""
    if ink_height is None:
        return None
    if ink_height < 14:
        return "SMALL"
    return "MEDIUM" if ink_height < 24 else "LARGE"


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * ((p * (1 - p) / trials
                   + z * z / (4 * trials * trials)) ** 0.5) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in (args.dir / "checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    # Only bare-e occurrences that reached a clean comparison can hallucinate,
    # so they are the denominator; anything else would dilute the rate with
    # samples that had no opportunity to show the effect.
    eligible = [r for r in rows
                if r["clean_eligible"] and r["visual_target"] == "e"
                and r["runtime_ink_height"] is not None]

    fonts = sorted({r["font"] for r in eligible})
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in eligible:
        key = (row["font"], macro_stratum(row["runtime_ink_height"]))
        cells[key][0] += 1
        cells[key][1] += int(row["clean_hallucination"])

    table = {f"{font}|{stratum}": {
        "exposure": cells[(font, stratum)][0],
        "events": cells[(font, stratum)][1],
        "rate": (cells[(font, stratum)][1] / cells[(font, stratum)][0]
                 if cells[(font, stratum)][0] else None),
        "ci95": wilson(cells[(font, stratum)][1], cells[(font, stratum)][0]),
    } for font in fonts for stratum in MACRO_STRATA}

    by_stratum = {s: [0, 0] for s in MACRO_STRATA}
    by_font = {f: [0, 0] for f in fonts}
    for (font, stratum), (exposure, events) in cells.items():
        by_stratum[stratum][0] += exposure
        by_stratum[stratum][1] += events
        by_font[font][0] += exposure
        by_font[font][1] += events

    # ---- design matrices: intercept, font dummies, stratum dummies ----------
    index_font = {f: i for i, f in enumerate(fonts)}
    index_stratum = {s: i for i, s in enumerate(MACRO_STRATA)}
    outcome = np.array([float(r["clean_hallucination"]) for r in eligible])
    main_columns = 1 + (len(fonts) - 1) + (len(MACRO_STRATA) - 1)
    main = np.zeros((len(eligible), main_columns))
    main[:, 0] = 1.0
    for position, row in enumerate(eligible):
        font_index = index_font[row["font"]]
        if font_index > 0:
            main[position, font_index] = 1.0
        stratum_index = index_stratum[macro_stratum(row["runtime_ink_height"])]
        if stratum_index > 0:
            main[position, len(fonts) - 1 + stratum_index] = 1.0

    interaction_columns = (len(fonts) - 1) * (len(MACRO_STRATA) - 1)
    full = np.zeros((len(eligible), main_columns + interaction_columns))
    full[:, :main_columns] = main
    for position, row in enumerate(eligible):
        font_index = index_font[row["font"]]
        stratum_index = index_stratum[macro_stratum(row["runtime_ink_height"])]
        if font_index > 0 and stratum_index > 0:
            offset = (font_index - 1) * (len(MACRO_STRATA) - 1) + (stratum_index - 1)
            full[position, main_columns + offset] = 1.0

    main_fit = fit_logistic(main, outcome)
    full_fit = fit_logistic(full, outcome)
    interaction_test = likelihood_ratio_test(main_fit, full_fit)

    # Main effects, each against an intercept-only model.
    intercept = fit_logistic(np.ones((len(eligible), 1)), outcome)
    font_only = np.zeros((len(eligible), len(fonts)))
    font_only[:, 0] = 1.0
    stratum_only = np.zeros((len(eligible), len(MACRO_STRATA)))
    stratum_only[:, 0] = 1.0
    for position, row in enumerate(eligible):
        font_index = index_font[row["font"]]
        if font_index > 0:
            font_only[position, font_index] = 1.0
        stratum_index = index_stratum[macro_stratum(row["runtime_ink_height"])]
        if stratum_index > 0:
            stratum_only[position, stratum_index] = 1.0
    font_test = likelihood_ratio_test(intercept, fit_logistic(font_only, outcome))
    stratum_test = likelihood_ratio_test(intercept,
                                         fit_logistic(stratum_only, outcome))

    # ---- pairwise Fisher with Holm correction -------------------------------
    font_pairs = {}
    for first, second in itertools.combinations(fonts, 2):
        a_events, a_exposure = by_font[first][1], by_font[first][0]
        b_events, b_exposure = by_font[second][1], by_font[second][0]
        font_pairs[f"{first} vs {second}"] = fisher_exact(
            ((a_events, a_exposure - a_events), (b_events, b_exposure - b_events)))
    stratum_pairs = {}
    for first, second in itertools.combinations(MACRO_STRATA, 2):
        a_events, a_exposure = by_stratum[first][1], by_stratum[first][0]
        b_events, b_exposure = by_stratum[second][1], by_stratum[second][0]
        stratum_pairs[f"{first} vs {second}"] = fisher_exact(
            ((a_events, a_exposure - a_events), (b_events, b_exposure - b_events)))
    font_adjusted = holm_adjust(font_pairs)
    stratum_adjusted = holm_adjust(stratum_pairs)

    separation = detect_separation({f"{f}|{s}": tuple(cells[(f, s)])
                                    for f in fonts for s in MACRO_STRATA})

    # ---- verdicts -----------------------------------------------------------
    alpha = 0.05
    geometry_effect = (stratum_test["p_value"] < alpha
                       and any(v < alpha for v in stratum_adjusted.values()))
    font_effect = (font_test["p_value"] < alpha
                   and any(v < alpha for v in font_adjusted.values()))

    # An interaction needs its own evidence, and with five empty cells the
    # likelihood-ratio test is not trustworthy even if it happens to be small.
    events = int(outcome.sum())
    if separation["quasi_separation_risk"] or events < 100:
        interaction_verdict = "INCONCLUSIVE"
        interaction_reason = (
            f"{events} events across {separation['cells']} cells with "
            f"{len(separation['cells_with_zero_events'])} empty; a "
            f"{interaction_columns}-parameter interaction cannot be identified "
            "at this event count")
    elif interaction_test["p_value"] < alpha:
        interaction_verdict = "CONFIRMED"
        interaction_reason = f"likelihood ratio p={interaction_test['p_value']:.4g}"
    else:
        interaction_verdict = "NOT_CONFIRMED"
        interaction_reason = f"likelihood ratio p={interaction_test['p_value']:.4g}"

    if interaction_verdict == "CONFIRMED":
        domain = "INTERACTION_DEPENDENT"
    elif geometry_effect and font_effect:
        domain = "MULTIFACTOR_DEPENDENT"
    elif geometry_effect:
        domain = "PIXEL_GEOMETRY_DEPENDENT"
    else:
        domain = "NOT_CONFIRMED"

    report = {
        "erratum": "statistical_erratum_v1",
        "supersedes_verdict_in": str(args.dir / "report.json"),
        "original_report_bytes_unmodified": True,
        "correction": (
            "the superseded verdict inferred INTERACTION_DEPENDENT from the "
            "coexistence of a geometry effect and a font effect. That does not "
            "follow: an interaction requires the geometry effect to differ by "
            "font, which is tested explicitly here."),
        "macro_strata": {"SMALL": "[0,14)", "MEDIUM": "[14,24)", "LARGE": "[24,inf)"},
        "denominator": "bare-e occurrences reaching a clean comparison",
        "total_events": events,
        "total_exposure": len(eligible),
        "cell_table": table,
        "by_macro_stratum": {s: {"exposure": v[0], "events": v[1],
                                 "rate": v[1] / v[0] if v[0] else None,
                                 "ci95": wilson(v[1], v[0])}
                             for s, v in by_stratum.items()},
        "by_font": {f: {"exposure": v[0], "events": v[1],
                        "rate": v[1] / v[0] if v[0] else None,
                        "ci95": wilson(v[1], v[0])}
                    for f, v in by_font.items()},
        "separation": separation,
        "tests": {
            "geometry_main_effect": stratum_test,
            "font_main_effect": font_test,
            "font_x_geometry_interaction": interaction_test,
            "interaction_parameters": interaction_columns,
        },
        "pairwise_fisher_holm": {
            "font": {"raw": font_pairs, "holm_adjusted": font_adjusted},
            "macro_stratum": {"raw": stratum_pairs,
                              "holm_adjusted": stratum_adjusted},
        },
        "PIXEL_GEOMETRY_EFFECT": "CONFIRMED" if geometry_effect else "NOT_CONFIRMED",
        "FONT_RATE_HETEROGENEITY": "CONFIRMED" if font_effect else "NOT_CONFIRMED",
        "FONT_X_GEOMETRY_INTERACTION": interaction_verdict,
        "interaction_reason": interaction_reason,
        "HALLUCINATION_DOMAIN": domain,
    }
    digest = atomic_write_json(args.out, report)

    print(f"events {events} over {len(eligible)} eligible, "
          f"{separation['cells']} cells "
          f"({len(separation['cells_with_zero_events'])} empty)")
    print(f"\n{'stratum':8s} {'exposure':>9s} {'events':>7s} {'rate':>8s}")
    for stratum in MACRO_STRATA:
        exposure, count = by_stratum[stratum]
        print(f"{stratum:8s} {exposure:9d} {count:7d} {count / exposure:8.4f}")
    print(f"\n{'font':14s} {'exposure':>9s} {'events':>7s} {'rate':>8s}")
    for font in fonts:
        exposure, count = by_font[font]
        print(f"{font:14s} {exposure:9d} {count:7d} {count / exposure:8.4f}")
    print(f"\ngeometry main effect   p={stratum_test['p_value']:.5g}")
    print(f"font main effect       p={font_test['p_value']:.5g}")
    print(f"interaction            p={interaction_test['p_value']:.5g} "
          f"(df={interaction_test['degrees_of_freedom']})")
    print(f"\nPIXEL_GEOMETRY_EFFECT       {report['PIXEL_GEOMETRY_EFFECT']}")
    print(f"FONT_RATE_HETEROGENEITY     {report['FONT_RATE_HETEROGENEITY']}")
    print(f"FONT_X_GEOMETRY_INTERACTION {interaction_verdict}")
    print(f"  {interaction_reason}")
    print(f"\nHALLUCINATION_DOMAIN        {domain}")
    print(f"erratum sha256              {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
