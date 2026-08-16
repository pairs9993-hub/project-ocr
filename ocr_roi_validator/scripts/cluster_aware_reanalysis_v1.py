"""Re-test the main effects and interaction accounting for matched clusters.

The likelihood-ratio results in the earlier erratum assumed independent
occurrences. They are not independent: each base condition was rendered in up
to six fonts, and rows sharing a base condition share the text, size, padding,
upscale, polarity, contrast, blur and seed. Hallucinations cluster accordingly.

Treating clustered rows as independent understates standard errors, so a
p-value computed that way is optimistic. Three cluster-aware procedures are run
and reported next to the naive result rather than replacing it silently, so the
size of the correction is visible.

The estimand is conditional and is stated as such in the output. It is

    P(clean single-position e -> é hallucination |
      detector found the line, the target line was matched, the CTC sequence
      aligned, and the comparison was clean)

which is not the unconditional rate at which the product hallucinates on a
screen. Roughly four fifths of eligible occurrences never reach that condition.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
for extra in (VALIDATOR_ROOT, VALIDATOR_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from erratum_interaction_v1 import MACRO_STRATA, macro_stratum  # noqa: E402
from ocr_roi_validator.cluster_inference import (  # noqa: E402
    cluster_bootstrap, cluster_permutation_test, cluster_summary,
    cluster_wald_test,
)
from ocr_roi_validator.diagnostic_runner import atomic_write_json  # noqa: E402
from ocr_roi_validator.interaction_stats import (  # noqa: E402
    detect_separation, fit_logistic, likelihood_ratio_test,
)

BOOTSTRAP_DRAWS = 2000
PERMUTATION_DRAWS = 5000
SEED = 20260816


def build_design(rows, fonts, include_interaction):
    """Intercept, font dummies, stratum dummies, optionally their products."""
    index_font = {f: i for i, f in enumerate(fonts)}
    index_stratum = {s: i for i, s in enumerate(MACRO_STRATA)}
    main_columns = 1 + (len(fonts) - 1) + (len(MACRO_STRATA) - 1)
    interaction_columns = ((len(fonts) - 1) * (len(MACRO_STRATA) - 1)
                           if include_interaction else 0)
    design = np.zeros((len(rows), main_columns + interaction_columns))
    design[:, 0] = 1.0
    for position, row in enumerate(rows):
        font_index = index_font[row["font"]]
        stratum_index = index_stratum[macro_stratum(row["runtime_ink_height"])]
        if font_index > 0:
            design[position, font_index] = 1.0
        if stratum_index > 0:
            design[position, len(fonts) - 1 + stratum_index] = 1.0
        if include_interaction and font_index > 0 and stratum_index > 0:
            offset = (font_index - 1) * (len(MACRO_STRATA) - 1) + (stratum_index - 1)
            design[position, main_columns + offset] = 1.0
    font_columns = list(range(1, len(fonts)))
    stratum_columns = list(range(len(fonts), main_columns))
    interaction_range = list(range(main_columns,
                                   main_columns + interaction_columns))
    return design, font_columns, stratum_columns, interaction_range


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--erratum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in (args.dir / "checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    eligible = [r for r in rows
                if r["clean_eligible"] and r["visual_target"] == "e"
                and r["runtime_ink_height"] is not None]
    fonts = sorted({r["font"] for r in eligible})

    outcome = np.array([float(r["clean_hallucination"]) for r in eligible])
    clusters = np.array([int(r["base_condition"]) for r in eligible])
    structure = cluster_summary(clusters, outcome)

    # How concentrated are the events? A handful of clusters carrying several
    # events each is exactly the situation independence assumptions mishandle.
    per_cluster = defaultdict(int)
    for cluster, value in zip(clusters, outcome):
        per_cluster[int(cluster)] += int(value)
    concentration = defaultdict(int)
    for count in per_cluster.values():
        if count:
            concentration[count] += 1

    main_design, font_columns, stratum_columns, _ = build_design(
        eligible, fonts, include_interaction=False)
    full_design, _, _, interaction_columns = build_design(
        eligible, fonts, include_interaction=True)

    # ---- naive, for side-by-side comparison ---------------------------------
    intercept = fit_logistic(np.ones((len(eligible), 1)), outcome)
    main_fit = fit_logistic(main_design, outcome)
    full_fit = fit_logistic(full_design, outcome)
    font_only, _, _, _ = build_design(eligible, fonts, False)
    naive = {
        "geometry_main_effect": likelihood_ratio_test(
            fit_logistic(np.delete(main_design, stratum_columns, axis=1), outcome),
            main_fit),
        "font_main_effect": likelihood_ratio_test(
            fit_logistic(np.delete(main_design, font_columns, axis=1), outcome),
            main_fit),
        "interaction": likelihood_ratio_test(main_fit, full_fit),
        "note": "assumes independent occurrences; retained only for comparison",
    }

    # ---- cluster-robust Wald -------------------------------------------------
    wald = {
        "geometry_main_effect": cluster_wald_test(main_design, outcome, clusters,
                                                  stratum_columns),
        "font_main_effect": cluster_wald_test(main_design, outcome, clusters,
                                              font_columns),
        "interaction": cluster_wald_test(full_design, outcome, clusters,
                                         interaction_columns),
    }

    # ---- cluster bootstrap on the SMALL-vs-rest contrast ---------------------
    # SMALL is the reference level, so each stratum coefficient is that
    # stratum against SMALL; MEDIUM is the largest single contrast.
    medium_contrast = np.zeros(main_design.shape[1])
    medium_contrast[stratum_columns[0]] = 1.0
    large_contrast = np.zeros(main_design.shape[1])
    large_contrast[stratum_columns[1]] = 1.0
    bootstrap = {
        "medium_vs_small": cluster_bootstrap(
            main_design, outcome, clusters, medium_contrast,
            draws=BOOTSTRAP_DRAWS, seed=SEED),
        "large_vs_small": cluster_bootstrap(
            main_design, outcome, clusters, large_contrast,
            draws=BOOTSTRAP_DRAWS, seed=SEED),
    }

    # ---- permutation, preserving matched blocks ------------------------------
    # Stratum is a property of the rendered condition and is constant within a
    # base condition, so it can be permuted across clusters. Font varies within
    # a cluster by construction, so cluster permutation does not apply to it and
    # is not attempted.
    stratum_labels = np.array(
        [float(MACRO_STRATA.index(macro_stratum(r["runtime_ink_height"])))
         for r in eligible])
    font_labels = np.array([float(fonts.index(r["font"])) for r in eligible])

    def spread_statistic(values, labels):
        rates = [values[labels == level].mean()
                 for level in np.unique(labels) if (labels == level).any()]
        return max(rates) - min(rates) if len(rates) > 1 else 0.0

    permutation = {
        "geometry": cluster_permutation_test(
            outcome, clusters, stratum_labels, spread_statistic,
            draws=PERMUTATION_DRAWS, seed=SEED),
        "font": cluster_permutation_test(
            outcome, clusters, font_labels, spread_statistic,
            draws=PERMUTATION_DRAWS, seed=SEED),
    }

    cells = defaultdict(lambda: [0, 0])
    for row in eligible:
        key = f"{row['font']}|{macro_stratum(row['runtime_ink_height'])}"
        cells[key][0] += 1
        cells[key][1] += int(row["clean_hallucination"])
    separation = detect_separation({k: tuple(v) for k, v in cells.items()})

    alpha = 0.05

    def verdict(wald_result, bootstrap_results=None, permutation_result=None):
        """Confirm only when the cluster-aware evidence actually supports it."""
        if not wald_result.get("reliable"):
            return "INCONCLUSIVE"
        signals = [wald_result["p_value"] < alpha]
        if bootstrap_results:
            usable = [b for b in bootstrap_results if b.get("reliable")]
            if not usable:
                return "INCONCLUSIVE"
            signals.append(any(b["excludes_zero"] for b in usable))
        if permutation_result and permutation_result.get("reliable"):
            signals.append(permutation_result["p_value"] < alpha)
        return "CONFIRMED" if all(signals) else "NOT_CONFIRMED"

    geometry_verdict = verdict(
        wald["geometry_main_effect"],
        [bootstrap["medium_vs_small"], bootstrap["large_vs_small"]],
        permutation["geometry"])
    font_verdict = verdict(wald["font_main_effect"])
    if separation["quasi_separation_risk"] or int(outcome.sum()) < 100:
        interaction_verdict = "INCONCLUSIVE"
        interaction_reason = (
            f"{int(outcome.sum())} events over {separation['cells']} cells with "
            f"{len(separation['cells_with_zero_events'])} empty; "
            f"{len(interaction_columns)} interaction parameters are not "
            "identifiable at this event count, clustered or not")
    elif wald["interaction"].get("p_value") is not None \
            and wald["interaction"]["p_value"] < alpha:
        interaction_verdict = "CONFIRMED"
        interaction_reason = f"cluster-robust Wald p={wald['interaction']['p_value']:.4g}"
    else:
        interaction_verdict = "INCONCLUSIVE"
        interaction_reason = "insufficient evidence either way"

    if interaction_verdict == "CONFIRMED":
        domain = "INTERACTION_DEPENDENT"
    elif geometry_verdict == "CONFIRMED" and font_verdict == "CONFIRMED":
        domain = "MULTIFACTOR_DEPENDENT"
    elif geometry_verdict == "CONFIRMED":
        domain = "PIXEL_GEOMETRY_DEPENDENT"
    elif font_verdict == "CONFIRMED":
        domain = "FONT_RATE_DEPENDENT"
    else:
        domain = "NOT_CONFIRMED"

    report = {
        "analysis": "cluster_aware_reanalysis_v1",
        "supersedes_inference_in": str(args.erratum),
        "prior_bytes_unmodified": True,
        "estimand": (
            "P(clean single-position e->é hallucination | detector found the "
            "line, target line matched, CTC sequence aligned, clean "
            "comparison). This is conditional. It is NOT the unconditional "
            "rate at which the product hallucinates on a screen: only "
            f"{len(eligible)} of {len(rows)} eligible occurrences reached this "
            "condition."),
        "cluster_key": "base_condition (all render settings except font)",
        "cluster_structure": structure,
        "events_per_cluster_distribution": dict(sorted(concentration.items())),
        "separation": separation,
        "naive_independent_tests": naive,
        "cluster_robust_wald": wald,
        "cluster_bootstrap": bootstrap,
        "cluster_permutation": permutation,
        "settings": {"bootstrap_draws": BOOTSTRAP_DRAWS,
                     "permutation_draws": PERMUTATION_DRAWS, "seed": SEED},
        "CLUSTER_AWARE_GEOMETRY_EFFECT": geometry_verdict,
        "CLUSTER_AWARE_FONT_EFFECT": font_verdict,
        "CLUSTER_AWARE_INTERACTION": interaction_verdict,
        "interaction_reason": interaction_reason,
        "HALLUCINATION_DOMAIN": domain,
    }
    digest = atomic_write_json(args.out, report)

    print(f"clusters {structure['clusters']}, rows {structure['rows']}, "
          f"events {structure['total_events']} in "
          f"{structure['clusters_with_events']} clusters "
          f"(max {structure['max_events_in_one_cluster']} in one)")
    print(f"\n{'test':22s} {'naive p':>12s} {'cluster p':>12s}")
    for name, key in (("geometry main", "geometry_main_effect"),
                      ("font main", "font_main_effect"),
                      ("interaction", "interaction")):
        naive_p = naive[key]["p_value"]
        cluster_p = wald[key].get("p_value")
        shown = f"{cluster_p:.5g}" if cluster_p is not None else "n/a"
        print(f"{name:22s} {naive_p:12.5g} {shown:>12s}")
    print("\nbootstrap (log-odds vs SMALL):")
    for name, result in bootstrap.items():
        if result.get("reliable"):
            low, high = result["ci95"]
            print(f"  {name:18s} {result['observed']:+.3f} "
                  f"CI[{low:+.3f},{high:+.3f}] excludes0={result['excludes_zero']} "
                  f"degenerate={result['degenerate_draws']}")
        else:
            print(f"  {name:18s} unreliable: {result.get('note')}")
    print("\npermutation:")
    for name, result in permutation.items():
        if result.get("reliable"):
            print(f"  {name:10s} p={result['p_value']:.5g}")
        else:
            print(f"  {name:10s} not applicable: {result.get('note')}")
    print(f"\nCLUSTER_AWARE_GEOMETRY_EFFECT {geometry_verdict}")
    print(f"CLUSTER_AWARE_FONT_EFFECT     {font_verdict}")
    print(f"CLUSTER_AWARE_INTERACTION     {interaction_verdict}")
    print(f"HALLUCINATION_DOMAIN          {domain}")
    print(f"report sha256                 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
