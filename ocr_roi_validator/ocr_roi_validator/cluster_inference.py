"""Cluster-aware inference for matched diagnostic data.

The 19,200 rows are not 19,200 independent observations. Each base condition --
one text, size, padding, upscale, polarity, contrast, blur and seed -- was
rendered in up to six fonts, so rows sharing a base condition share almost
everything except the typeface. Hallucinations bunch accordingly: of the 41
clusters that produced any, four produced three or four apiece.

Treating those as independent understates the standard errors, which makes a
p-value look smaller than the data earns. Two corrections are offered here.

``cluster_bootstrap``
    Resample whole base conditions with replacement, keeping each cluster's
    rows together. Makes no distributional assumption about the clustering.

``cluster_robust_covariance``
    The sandwich estimator, summing the score contributions within a cluster
    before taking the outer product.

``cluster_permutation_test``
    Permutes a label across clusters rather than across rows, preserving the
    matched blocks. Used where a font-level label is constant within a cluster.

None of these manufacture power. If the events are too few or too concentrated
to separate hypotheses, they will say so by returning wide intervals and large
p-values, and that is the correct answer rather than a failure.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ocr_roi_validator.interaction_stats import chi_square_sf, fit_logistic

__all__ = [
    "cluster_summary",
    "cluster_bootstrap",
    "cluster_robust_covariance",
    "cluster_wald_test",
    "cluster_permutation_test",
]


def cluster_summary(clusters: np.ndarray, outcome: np.ndarray) -> dict:
    """Describe the cluster structure so its influence is visible, not implied."""
    sizes: dict[int, int] = defaultdict(int)
    events: dict[int, int] = defaultdict(int)
    for cluster, value in zip(clusters, outcome):
        sizes[int(cluster)] += 1
        events[int(cluster)] += int(value)
    size_values = list(sizes.values())
    event_values = [v for v in events.values() if v > 0]
    return {
        "clusters": len(sizes),
        "rows": int(len(outcome)),
        "rows_per_cluster_min": min(size_values) if size_values else 0,
        "rows_per_cluster_max": max(size_values) if size_values else 0,
        "rows_per_cluster_mean": (sum(size_values) / len(size_values)
                                  if size_values else 0.0),
        "clusters_with_events": len(event_values),
        "max_events_in_one_cluster": max(event_values) if event_values else 0,
        "total_events": int(outcome.sum()),
    }


def cluster_bootstrap(design: np.ndarray, outcome: np.ndarray,
                      clusters: np.ndarray, contrast: np.ndarray,
                      draws: int = 2000, seed: int = 20260816) -> dict:
    """Bootstrap a linear contrast of the coefficients over whole clusters.

    Draws that fail to converge, or that contain no events at all, are counted
    and excluded rather than quietly contributing a degenerate estimate.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    index_by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    base = fit_logistic(design, outcome)
    observed = float(contrast @ base.coefficients)

    estimates: list[float] = []
    degenerate = 0
    for _ in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_cluster[c] for c in picked])
        sample_outcome = outcome[rows]
        if sample_outcome.sum() == 0 or sample_outcome.sum() == len(rows):
            degenerate += 1
            continue
        fit = fit_logistic(design[rows], sample_outcome)
        if not fit.converged:
            degenerate += 1
            continue
        estimates.append(float(contrast @ fit.coefficients))

    if len(estimates) < draws * 0.5:
        return {
            "observed": observed, "draws": draws, "usable_draws": len(estimates),
            "degenerate_draws": degenerate, "seed": seed,
            "ci95": None, "p_value": None,
            "reliable": False,
            "note": "more than half the bootstrap draws were degenerate",
        }
    values = np.array(estimates)
    low, high = np.percentile(values, [2.5, 97.5])
    # Two-sided p from the bootstrap distribution's overlap with zero.
    proportion = float(np.mean(values <= 0.0))
    p_value = 2.0 * min(proportion, 1.0 - proportion)
    return {
        "observed": observed,
        "draws": draws,
        "usable_draws": len(estimates),
        "degenerate_draws": degenerate,
        "seed": seed,
        "ci95": [float(low), float(high)],
        "p_value": float(min(1.0, p_value)),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "reliable": True,
    }


def cluster_robust_covariance(design: np.ndarray, outcome: np.ndarray,
                              clusters: np.ndarray,
                              coefficients: np.ndarray) -> np.ndarray:
    """Sandwich covariance with scores summed within each cluster."""
    design = np.asarray(design, dtype=float)
    eta = np.clip(design @ coefficients, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    weights = np.clip(mu * (1.0 - mu), 1e-10, None)
    bread = np.linalg.pinv((design.T * weights) @ design)

    residual = outcome - mu
    meat = np.zeros((design.shape[1], design.shape[1]))
    for cluster in np.unique(clusters):
        rows = np.where(clusters == cluster)[0]
        score = design[rows].T @ residual[rows]
        meat += np.outer(score, score)

    count = len(np.unique(clusters))
    correction = count / max(1, count - 1)
    return bread @ meat @ bread * correction


def cluster_wald_test(design: np.ndarray, outcome: np.ndarray,
                      clusters: np.ndarray, columns: list[int]) -> dict:
    """Wald test on a set of coefficients using cluster-robust covariance."""
    fit = fit_logistic(design, outcome)
    covariance = cluster_robust_covariance(design, outcome, clusters,
                                           fit.coefficients)
    selected = np.array(columns, dtype=int)
    beta = fit.coefficients[selected]
    block = covariance[np.ix_(selected, selected)]

    # A pseudo-inverse will happily invert a rank-deficient block and return an
    # enormous statistic. That is not evidence -- it means some tested columns
    # carry no information (here, interaction cells with zero events), and the
    # resulting p-value is an artifact of the inversion rather than a finding.
    # Rank is checked first, and an under-determined test declines to report.
    rank = int(np.linalg.matrix_rank(block, tol=1e-10))
    conditioning = float(np.linalg.cond(block)) if rank == len(columns) else float("inf")

    # Rank and conditioning both look healthy when a tested column simply has
    # no events on it: the column is linearly independent, the covariance block
    # inverts cleanly, and yet the coefficient is pushed to an extreme because
    # nothing bounds it from one side. That produced a Wald statistic of 1833
    # and p=0 on data with two per cent prevalence. Checking the columns
    # against the outcome catches what the matrix diagnostics cannot.
    starved = []
    for column in columns:
        active = design[:, column] != 0
        if active.sum() == 0:
            starved.append(int(column))
        elif outcome[active].sum() == 0 or outcome[active].sum() == active.sum():
            starved.append(int(column))
    if starved:
        return {
            "statistic": None, "p_value": None,
            "degrees_of_freedom": len(columns),
            "rank": rank, "condition_number": conditioning,
            "converged": fit.converged, "separation_detected": True,
            "starved_columns": starved,
            "clusters": int(len(np.unique(clusters))),
            "reliable": False,
            "note": (f"columns {starved} have no outcome variation on them; "
                     "their coefficients are unbounded and the Wald statistic "
                     "would be an artifact"),
        }
    if rank < len(columns) or conditioning > 1e12:
        return {
            "statistic": None, "p_value": None,
            "degrees_of_freedom": len(columns),
            "rank": rank, "condition_number": conditioning,
            "converged": fit.converged, "separation_detected": fit.separated,
            "clusters": int(len(np.unique(clusters))),
            "reliable": False,
            "note": (f"covariance block is rank {rank} of {len(columns)}; the "
                     "tested coefficients are not jointly identified"),
        }
    try:
        statistic = float(beta @ np.linalg.solve(block, beta))
    except np.linalg.LinAlgError:                    # pragma: no cover
        return {"statistic": None, "p_value": None, "converged": fit.converged,
                "degrees_of_freedom": len(columns), "reliable": False}
    degrees = len(columns)
    return {
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": chi_square_sf(statistic, degrees),
        "rank": rank,
        "condition_number": conditioning,
        "converged": fit.converged,
        "separation_detected": fit.separated,
        "clusters": int(len(np.unique(clusters))),
        "reliable": bool(fit.converged and not fit.separated),
    }


def cluster_permutation_test(outcome: np.ndarray, clusters: np.ndarray,
                             labels: np.ndarray, statistic_fn,
                             draws: int = 5000, seed: int = 20260816) -> dict:
    """Permute a cluster-constant label across clusters, keeping blocks intact.

    Rows within a cluster move together, so the matched structure survives the
    permutation and the null being tested is "this label is exchangeable
    between clusters" rather than "rows are exchangeable".
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    rows_by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    cluster_label = {c: labels[rows_by_cluster[c][0]] for c in unique}
    if not all(len(set(labels[rows_by_cluster[c]].tolist())) == 1 for c in unique):
        return {"p_value": None, "reliable": False,
                "note": "label is not constant within clusters; "
                        "cluster permutation does not apply"}

    observed = statistic_fn(outcome, labels)
    values = np.array([cluster_label[c] for c in unique])
    extreme = 0
    for _ in range(draws):
        shuffled = rng.permutation(values)
        permuted = np.empty_like(labels)
        for cluster, value in zip(unique, shuffled):
            permuted[rows_by_cluster[cluster]] = value
        if statistic_fn(outcome, permuted) >= observed - 1e-12:
            extreme += 1
    return {
        "observed_statistic": float(observed),
        "draws": draws,
        "seed": seed,
        "p_value": float((extreme + 1) / (draws + 1)),
        "reliable": True,
    }
