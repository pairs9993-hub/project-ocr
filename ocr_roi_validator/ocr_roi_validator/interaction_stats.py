"""Small-sample tests for whether a geometry effect differs by font.

Fifty events spread over eighteen cells is not much to work with, and the
temptation is to reach for a model whose degrees of freedom exceed what the
data can support. The tools here are deliberately modest: a logistic fit with
few parameters, a likelihood-ratio comparison against the same fit plus
interaction terms, and Fisher's exact test for pairwise work.

The important property is that they are allowed to answer "not enough
evidence". Two main effects existing separately does not make an interaction;
that is a distinct claim which has to be tested on its own, and with five empty
cells out of eighteen it may simply not be answerable here.

Implemented directly rather than pulled from statsmodels/scipy, which are not
in the product venv. Everything is plain Python and NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LogisticFit",
    "fisher_exact",
    "fit_logistic",
    "holm_adjust",
    "likelihood_ratio_test",
    "chi_square_sf",
    "detect_separation",
]


@dataclass(frozen=True)
class LogisticFit:
    coefficients: np.ndarray
    log_likelihood: float
    parameters: int
    converged: bool
    separated: bool


def fit_logistic(design: np.ndarray, outcome: np.ndarray,
                 max_iterations: int = 200, ridge: float = 1e-6) -> LogisticFit:
    """Newton-Raphson logistic regression with a whisper of ridge.

    The ridge term is there only to keep the Hessian invertible when a cell is
    empty; it is small enough not to move a well-identified estimate, and
    separation is reported separately rather than being hidden by it.
    """
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    samples, parameters = design.shape
    beta = np.zeros(parameters)
    converged = False
    for _ in range(max_iterations):
        eta = np.clip(design @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(mu * (1.0 - mu), 1e-10, None)
        gradient = design.T @ (outcome - mu) - ridge * beta
        hessian = (design.T * weights) @ design + ridge * np.eye(parameters)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:               # pragma: no cover
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    eta = np.clip(design @ beta, -30.0, 30.0)
    mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    log_likelihood = float(np.sum(outcome * np.log(mu)
                                  + (1 - outcome) * np.log(1 - mu)))
    return LogisticFit(beta, log_likelihood, parameters, converged,
                       bool(np.max(np.abs(beta)) > 15.0))


def chi_square_sf(statistic: float, degrees: int) -> float:
    """Upper tail of a chi-square distribution.

    Uses the regularised incomplete gamma function, computed by series or
    continued fraction depending on which converges.
    """
    if degrees <= 0:
        return 1.0
    if statistic <= 0:
        return 1.0
    a, x = degrees / 2.0, statistic / 2.0
    if x < a + 1.0:                                  # series for the lower tail
        term = 1.0 / a
        total = term
        for n in range(1, 1000):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - lower))
    # continued fraction for the upper tail
    tiny = 1e-300
    b, c = x + 1.0 - a, 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = h * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return max(0.0, min(1.0, upper))


def likelihood_ratio_test(restricted: LogisticFit, full: LogisticFit) -> dict:
    """Compare nested fits. The full model must nest the restricted one."""
    degrees = full.parameters - restricted.parameters
    statistic = 2.0 * (full.log_likelihood - restricted.log_likelihood)
    statistic = max(0.0, statistic)
    return {
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": chi_square_sf(statistic, degrees) if degrees > 0 else 1.0,
        "restricted_log_likelihood": restricted.log_likelihood,
        "full_log_likelihood": full.log_likelihood,
        "converged": restricted.converged and full.converged,
        "separation_detected": restricted.separated or full.separated,
    }


def fisher_exact(table: tuple[tuple[int, int], tuple[int, int]]) -> float:
    """Two-sided Fisher exact p-value for a 2x2 table."""
    (a, b), (c, d) = table
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or (b + d) == 0:
        return 1.0

    def probability(x: int) -> float:
        return math.exp(
            math.lgamma(row1 + 1) + math.lgamma(row2 + 1)
            + math.lgamma(col1 + 1) + math.lgamma(total - col1 + 1)
            - math.lgamma(total + 1) - math.lgamma(x + 1)
            - math.lgamma(row1 - x + 1) - math.lgamma(col1 - x + 1)
            - math.lgamma(total - col1 - row1 + x + 1))

    observed = probability(a)
    low = max(0, col1 - row2)
    high = min(col1, row1)
    total_probability = sum(probability(x) for x in range(low, high + 1)
                            if probability(x) <= observed * (1 + 1e-9))
    return max(0.0, min(1.0, total_probability))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction for a family of comparisons."""
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)      # enforce monotonicity
        adjusted[name] = running
    return adjusted


def detect_separation(counts: dict) -> dict:
    """Report empty and sparse cells rather than letting a fit paper over them."""
    empty = [str(k) for k, (n, h) in counts.items() if n > 0 and h == 0]
    no_exposure = [str(k) for k, (n, _) in counts.items() if n == 0]
    sparse = [str(k) for k, (n, h) in counts.items() if 0 < h < 5]
    return {
        "cells": len(counts),
        "cells_with_zero_events": empty,
        "cells_with_no_exposure": no_exposure,
        "cells_with_fewer_than_five_events": sparse,
        "quasi_separation_risk": bool(empty or no_exposure),
    }
