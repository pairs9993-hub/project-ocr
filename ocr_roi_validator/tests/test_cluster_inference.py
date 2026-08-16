"""Tests for cluster-aware inference.

The property that matters is that clustering widens intervals rather than
narrowing them. A correction that made a clustered dataset look *more*
significant than an independent one would be worse than no correction, so that
comparison is tested directly against simulated data with a known answer.
"""

import unittest

import numpy as np

from ocr_roi_validator.cluster_inference import (
    cluster_bootstrap,
    cluster_permutation_test,
    cluster_robust_covariance,
    cluster_summary,
    cluster_wald_test,
)
from ocr_roi_validator.interaction_stats import fit_logistic


def clustered_data(clusters=200, per_cluster=6, effect=0.0,
                   cluster_noise=1.5, seed=7):
    """Rows grouped into clusters that share a random intercept."""
    rng = np.random.default_rng(seed)
    outcome, group, label = [], [], []
    for index in range(clusters):
        shared = rng.normal(0.0, cluster_noise)
        assignment = index % 2
        for _ in range(per_cluster):
            logit = -3.0 + shared + effect * assignment
            outcome.append(float(rng.random() < 1 / (1 + np.exp(-logit))))
            group.append(index)
            label.append(assignment)
    design = np.column_stack([np.ones(len(outcome)), np.array(label, dtype=float)])
    return design, np.array(outcome), np.array(group), np.array(label, dtype=float)


class SummaryTests(unittest.TestCase):
    def test_reports_structure(self) -> None:
        _, outcome, clusters, _ = clustered_data(clusters=50, per_cluster=4)
        summary = cluster_summary(clusters, outcome)
        self.assertEqual(summary["clusters"], 50)
        self.assertEqual(summary["rows"], 200)
        self.assertEqual(summary["rows_per_cluster_min"], 4)
        self.assertEqual(summary["rows_per_cluster_max"], 4)

    def test_uneven_clusters_are_described(self) -> None:
        clusters = np.array([0, 0, 0, 1, 2, 2])
        outcome = np.array([1.0, 0, 0, 0, 1.0, 1.0])
        summary = cluster_summary(clusters, outcome)
        self.assertEqual(summary["rows_per_cluster_min"], 1)
        self.assertEqual(summary["rows_per_cluster_max"], 3)
        self.assertEqual(summary["clusters_with_events"], 2)
        self.assertEqual(summary["max_events_in_one_cluster"], 2)


class RobustCovarianceTests(unittest.TestCase):
    def test_clustering_inflates_variance(self) -> None:
        """The whole point: correlated rows must not buy false precision."""
        design, outcome, clusters, _ = clustered_data(effect=0.0, cluster_noise=2.0)
        fit = fit_logistic(design, outcome)
        robust = cluster_robust_covariance(design, outcome, clusters,
                                           fit.coefficients)
        independent = np.arange(len(outcome))     # every row its own cluster
        naive = cluster_robust_covariance(design, outcome, independent,
                                          fit.coefficients)
        self.assertGreater(robust[1, 1], naive[1, 1])

    def test_covariance_is_symmetric_positive_diagonal(self) -> None:
        design, outcome, clusters, _ = clustered_data()
        fit = fit_logistic(design, outcome)
        covariance = cluster_robust_covariance(design, outcome, clusters,
                                               fit.coefficients)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
        self.assertTrue((np.diag(covariance) > 0).all())


class WaldTests(unittest.TestCase):
    def test_null_effect_is_not_significant(self) -> None:
        design, outcome, clusters, _ = clustered_data(effect=0.0, seed=3)
        result = cluster_wald_test(design, outcome, clusters, [1])
        self.assertGreater(result["p_value"], 0.05)

    def test_large_real_effect_is_detected(self) -> None:
        design, outcome, clusters, _ = clustered_data(
            clusters=400, effect=2.5, cluster_noise=0.5, seed=5)
        result = cluster_wald_test(design, outcome, clusters, [1])
        self.assertLess(result["p_value"], 0.01)

    def test_cluster_count_is_reported(self) -> None:
        design, outcome, clusters, _ = clustered_data(clusters=120)
        self.assertEqual(cluster_wald_test(design, outcome, clusters, [1])
                         ["clusters"], 120)


class BootstrapTests(unittest.TestCase):
    def test_interval_covers_zero_under_the_null(self) -> None:
        design, outcome, clusters, _ = clustered_data(effect=0.0, seed=9)
        result = cluster_bootstrap(design, outcome, clusters,
                                   np.array([0.0, 1.0]), draws=300)
        self.assertTrue(result["reliable"])
        self.assertFalse(result["excludes_zero"])

    def test_interval_excludes_zero_for_a_real_effect(self) -> None:
        design, outcome, clusters, _ = clustered_data(
            clusters=400, effect=2.5, cluster_noise=0.5, seed=13)
        result = cluster_bootstrap(design, outcome, clusters,
                                   np.array([0.0, 1.0]), draws=300)
        self.assertTrue(result["excludes_zero"])

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        design, outcome, clusters, _ = clustered_data(clusters=60)
        first = cluster_bootstrap(design, outcome, clusters,
                                  np.array([0.0, 1.0]), draws=100, seed=42)
        second = cluster_bootstrap(design, outcome, clusters,
                                   np.array([0.0, 1.0]), draws=100, seed=42)
        self.assertEqual(first["ci95"], second["ci95"])

    def test_degenerate_draws_are_reported_not_hidden(self) -> None:
        """Almost no events: the method must decline rather than invent one."""
        outcome = np.zeros(120)
        outcome[0] = 1.0
        clusters = np.repeat(np.arange(20), 6)
        design = np.column_stack([np.ones(120), np.repeat([0.0, 1.0], 60)])
        result = cluster_bootstrap(design, outcome, clusters,
                                   np.array([0.0, 1.0]), draws=100)
        self.assertGreater(result["degenerate_draws"], 0)


class PermutationTests(unittest.TestCase):
    @staticmethod
    def difference(outcome, labels):
        first = outcome[labels == 1].mean() if (labels == 1).any() else 0.0
        second = outcome[labels == 0].mean() if (labels == 0).any() else 0.0
        return abs(first - second)

    def test_null_label_is_not_significant(self) -> None:
        _, outcome, clusters, labels = clustered_data(effect=0.0, seed=17)
        result = cluster_permutation_test(outcome, clusters, labels,
                                          self.difference, draws=400)
        self.assertGreater(result["p_value"], 0.05)

    def test_rejects_a_label_that_varies_within_clusters(self) -> None:
        _, outcome, clusters, _ = clustered_data(clusters=30, per_cluster=4)
        varying = np.arange(len(outcome)) % 2.0
        result = cluster_permutation_test(outcome, clusters, varying,
                                          self.difference, draws=50)
        self.assertFalse(result["reliable"])
        self.assertIsNone(result["p_value"])

    def test_p_value_is_never_zero(self) -> None:
        _, outcome, clusters, labels = clustered_data(
            clusters=200, effect=3.0, cluster_noise=0.3, seed=21)
        result = cluster_permutation_test(outcome, clusters, labels,
                                          self.difference, draws=200)
        self.assertGreater(result["p_value"], 0.0)


if __name__ == "__main__":
    unittest.main()


class RankDeficiencyTests(unittest.TestCase):
    """A pinv will invert an under-determined block and invent significance."""

    def test_duplicate_column_is_refused(self) -> None:
        design, outcome, clusters, _ = clustered_data(clusters=100)
        duplicated = np.column_stack([design, design[:, 1]])   # collinear
        result = cluster_wald_test(duplicated, outcome, clusters, [1, 2])
        self.assertFalse(result["reliable"])
        self.assertIsNone(result["p_value"])
        self.assertLess(result["rank"], 2)

    def test_zero_event_columns_do_not_yield_a_tiny_p(self) -> None:
        rng = np.random.default_rng(3)
        rows = 600
        outcome = (rng.random(rows) < 0.02).astype(float)
        clusters = np.repeat(np.arange(100), 6)
        empty = np.zeros(rows)
        empty[outcome == 0] = (rng.random((outcome == 0).sum()) < 0.3)
        design = np.column_stack([np.ones(rows), rng.normal(size=rows), empty])
        result = cluster_wald_test(design, outcome, clusters, [1, 2])
        if result["p_value"] is not None:
            self.assertGreater(result["p_value"], 1e-20)

    def test_well_conditioned_block_still_reports(self) -> None:
        design, outcome, clusters, _ = clustered_data(clusters=300, effect=2.0,
                                                      cluster_noise=0.5, seed=8)
        result = cluster_wald_test(design, outcome, clusters, [1])
        self.assertTrue(result["reliable"])
        self.assertIsNotNone(result["p_value"])
