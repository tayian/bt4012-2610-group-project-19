# Hand-calculated checks for the budget metrics; no project data is used.

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.metrics import (
    bootstrap_ci,
    bootstrap_values,
    budget_metrics,
    budget_size,
    expected_tp_curve,
    net_recovery_curve,
)
from src.eval.oof import feature_version


class MetricTests(unittest.TestCase):
    def test_expected_tp_splits_tie_groups(self):
        # Scores 0.9 | 0.5 0.5 0.5 0.5 | 0.1, with 2 of the four ties positive.
        y = np.array([1, 1, 0, 1, 0, 0])
        s = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])
        np.testing.assert_allclose(expected_tp_curve(y, s), [0, 1, 1.5, 2, 2.5, 3, 3])

    def test_top_k_metrics_do_not_depend_on_row_order(self):
        rng = np.random.default_rng(4012)
        y = (rng.random(500) < 0.1).astype(int)
        s = rng.integers(0, 4, 500) / 4
        a = budget_metrics(y, s)
        for _ in range(5):
            order = rng.permutation(500)
            b = budget_metrics(y[order], s[order])
            for key in a:
                self.assertAlmostEqual(a[key], b[key], places=12, msg=key)

    def test_precision_and_recall_at_budget(self):
        y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 10)
        s = np.linspace(1, 0, 100)
        out = budget_metrics(y, s, budgets=(0.01, 0.05, 0.10))
        self.assertEqual(out["precision@1%"], 1.0)
        self.assertEqual(out["precision@5%"], 2 / 5)
        self.assertEqual(out["recall@10%"], 2 / 20)
        self.assertAlmostEqual(out["pr_auc"], average_precision_score(y, s))
        self.assertAlmostEqual(out["roc_auc"], roc_auc_score(y, s))

    def test_budget_size_is_floor_with_minimum_one(self):
        self.assertEqual(budget_size(99, 0.05), 4)
        self.assertEqual(budget_size(50, 0.01), 1)
        self.assertEqual(budget_size(100, 0.29), 29)

    def test_brier_only_for_probabilities(self):
        y = np.array([0, 1, 0, 1])
        out = budget_metrics(y, np.array([0.0, 1.0, 0.5, 0.5]))
        self.assertAlmostEqual(out["brier"], 0.125)
        self.assertTrue(np.isnan(budget_metrics(y, np.array([-1.0, 2.0, 0.0, 1.0]))["brier"]))

    def test_net_recovery_at_optimal_budget(self):
        # TP(k) = 1, 1, 2, 2, 2: Net = V * TP(k) - k.
        y = np.array([1, 0, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        k, net = net_recovery_curve(y, s, value_per_fraud=5)
        np.testing.assert_array_equal(k, [1, 2, 3, 4, 5])
        np.testing.assert_allclose(net, [4, 3, 7, 6, 5])
        out = budget_metrics(y, s)
        self.assertEqual(out["net_opt_vc5"], 7)
        self.assertEqual(out["k_opt_share_vc5"], 3 / 5)
        self.assertEqual(out["net_opt_vc20"], 37)

    def test_bootstrap_skips_single_class_resamples(self):
        y = np.array([1, 1, 1, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        for metric in ("roc_auc", roc_auc_score):
            low, high = bootstrap_ci(y, s, metric, n_boot=200)
            self.assertTrue(np.isfinite([low, high]).all())
            np.testing.assert_array_equal(
                bootstrap_ci(np.ones(3), s[:3], metric, n_boot=5), [np.nan, np.nan]
            )

    def test_fast_bootstrap_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(4012)
        y = (rng.random(400) < 0.1).astype(int)
        s = rng.integers(0, 6, 400) / 6
        for name, fn in (("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)):
            fast, slow = (
                bootstrap_values(y, s, name, n_boot=200),
                bootstrap_values(y, s, fn, n_boot=200),
            )
            np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                bootstrap_ci(y, s, name), bootstrap_ci(y, s, fn), rtol=0, atol=1e-12
            )

    def test_feature_version_from_filename(self):
        self.assertEqual(
            feature_version("data/features_v1_with_inspection.parquet"),
            ("v1", "v1_with_inspection"),
        )
        self.assertEqual(feature_version("features_v0.parquet"), ("v0", "v0"))
        with self.assertRaises(ValueError):
            feature_version("seq.parquet")


if __name__ == "__main__":
    unittest.main()
