# Synthetic experiment checks; no result in this file is a STEG performance estimate.

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_baseline import run_step, write_fixture

from src.eval.calibration import IsotonicClassifier
from src.eval.calibration import run as calibrate
from src.eval.explain import SHAP_COLUMNS, positive_shap
from src.eval.explain import run as explain
from src.models.experiments import context
from src.models.forest import model_fn as forest_model
from src.models.forest import search as forest_search
from src.models.imbalance import run as imbalance
from src.models.lr_search import search as lr_search
from src.models.trees import M3Classifier, fit_m3, run_lightgbm


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = tempfile.TemporaryDirectory()
        write_fixture(Path(cls.built.name))
        run_step(cls(), Path(cls.built.name), "baseline")

    @classmethod
    def tearDownClass(cls):
        cls.built.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project"
        shutil.copytree(self.built.name, self.root)
        self.clients = self.root / "actual data (don't put on git)/client_train.csv"
        self.features = self.root / "data/processed/features_v0.parquet"
        self.splits = self.root / "data/splits"
        self.results = self.root / "results"

    def tearDown(self):
        self.temp.cleanup()

    def ctx(self):
        return context(self.features, self.clients, self.splits, self.results)

    def test_lr_grid_exports_coefficients_resumes_and_keeps_baseline(self):
        baseline = (self.results / "oof/M1_v0.parquet").read_bytes()
        ctx = self.ctx()
        oof, selected = lr_search(ctx, max_iter=5000)
        grid = pd.read_csv(self.results / "tables/M1_grid_v0.csv")
        self.assertEqual(len(grid), 12)
        self.assertEqual(set(grid.penalty), {"l1", "l2", "elasticnet"})
        self.assertEqual(set(grid.C), {0.01, 0.1, 1, 10})
        coef = pd.read_csv(self.results / "coefficients/M1_grid_v0.csv")
        self.assertTrue(coef.feature.str.startswith(("numeric__", "categorical__")).all())
        self.assertEqual(set(coef.fit_scope), {"all_train_refit"})
        fold_coef = pd.read_csv(self.results / "coefficients/M1_grid_v0_folds.csv")
        self.assertEqual(fold_coef.candidate.nunique(), 12)
        self.assertEqual(set(fold_coef.fold), set(range(5)))
        self.assertEqual(set(oof.client_id), set(ctx.X.client_id))
        self.assertEqual(baseline, (self.results / "oof/M1_v0.parquet").read_bytes())
        with patch(
            "src.models.lr_search.model_fn",
            side_effect=AssertionError("refitted completed candidate"),
        ):
            resumed, again = lr_search(ctx, max_iter=10000)
        pd.testing.assert_frame_equal(oof, resumed)
        self.assertEqual(selected, again)
        table = pd.read_csv(self.results / "tables/main_results.csv")
        self.assertEqual(table.model.eq("M1_grid").sum(), 1)
        self.assertEqual(table.model.eq("M1").sum(), 1)

    def test_nonconverged_grid_does_not_publish_a_winner(self):
        with self.assertRaisesRegex(ValueError, "grid incomplete"):
            lr_search(self.ctx(), max_iter=1, Cs=(0.01,))
        self.assertFalse((self.results / "oof/M1_grid_v0.parquet").exists())
        table = pd.read_csv(self.results / "tables/main_results.csv")
        self.assertEqual(table.model.tolist(), ["M1"])

    def test_random_forest_grid_and_default_capacity(self):
        default = forest_model().named_steps["classifier"]
        self.assertEqual(default.n_estimators, 500)
        self.assertEqual(default.class_weight, "balanced_subsample")
        ctx = self.ctx()
        oof, _ = forest_search(ctx, n_jobs=1, n_estimators=12)
        grid = pd.read_csv(self.results / "tables/M2_grid_v0.csv")
        self.assertEqual(len(grid), 4)
        self.assertEqual(set(grid.min_samples_leaf), {5, 20})
        self.assertTrue(oof.score.between(0, 1).all())
        with patch("src.models.forest.model_fn", side_effect=AssertionError("refitted cached RF")):
            again, _ = forest_search(ctx, n_jobs=1, n_estimators=12)
        pd.testing.assert_frame_equal(oof, again)

    def test_fingerprint_ignores_holdout_but_detects_training_changes(self):
        before = self.ctx()
        heldout = pd.read_csv(self.splits / "test_ids.csv").client_id
        features = pd.read_parquet(self.features)
        features.loc[features.client_id.isin(heldout), "cons_pm_mean"] = 1e20
        features.to_parquet(self.features, index=False)
        clients = pd.read_csv(self.clients)
        mask = clients.client_id.isin(heldout)
        clients.loc[mask, "target"] = 1 - clients.loc[mask, "target"]
        clients.to_csv(self.clients, index=False)
        self.assertEqual(before.fingerprint, self.ctx().fingerprint)
        mask = features.client_id.eq(before.X.client_id.iloc[0])
        features.loc[mask, "cons_pm_mean"] += 1
        features.to_parquet(self.features, index=False)
        self.assertNotEqual(before.fingerprint, self.ctx().fingerprint)

    def test_sampling_counts_missing_values_and_categories(self):
        ctx = self.ctx()
        X, y = ctx.X.drop(columns="client_id"), ctx.y
        X = X.copy()
        X.loc[X.index[::4], "cons_pm_mean"] = np.nan
        X["empty"] = np.nan
        counts = {}
        for strategy in ("none", "scale_pos_weight", "undersample", "smote"):
            model = fit_m3(X, y, imbalance=strategy, n_jobs=1)
            counts[strategy] = model.sampling_counts_
            pred = model.predict_proba(X.iloc[:5])
            self.assertTrue(np.isfinite(pred).all())
            self.assertEqual(
                model.scale_pos_weight_,
                float(y.eq(0).sum() / y.eq(1).sum()) if strategy == "scale_pos_weight" else 1.0,
            )
        self.assertEqual(counts["undersample"]["fitted_negative"], 3 * int(y.eq(1).sum()))
        self.assertEqual(counts["smote"]["fitted_positive"], int(y.eq(0).sum()))
        self.assertEqual(counts["none"]["fitted_positive"], int(y.eq(1).sum()))

    def test_smote_preserves_original_rows_and_valid_category_levels(self):
        from src.models.sampling import balance_training

        ctx = self.ctx()
        fitted = fit_m3(ctx.X.drop(columns="client_id"), ctx.y, n_jobs=1)
        prepared = fitted._prepare(ctx.X.drop(columns="client_id"))
        prepared.loc[0, "cons_pm_mean"] = np.nan
        sampled, labels = balance_training(prepared, ctx.y, "smote", 4012)
        pd.testing.assert_frame_equal(
            prepared.reset_index(drop=True), sampled.iloc[: len(prepared)].reset_index(drop=True)
        )
        np.testing.assert_array_equal(labels[: len(prepared)], ctx.y)
        for c in fitted.categories_:
            self.assertEqual(list(sampled[c].cat.categories), list(prepared[c].cat.categories))

    def test_ablation_scale_arm_matches_existing_lightgbm_and_reuses_folds(self):
        ctx = self.ctx()
        frozen = {p.name: p.read_bytes() for p in self.splits.glob("*.csv")}
        baseline, _ = run_lightgbm(self.features, self.clients, self.splits, self.results, 1)
        table = imbalance(ctx, n_jobs=1)
        weighted = pd.read_parquet(self.results / "oof/M3_scale_pos_weight_v0.parquet")
        np.testing.assert_allclose(weighted.score, baseline.score, rtol=0, atol=0)
        self.assertEqual(len(table), 4)
        self.assertEqual(frozen, {p.name: p.read_bytes() for p in self.splits.glob("*.csv")})

    def test_calibration_inner_fits_never_see_outer_validation(self):
        ctx = self.ctx()
        X = ctx.X.drop(columns="client_id")
        train, valid = ctx.fold_ids != 0, ctx.fold_ids == 0
        records = []
        original = M3Classifier.fit

        def record(estimator, data, labels):
            records.append(set(data.index))
            return original(estimator, data, labels)

        with patch.object(M3Classifier, "fit", new=record):
            model = IsotonicClassifier("M3", n_jobs=1).fit(X.loc[train], ctx.y[train])
        self.assertEqual(len(records), 4)
        for indices in records:
            self.assertTrue(indices <= set(X.index[train]))
            self.assertFalse(indices & set(X.index[valid]))
        self.assertTrue(all(len(indices) < train.sum() for indices in records[:3]))
        self.assertEqual(records[-1], set(X.index[train]))
        self.assertTrue(np.isfinite(model.predict_proba(X.loc[valid])).all())

    def test_calibration_outputs_raw_pairs_plot_and_resumable_results(self):
        ctx = self.ctx()
        baseline, _ = run_lightgbm(self.features, self.clients, self.splits, self.results, 1)
        table, bins = calibrate(ctx, max_iter=5000, n_jobs=1)
        self.assertEqual(len(table), 4)
        self.assertEqual(set(bins.model), {"M1_raw", "M1_isotonic", "M3_raw", "M3_isotonic"})
        raw = pd.read_parquet(self.results / "oof/M3_calibration_raw_v0.parquet")
        np.testing.assert_allclose(raw.score, baseline.score)
        self.assertGreater((self.results / "figures/reliability_v0.png").stat().st_size, 1000)
        for kind in ("M1", "M3"):
            oof = pd.read_parquet(self.results / f"oof/{kind}_isotonic_v0.parquet")
            self.assertEqual(set(oof.client_id), set(ctx.X.client_id))
            self.assertTrue(oof.score.between(0, 1).all())
        with patch.object(
            IsotonicClassifier, "fit", side_effect=AssertionError("refitted completed calibration")
        ):
            again, _ = calibrate(ctx, max_iter=10000, n_jobs=1)
        pd.testing.assert_frame_equal(table, again, check_dtype=False)

    def test_shap_additivity_and_top_five_export(self):
        import shap

        ctx = self.ctx()
        X = ctx.X.drop(columns="client_id")
        model = fit_m3(X, ctx.y, n_jobs=1)
        prepared = model._prepare(X.iloc[:8])
        explainer = shap.TreeExplainer(
            model.estimator_, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        values = positive_shap(explainer, prepared)
        np.testing.assert_allclose(
            values.sum(axis=1) + explainer.expected_value,
            model.estimator_.predict(prepared, raw_score=True),
        )
        run_lightgbm(self.features, self.clients, self.splits, self.results, 1)
        top = explain(ctx, n_jobs=1, batch_size=11)
        self.assertEqual(list(top), SHAP_COLUMNS)
        self.assertEqual(len(top), 5 * len(ctx.X))
        self.assertTrue(top.groupby("client_id").size().eq(5).all())
        self.assertTrue(
            top.groupby("client_id")["rank"].apply(list).apply(lambda x: x == [1, 2, 3, 4, 5]).all()
        )
        for _, group in top.groupby("client_id", sort=False):
            self.assertTrue((np.diff(np.abs(group.shap_value)) <= 1e-12).all())
        self.assertEqual(top.value.dtype, np.dtype("float64"))
        with patch("src.eval.explain.fit_m3", side_effect=AssertionError("refitted SHAP")):
            resumed = explain(ctx, n_jobs=1)
        pd.testing.assert_frame_equal(top, resumed)

    def test_shap_rejects_a_different_baseline_without_overwriting(self):
        ctx = self.ctx()
        run_lightgbm(self.features, self.clients, self.splits, self.results, 1)
        explain(ctx, n_jobs=1)
        output = self.results / "shap_top5_v0.parquet"
        before = output.read_bytes()
        path = self.results / "oof/M3_v0.parquet"
        frame = pd.read_parquet(path)
        frame["score"] = 1 - frame["score"]
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "OOF scores differ"):
            explain(ctx, n_jobs=1)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
