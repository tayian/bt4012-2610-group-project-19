# Synthetic checks of saved-fold isolation, the shared tree interface and output contracts.

import hashlib
import importlib.util
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from test_baseline import run_step, write_fixture

from src.eval.oof import OOF_COLUMNS
from src.models.linear import load_training
from src.models.trees import PARAMS, M3Classifier, fit_m3, run_lightgbm
from src.models.trees_standin import PARAMS as COMPAT_PARAMS
from src.models.trees_standin import fit_m3 as compat_fit_m3


class TreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = tempfile.TemporaryDirectory()
        write_fixture(Path(cls.built.name))
        run_step(cls(), Path(cls.built.name), "baseline")

    @classmethod
    def tearDownClass(cls):
        cls.built.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project"
        shutil.copytree(self.built.name, self.root)
        self.clients = self.root / "actual data (don't put on git)/client_train.csv"
        self.features = self.root / "data/processed/features_v0.parquet"
        self.splits = self.root / "data/splits"
        self.results = self.root / "results"
        self.table = self.results / "tables/main_results.csv"
        self.oof = self.results / "oof/M3_v0.parquet"

    def tearDown(self):
        self.temporary.cleanup()

    def train(self):
        return run_lightgbm(self.features, self.clients, self.splits, self.results, n_jobs=1)

    def split_hashes(self):
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.splits.glob("*.csv")
        }

    def test_saved_folds_weights_categories_and_metrics(self):
        X, y, folds = load_training(self.features, self.clients, self.splits)
        assigned = X.client_id.map(folds.set_index("client_id").fold)
        features = pd.read_parquet(self.features)
        only_valid = set(X.loc[assigned.eq(0), "client_id"])
        features.loc[features.client_id.isin(only_valid), "region"] = "VALIDATION-ONLY"
        features.to_parquet(self.features, index=False)
        records = []

        class RecordingModel(M3Classifier):
            def fit(estimator, X, y):
                fitted = super().fit(X, y)
                records.append(
                    (set(X.index), fitted.scale_pos_weight_, fitted.categories_["region"])
                )
                return fitted

        before = self.split_hashes()
        with patch(
            "src.models.trees.model_fn", side_effect=lambda **params: RecordingModel(params=params)
        ):
            oof, row = self.train()

        self.assertEqual(self.split_hashes(), before)
        self.assertEqual(len(records), 5)
        for fold, (indices, weight, _) in enumerate(records):
            training = assigned.ne(fold)
            self.assertEqual(indices, set(X.index[training]))
            self.assertAlmostEqual(weight, y[training].eq(0).sum() / y[training].eq(1).sum())
        self.assertNotIn("VALIDATION-ONLY", records[0][2])
        self.assertIn("VALIDATION-ONLY", records[1][2])
        self.assertEqual(list(oof), OOF_COLUMNS)
        self.assertEqual(set(oof.client_id), set(X.client_id))
        self.assertTrue(oof.client_id.is_unique)
        self.assertTrue(oof.score.between(0, 1).all())
        self.assertEqual(set(oof.model), {"M3"})
        self.assertEqual(set(oof.feature_version), {"v0"})
        self.assertEqual(oof.fold.tolist(), assigned.tolist())
        ap = [
            average_precision_score(y[assigned.eq(fold)], oof.loc[assigned.eq(fold), "score"])
            for fold in range(5)
        ]
        self.assertAlmostEqual(row["pr_auc_mean"], np.mean(ap))
        self.assertAlmostEqual(row["pr_auc_std"], np.std(ap, ddof=1))
        self.assertIn("brier_mean", row)
        self.assertIn("net_opt_vc20_mean", row)
        self.assertFalse((self.results / "final_scores.parquet").exists())

    def test_interface_preserves_feature_order_and_handles_unseen_categories(self):
        X = pd.DataFrame(
            {
                "client_catg": pd.Categorical(
                    ["11", "12"] * 30, categories=["11", "12", "UNOBSERVED"]
                ),
                "district": [60, 61, 62] * 20,
                "cons": np.arange(60, dtype=float),
                "empty": [np.nan] * 60,
            }
        )
        y = np.array([0] * 45 + [1] * 15)
        model = fit_m3(X, y, n_jobs=1)
        self.assertIs(COMPAT_PARAMS, PARAMS)
        self.assertIs(compat_fit_m3, fit_m3)
        self.assertNotIn("scale_pos_weight", PARAMS)
        self.assertNotIn("monotone_constraints", PARAMS)
        self.assertEqual(model.scale_pos_weight_, 3)
        self.assertNotIn("UNOBSERVED", model.categories_["client_catg"])
        unknown = X.iloc[:3].copy()
        unknown["client_catg"] = ["NEW", None, "11"]
        unknown["district"] = 999
        self.assertTrue(model._prepare(unknown)["client_catg"].iloc[:2].isna().all())
        prediction = model.predict_proba(unknown)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(prediction.shape, (3, 2))
        np.testing.assert_allclose(prediction.sum(axis=1), 1)
        np.testing.assert_array_equal(
            prediction, model.predict_proba(unknown[list(reversed(unknown.columns))])
        )
        self.assertEqual(model._prepare(unknown).shape[1], X.shape[1])

    def test_monotone_interface_for_adversarial_defence(self):
        X = pd.DataFrame({"region": ["101"] * 120, "n_recon_mismatch": np.arange(120, dtype=float)})
        y = np.array([0] * 90 + [1] * 30)
        model = fit_m3(X, y, n_jobs=1, monotone_constraints=[0, 1])
        scores = model.predict_proba(X)[:, 1]
        self.assertTrue((np.diff(scores) >= -1e-12).all())
        self.assertEqual(model.estimator_.get_params()["monotone_constraints"], [0, 1])
        with self.assertRaisesRegex(ValueError, "computes scale_pos_weight"):
            fit_m3(X, y, scale_pos_weight=99)

    def test_rerun_replaces_only_its_row_and_preserves_lr(self):
        before = pd.read_csv(self.table)
        self.train()
        first_oof = pd.read_parquet(self.oof)
        self.train()
        after = pd.read_csv(self.table)
        self.assertEqual(len(after), len(before) + 1)
        self.assertEqual(after.model.eq("M3").sum(), 1)
        pd.testing.assert_frame_equal(
            before.reset_index(drop=True),
            after.loc[after.model.eq("M1"), before.columns].reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(first_oof, pd.read_parquet(self.oof))

    def test_adversarial_defence_uses_fitter_and_preserves_augmented_dtypes(self):
        # Only tree code runs here; isolate it from unrelated optional PyTorch/CNN imports.
        cnn = types.ModuleType("src.models.cnn")
        for name in ("SEQ_SCALE", "DualCNN", "load_seq", "to_model_seq"):
            setattr(cnn, name, None)
        path = Path(__file__).resolve().parents[1] / "src/eval/adversarial.py"
        spec = importlib.util.spec_from_file_location("tree_defence_fixture", path)
        study = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": types.ModuleType("torch"), "src.models.cnn": cnn}):
            spec.loader.exec_module(study)

        ids = np.array([f"TREE-{i}" for i in range(100)])
        X = pd.DataFrame(
            {
                "region": pd.Series(["101", "102"] * 50, dtype="string"),
                "n_recon_mismatch": np.arange(100, dtype=float),
            }
        )
        y = (np.arange(100) % 4 == 0).astype(int)
        folds = np.arange(100) % 5
        attacked = X.loc[y == 1].copy()
        attacked.index = ids[y == 1]
        records = []

        def record_fit(X, y, **params):
            self.assertTrue(pd.api.types.is_numeric_dtype(X["n_recon_mismatch"]))
            model = fit_m3(X, y, n_jobs=1, **params)
            records.append(model.scale_pos_weight_)
            return model

        with (
            patch.object(study, "load_train", return_value=(ids, X, y, folds)),
            patch.object(
                study, "attacked_features", return_value={"uniform": attacked, "smooth": attacked}
            ),
            patch("src.models.trees.fit_m3", side_effect=record_fit),
            patch.object(pd.DataFrame, "to_csv"),
        ):
            result = study.tree_defence()
        self.assertEqual(len(result), 15)
        np.testing.assert_allclose(records, [3.0] * 5 + [1.5] * 10)

    def test_heldout_changes_do_not_affect_training_predictions(self):
        self.train()
        before = pd.read_parquet(self.oof)
        frozen = self.split_hashes()
        heldout = pd.read_csv(self.splits / "test_ids.csv").client_id
        clients = pd.read_csv(self.clients)
        mask = clients.client_id.isin(heldout)
        clients.loc[mask, "target"] = 1 - clients.loc[mask, "target"]
        clients.to_csv(self.clients, index=False)
        features = pd.read_parquet(self.features)
        mask = features.client_id.isin(heldout)
        features.loc[mask, "cons_pm_mean"] = 1e12
        features.loc[mask, "region"] = "HELDOUT-ONLY"
        features.to_parquet(self.features, index=False)
        self.train()
        pd.testing.assert_frame_equal(before, pd.read_parquet(self.oof))
        self.assertEqual(frozen, self.split_hashes())

    def test_invalid_features_or_partial_split_leave_outputs_intact(self):
        self.train()
        before = (self.oof.read_bytes(), self.table.read_bytes())
        features = pd.read_parquet(self.features)
        features.iloc[1:].to_parquet(self.features, index=False)
        with self.assertRaisesRegex(ValueError, "cover the raw client IDs"):
            self.train()
        features.to_parquet(self.features, index=False)
        (self.splits / "test_ids.csv").unlink()
        frozen = self.split_hashes()
        with self.assertRaises(FileNotFoundError):
            self.train()
        self.assertEqual(frozen, self.split_hashes())
        self.assertEqual(before, (self.oof.read_bytes(), self.table.read_bytes()))

    def test_failed_fit_keeps_previous_outputs(self):
        self.train()
        before = (self.oof.read_bytes(), self.table.read_bytes())
        with patch.object(M3Classifier, "fit", side_effect=ValueError("synthetic fit failure")):
            with self.assertRaisesRegex(ValueError, "synthetic fit failure"):
                self.train()
        self.assertEqual(before, (self.oof.read_bytes(), self.table.read_bytes()))

    def test_runner_builds_features_and_oof_from_raw_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            run_step(self, root, "lightgbm")
            features = pd.read_parquet(root / "data/processed/features_v0.parquet")
            train = pd.read_csv(root / "data/splits/train_ids.csv")
            oof = pd.read_parquet(root / "results/oof/M3_v0.parquet")
            table = pd.read_csv(root / "results/tables/main_results.csv")
            self.assertEqual(len(features), 120)
            self.assertEqual(set(oof.client_id), set(train.client_id))
            self.assertEqual(list(oof), OOF_COLUMNS)
            self.assertEqual(table.model.tolist(), ["M3"])


if __name__ == "__main__":
    unittest.main()
