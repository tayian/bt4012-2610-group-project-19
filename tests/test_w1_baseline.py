"""Small synthetic integration tests; their scores are not project results."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.models.linear import model_fn

PROJECT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shutil.copytree(PROJECT / "src", self.root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(PROJECT / "Makefile.w1", self.root / "Makefile.w1")
        (self.root / "Makefile").write_text("include Makefile.w1\n")
        raw = self.root / "actual data (don't put on git)"
        raw.mkdir()
        self.clients = raw / "client_train.csv"
        self.invoices = raw / "invoice_train.csv"
        rows, invoice_rows = [], []
        for i in range(120):
            target = int(i % 5 == 0)
            cid = f"FIXTURE-{i:04d}"
            rows.append({"client_id": cid, "target": target, "disrict": 60 + i % 3,
                         "client_catg": 11 + i % 2, "region": 101 + i % 4,
                         "creation_date": "01/01/2010"})
            for period, date in enumerate(("2019-01-01", "2020-01-01")):
                invoice_rows.append({
                    "client_id": cid, "invoice_date": date, "counter_number": str(i),
                    "counter_type": "ELEC", "months_number": period + 1,
                    "consommation_level_1": 100 + 200 * target + i % 7 + period * 10,
                    "consommation_level_2": 0, "consommation_level_3": 0,
                    "consommation_level_4": 0,
                })
        pd.DataFrame(rows).to_csv(self.clients, index=False)
        pd.DataFrame(invoice_rows + [invoice_rows[0]]).to_csv(self.invoices, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, target, success=True):
        command = ["make", target, f"PYTHON={sys.executable}",
                   f"CLIENTS={self.clients}", f"INVOICES={self.invoices}"]
        return self.run_command(command, success)

    def linear(self, success=True):
        return self.run_command([sys.executable, "-m", "src.models.linear", "--clients", str(self.clients)], success)

    def run_command(self, command, success=True):
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def split_hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (self.root / "data/splits").glob("*.csv")}

    def test_fresh_clone_chain_and_reuse(self):
        self.make("baseline")
        train = pd.read_csv(self.root / "data/splits/train_ids.csv")
        test = pd.read_csv(self.root / "data/splits/test_ids.csv")
        oof = pd.read_parquet(self.root / "results/oof/M1_v0.parquet")
        table = pd.read_csv(self.root / "results/tables/main_results.csv")
        self.assertEqual(len(train), 96)
        self.assertEqual(len(test), 24)
        self.assertEqual(set(oof.client_id), set(train.client_id))
        self.assertFalse(set(oof.client_id) & set(test.client_id))
        self.assertTrue(oof.score.between(0, 1).all())
        self.assertEqual(list(oof.columns), ["client_id", "fold", "score", "model", "feature_version"])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["C"], 1)
        self.assertEqual(table.iloc[0]["penalty"], "l2")
        self.assertEqual(table.iloc[0]["evaluation_split"], "train_cv")
        self.assertIn("pr_auc_mean", table)
        self.assertIn("pr_auc_std", table)
        self.assertFalse((self.root / "results/final_scores.parquet").exists())
        frozen = self.split_hashes()
        self.make("data")
        self.make("features")
        self.assertEqual(frozen, self.split_hashes())

    def test_results_upsert_preserves_other_models(self):
        self.make("baseline")
        path = self.root / "results/tables/main_results.csv"
        old = pd.read_csv(path)
        other = old.copy().assign(run_id="OTHER-M3", model="M3")
        pd.concat([old, other], ignore_index=True).to_csv(path, index=False)
        self.linear()
        updated = pd.read_csv(path)
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated.model.eq("M1").sum(), 1)
        pd.testing.assert_frame_equal(updated.loc[updated.model.eq("M3")].reset_index(drop=True), other)

    def test_heldout_labels_and_values_do_not_affect_cv(self):
        self.make("baseline")
        path = self.root / "results/oof/M1_v0.parquet"
        before = pd.read_parquet(path)
        test = pd.read_csv(self.root / "data/splits/test_ids.csv").client_id
        clients = pd.read_csv(self.clients)
        mask = clients.client_id.isin(test)
        clients.loc[mask, "target"] = 1 - clients.loc[mask, "target"]
        clients.to_csv(self.clients, index=False)
        feature_path = self.root / "data/processed/features_v0.parquet"
        features = pd.read_parquet(feature_path)
        mask = features.client_id.isin(test)
        features.loc[mask, "cons_pm_mean"] = 1e12
        features.loc[mask, "region"] = "UNSEEN-TEST-ONLY"
        features.to_parquet(feature_path, index=False)
        self.linear()
        pd.testing.assert_frame_equal(before, pd.read_parquet(path))

    def test_partial_splits_are_not_recreated(self):
        self.make("data")
        (self.root / "data/splits/test_ids.csv").unlink()
        frozen = self.split_hashes()
        result = self.make("data", success=False)
        self.assertIn("do not re-split", result.stderr)
        self.assertEqual(frozen, self.split_hashes())

    def test_missing_feature_client_rejected_before_output_changes(self):
        self.make("baseline")
        output = self.root / "results/oof/M1_v0.parquet"
        before = output.read_bytes()
        path = self.root / "data/processed/features_v0.parquet"
        pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
        self.linear(success=False)
        self.assertEqual(before, output.read_bytes())

    def test_nullable_categories_empty_numeric_and_unknown_categories(self):
        frame = pd.DataFrame({
            "client_catg": pd.Series(["11", pd.NA, "11", "12"], dtype="string"),
            "district": [60, 60, 61, 61], "region": ["101"] * 4,
            "cons_pm_mean": [2.0, np.nan, 8.0, 10.0],
            "cons_pm_std": [np.nan] * 4,
        })
        first, second = model_fn(), model_fn()
        self.assertIsNot(first, second)
        preprocessing = first.named_steps["preprocess"]
        preprocessing.fit(frame)
        numeric = preprocessing.named_transformers_["numeric"]
        np.testing.assert_array_equal(numeric.named_steps["impute"].statistics_, [8, 0])
        encoder = preprocessing.named_transformers_["categorical"].named_steps["encode"]
        self.assertNotIn("999", encoder.categories_[2])
        unknown = frame.copy()
        unknown["region"] = "999"
        transformed = preprocessing.transform(unknown)
        values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
