# Small synthetic integration tests; their scores are not project results.

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.linear import model_fn

PROJECT = Path(__file__).resolve().parents[1]
RAW = "actual data (don't put on git)"


def write_fixture(root):
    # A copy of the code plus small raw CSVs with the full raw schema, so load and clean run.
    shutil.copytree(PROJECT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PROJECT / "run.py", root / "run.py")
    raw = root / RAW
    raw.mkdir()
    rows, invoice_rows = [], []
    for i in range(120):
        target = int(i % 5 == 0)
        cid = f"FIXTURE-{i:04d}"
        rows.append(
            {
                "disrict": 60 + i % 3,
                "client_id": cid,
                "client_catg": 11 + i % 2,
                "region": 101 + i % 4,
                "creation_date": "01/01/2010",
                "target": target,
            }
        )
        for period, date in enumerate(("2019-01-01", "2020-01-01")):
            cons = 100 + 200 * target + i % 7 + period * 10
            invoice_rows.append(
                {
                    "client_id": cid,
                    "invoice_date": date,
                    "tarif_type": 11,
                    "counter_number": str(i),
                    "counter_statue": "0",
                    "counter_code": 203,
                    "reading_remarque": "6",
                    "counter_coefficient": 1,
                    "consommation_level_1": cons,
                    "consommation_level_2": 0,
                    "consommation_level_3": 0,
                    "consommation_level_4": 0,
                    "old_index": 1000 * period,
                    "new_index": 1000 * period + cons,
                    "months_number": period + 1,
                    "counter_type": "ELEC",
                }
            )
    pd.DataFrame(rows).to_csv(raw / "client_train.csv", index=False)
    pd.DataFrame(invoice_rows + [invoice_rows[0]]).to_csv(raw / "invoice_train.csv", index=False)


def run_command(test, root, command, success=True):
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if success:
        test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    else:
        test.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
    return result


def run_step(test, root, target, success=True):
    # --skip-raw-check: the fixture CSVs are not the course files listed in data/checksums.txt.
    return run_command(
        test,
        root,
        [
            sys.executable,
            "run.py",
            target,
            "--skip-raw-check",
            "--clients",
            str(root / RAW / "client_train.csv"),
            "--invoices",
            str(root / RAW / "invoice_train.csv"),
        ],
        success,
    )


class BaselineTests(unittest.TestCase):
    # The chain is built once; tests that only need a finished baseline get a fresh copy of it.

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
        self.clients = self.root / RAW / "client_train.csv"
        self.invoices = self.root / RAW / "invoice_train.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def fresh(self):
        self.root.mkdir()
        write_fixture(self.root)

    def finished(self):
        shutil.copytree(self.built.name, self.root)

    def run_step(self, target, success=True):
        return run_step(self, self.root, target, success)

    def linear(self, success=True):
        return run_command(
            self,
            self.root,
            [sys.executable, "-m", "src.models.linear", "--clients", str(self.clients)],
            success,
        )

    def split_hashes(self):
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (self.root / "data/splits").glob("*.csv")
        }

    def test_fresh_clone_chain_and_reuse(self):
        self.fresh()
        self.run_step("baseline")
        train = pd.read_csv(self.root / "data/splits/train_ids.csv")
        test = pd.read_csv(self.root / "data/splits/test_ids.csv")
        oof = pd.read_parquet(self.root / "results/oof/M1_v0.parquet")
        table = pd.read_csv(self.root / "results/tables/main_results.csv")
        self.assertEqual(len(train), 96)
        self.assertEqual(len(test), 24)
        self.assertEqual(set(oof.client_id), set(train.client_id))
        self.assertFalse(set(oof.client_id) & set(test.client_id))
        self.assertTrue(oof.score.between(0, 1).all())
        self.assertEqual(
            list(oof.columns), ["client_id", "fold", "score", "model", "feature_version"]
        )
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["C"], 1)
        self.assertEqual(table.iloc[0]["penalty"], "l2")
        self.assertEqual(table.iloc[0]["evaluation_split"], "train_cv")
        self.assertIn("pr_auc_mean", table)
        self.assertIn("pr_auc_std", table)
        for metric in (
            "brier",
            "net_opt_vc5",
            "k_opt_share_vc5",
            "net_opt_vc10",
            "k_opt_share_vc10",
            "net_opt_vc20",
            "k_opt_share_vc20",
        ):
            self.assertIn(f"{metric}_mean", table)
            self.assertIn(f"{metric}_std", table)
        self.assertTrue(table.iloc[0]["brier_mean"] >= 0)
        self.assertTrue(0 < table.iloc[0]["k_opt_share_vc10_mean"] <= 1)
        self.assertFalse((self.root / "results/final_scores.parquet").exists())
        features = pd.read_parquet(self.root / "data/processed/features_v0.parquet")
        self.assertEqual(features.n_invoices.sum(), 240)  # the exact duplicate row is dropped
        frozen = self.split_hashes()
        self.run_step("splits")
        self.run_step("features")
        self.assertEqual(frozen, self.split_hashes())

    def test_results_upsert_preserves_other_models(self):
        self.finished()
        path = self.root / "results/tables/main_results.csv"
        old = pd.read_csv(path)
        other = old.copy().assign(run_id="OTHER-M3", model="M3")
        pd.concat([old, other], ignore_index=True).to_csv(path, index=False)
        self.linear()
        updated = pd.read_csv(path)
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated.model.eq("M1").sum(), 1)
        pd.testing.assert_frame_equal(
            updated.loc[updated.model.eq("M3")].reset_index(drop=True), other
        )

    def test_heldout_labels_and_values_do_not_affect_cv(self):
        self.finished()
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
        self.fresh()
        self.run_step("splits")
        (self.root / "data/splits/test_ids.csv").unlink()
        frozen = self.split_hashes()
        result = self.run_step("splits", success=False)
        self.assertIn("do not re-split", result.stderr)
        self.assertEqual(frozen, self.split_hashes())

    def test_missing_feature_client_rejected_before_output_changes(self):
        self.finished()
        output = self.root / "results/oof/M1_v0.parquet"
        before = output.read_bytes()
        path = self.root / "data/processed/features_v0.parquet"
        pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
        self.linear(success=False)
        self.assertEqual(before, output.read_bytes())

    def test_nullable_categories_empty_numeric_and_unknown_categories(self):
        frame = pd.DataFrame(
            {
                "client_catg": pd.Series(["11", pd.NA, "11", "12"], dtype="string"),
                "district": [60, 60, 61, 61],
                "region": ["101"] * 4,
                "cons_pm_mean": [2.0, np.nan, 8.0, 10.0],
                "cons_pm_std": [np.nan] * 4,
            }
        )
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
