"""Hand-calculated fixtures; no training or held-out model scoring."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "src/features/aggregate.py"
spec = importlib.util.spec_from_file_location("aggregate", SCRIPT)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.clients = pd.DataFrame({
            "client_id": ["0001", "0002", "0003"],
            "disrict": [60, 61, 62], "client_catg": [11, 12, 11],
            "region": [101, 102, 103], "creation_date": ["01/01/2019"] * 3,
            "target": [0, 1, 0],
        })
        rows = [
            ["0001", "2020-01-01", "meter1", "ELEC", 2, 120, 0, 0, 0, "6"],
            ["0001", "01/01/2021", "meter2", "GAZ", 1, 60, 0, 0, 0, "6"],
            ["0002", "2020-06-01", "meter3", "ELEC", 0, 240, 0, 0, 0, "6"],
        ]
        self.invoices = pd.DataFrame(rows, columns=[*aggregate.INVOICE_COLUMNS, "reading_remarque"])
        self.invoices = pd.concat([self.invoices, self.invoices.iloc[[0]]], ignore_index=True)

    def test_known_values_and_exact_dedup(self):
        out = aggregate.build_features(self.invoices, self.clients).set_index("client_id")
        self.assertEqual(list(out.columns), aggregate.FEATURE_COLUMNS)
        self.assertEqual(list(out.index), ["0001", "0002", "0003"])
        a = out.loc["0001"]
        self.assertEqual(a.n_invoices, 2)
        self.assertEqual(a.n_counters, 2)
        self.assertEqual((a.has_elec, a.has_gaz), (1, 1))
        self.assertEqual(a.cons_pm_mean, 60)
        self.assertEqual(a.cons_pm_std, 0)
        self.assertEqual(a.months_number_mean, 1.5)
        self.assertEqual(a.months_number_max, 2)
        self.assertAlmostEqual(a.tenure_years, 731 / 365.25)
        self.assertAlmostEqual(a.invoices_per_year, 2 / (366 / 365.25))

    def test_invalid_months_singleton_and_no_invoices(self):
        out = aggregate.build_features(self.invoices, self.clients).set_index("client_id")
        self.assertEqual(out.loc["0002", "n_invoices"], 1)
        self.assertTrue(pd.isna(out.loc["0002", "cons_pm_mean"]))
        self.assertTrue(pd.isna(out.loc["0002", "cons_pm_std"]))
        self.assertEqual(out.loc["0003", "n_invoices"], 0)
        self.assertEqual(out.loc["0003", "n_counters"], 0)
        self.assertTrue(pd.isna(out.loc["0003", "tenure_years"]))
        self.assertTrue(pd.isna(out.loc["0003", "invoices_per_year"]))
        numeric = out.select_dtypes(include="number")
        self.assertFalse(np.isinf(numeric.to_numpy()).any())

    def test_distinct_raw_fields_are_not_collapsed(self):
        extra = self.invoices.iloc[[0]].copy()
        extra["reading_remarque"] = "9"
        inv = pd.concat([self.invoices, extra], ignore_index=True)
        out = aggregate.build_features(inv, self.clients).set_index("client_id")
        self.assertEqual(out.loc["0001", "n_invoices"], 3)

    def test_labels_unused_and_inputs_unmodified(self):
        original = self.invoices.copy(deep=True)
        a = aggregate.build_features(self.invoices, self.clients)
        b = aggregate.build_features(self.invoices, self.clients.assign(target=[1, 0, 1]))
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(self.invoices, original)
        self.assertNotIn("target", a.columns)
        self.assertEqual(str(a["district"].dtype), "string")

    def test_bad_keys_and_mismatched_sources(self):
        with self.assertRaises(ValueError):
            aggregate.build_features(self.invoices, pd.concat([self.clients, self.clients.iloc[[0]]]))
        with self.assertRaises(ValueError):
            aggregate.build_features(self.invoices.assign(client_id="UNKNOWN"), self.clients)

    def test_csv_to_parquet_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clients = root / "client input.csv"
            invoices = root / "invoice input.csv"
            output = root / "processed/features_v0.parquet"
            self.clients.to_csv(clients, index=False)
            self.invoices.to_csv(invoices, index=False)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--clients", str(clients), "--invoices", str(invoices), "--out", str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = pd.read_parquet(output)
            self.assertEqual(saved.shape, (3, 16))
            self.assertEqual(saved.client_id.tolist(), ["0001", "0002", "0003"])
            self.assertIn("removed 1 exact duplicates", result.stdout)


if __name__ == "__main__":
    unittest.main()
