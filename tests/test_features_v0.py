# Hand-calculated fixtures; no training or held-out model scoring.
# Invoices go through clean.clean_invoices first, as they do in the pipeline.

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import clean
from src.features import aggregate

ROOT = Path(__file__).resolve().parents[1]


def raw_invoice(client, date, counter, kind, months, cons, remarque="6"):
    return {
        "client_id": client,
        "invoice_date": pd.Timestamp(date),
        "tarif_type": 11,
        "counter_number": counter,
        "counter_statue": "0",
        "counter_code": 203,
        "reading_remarque": remarque,
        "counter_coefficient": 1,
        "consommation_level_1": cons,
        "consommation_level_2": 0,
        "consommation_level_3": 0,
        "consommation_level_4": 0,
        "old_index": 0,
        "new_index": cons,
        "months_number": months,
        "counter_type": kind,
    }


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.clients = pd.DataFrame(
            {
                "client_id": pd.array(["0001", "0002", "0003"], dtype="string"),
                "district": pd.array(["60", "61", "62"], dtype="string"),
                "client_catg": pd.array(["11", "12", "11"], dtype="string"),
                "region": pd.array(["101", "102", "103"], dtype="string"),
                "creation_date": pd.to_datetime(["2019-01-01"] * 3),
                "target": [0, 1, 0],
            }
        )
        self.rows = [
            raw_invoice("0001", "2020-01-01", "meter1", "ELEC", 2, 120),
            raw_invoice("0001", "2021-01-01", "meter2", "GAZ", 1, 60),
            raw_invoice("0002", "2020-06-01", "meter3", "ELEC", 0, 240),
        ]
        self.rows.append(dict(self.rows[0]))

    def build(self, rows=None, clients=None):
        invoices = clean.clean_invoices(pd.DataFrame(self.rows if rows is None else rows))
        return aggregate.build_features(invoices, self.clients if clients is None else clients)

    def test_known_values_and_exact_dedup(self):
        out = self.build().set_index("client_id")
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
        out = self.build().set_index("client_id")
        self.assertEqual(out.loc["0002", "n_invoices"], 1)
        self.assertTrue(pd.isna(out.loc["0002", "cons_pm_mean"]))
        self.assertTrue(pd.isna(out.loc["0002", "cons_pm_std"]))
        self.assertEqual(out.loc["0003", "n_invoices"], 0)
        self.assertEqual(out.loc["0003", "n_counters"], 0)
        self.assertTrue(pd.isna(out.loc["0003", "tenure_years"]))
        self.assertTrue(pd.isna(out.loc["0003", "invoices_per_year"]))
        numeric = out.select_dtypes(include="number")
        self.assertFalse(np.isinf(numeric.to_numpy()).any())

    def test_months_number_above_120_is_invalid(self):
        # 636,624 months appears in the raw data; 120 is the largest valid value.
        rows = self.rows + [
            raw_invoice("0001", "2020-03-01", "meter1", "ELEC", 636624, 999),
            raw_invoice("0003", "2020-03-01", "meter4", "ELEC", 120, 240),
        ]
        out = self.build(rows).set_index("client_id")
        a = out.loc["0001"]
        self.assertEqual(a.n_invoices, 3)
        self.assertEqual(a.cons_pm_mean, 60)
        self.assertEqual(a.cons_pm_max, 60)
        self.assertEqual(a.months_number_mean, 1.5)
        self.assertEqual(a.months_number_max, 2)
        self.assertEqual(out.loc["0003", "months_number_max"], 120)
        self.assertEqual(out.loc["0003", "cons_pm_mean"], 2)

    def test_distinct_raw_fields_are_not_collapsed(self):
        rows = self.rows + [dict(self.rows[0], reading_remarque="9")]
        out = self.build(rows).set_index("client_id")
        self.assertEqual(out.loc["0001", "n_invoices"], 3)

    def test_labels_unused_and_inputs_unmodified(self):
        invoices = clean.clean_invoices(pd.DataFrame(self.rows))
        original = invoices.copy(deep=True)
        a = aggregate.build_features(invoices, self.clients)
        b = aggregate.build_features(invoices, self.clients.assign(target=[1, 0, 1]))
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(invoices, original)
        self.assertNotIn("target", a.columns)
        self.assertEqual(str(a["district"].dtype), "string")

    def test_bad_keys_and_mismatched_sources(self):
        with self.assertRaises(ValueError):
            self.build(clients=pd.concat([self.clients, self.clients.iloc[[0]]]))
        with self.assertRaises(ValueError):
            self.build([dict(row, client_id="UNKNOWN") for row in self.rows])
        with self.assertRaisesRegex(ValueError, "months_valid"):
            aggregate.build_features(pd.DataFrame(self.rows), self.clients)

    def test_interim_to_parquet_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            interim = Path(directory) / "interim files"
            interim.mkdir()
            output = Path(directory) / "processed/features_v0.parquet"
            self.clients.to_parquet(interim / "client.parquet", index=False)
            clean.clean_invoices(pd.DataFrame(self.rows)).to_parquet(
                interim / "invoice_clean.parquet", index=False
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.features.aggregate",
                    "--interim",
                    str(interim),
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = pd.read_parquet(output)
            self.assertEqual(saved.shape, (3, 16))
            self.assertEqual(saved.client_id.tolist(), ["0001", "0002", "0003"])
            self.assertIn("from 3 clean invoices", result.stdout)


if __name__ == "__main__":
    unittest.main()
