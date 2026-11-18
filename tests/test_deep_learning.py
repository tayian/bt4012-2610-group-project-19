# Hand-made inputs for the sequence tensor, neural models, PU learning, attacks and results writer.

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.eval import adversarial
from src.eval.oof import OOF_COLUMNS
from src.features.sequence import build_sequence, last_invoice_month
from src.models import pu
from src.models.cnn import DualCNN, WideDeepCNN, to_calendar
from src.models.common import save_run


def invoices(rows):
    # rows: (client, date, cons, months_valid, counter_type)
    out = pd.DataFrame(rows, columns=["client_id", "invoice_date", "cons", "months_valid", "ct"])
    out["invoice_date"] = pd.to_datetime(out["invoice_date"])
    out["counter_type"] = out.pop("ct")
    out["consommation_level_1"] = out["cons"]
    for i in (2, 3, 4):
        out[f"consommation_level_{i}"] = 0
    out["cons_pm"] = out["cons"] / out["months_valid"]
    return out


class SequenceTests(unittest.TestCase):
    def test_spread_alignment_and_mask(self):
        inv = invoices(
            [
                ("a", "2019-03-15", 300, 3, "ELEC"),  # Jan-Mar 2019, 100 a month
                ("a", "2018-12-10", 40, 4, "GAZ"),  # Sep-Dec 2018, 10 a month
                ("a", "2017-01-01", 999, np.nan, "ELEC"),  # impossible period: left out
            ]
        )
        seq, ids = build_sequence(inv, ["a", "b"])
        self.assertEqual(seq.shape, (2, 120, 3))
        np.testing.assert_allclose(np.expm1(seq[0, -3:, 0]), [100, 100, 100], rtol=1e-5)
        np.testing.assert_allclose(np.expm1(seq[0, -7:-3, 1]), [10] * 4, rtol=1e-5)
        self.assertEqual(seq[0, -7:, 2].sum(), 0)  # covered months are unmasked
        self.assertEqual(seq[0, :-7, 2].min(), 1)  # everything older is masked
        self.assertTrue((seq[1, :, 2] == 1).all())  # client without invoices: all masked
        np.testing.assert_array_equal(last_invoice_month(inv, ["a", "b"]), [2, -1])

    def test_calendar_shift_ends_in_december(self):
        s = np.zeros((2, 120, 3), np.float32)
        s[:, :, 0] = np.arange(120)
        out = to_calendar(s, np.array([11, 8]))
        np.testing.assert_array_equal(out[0], s[0])  # already December
        self.assertEqual(out[1, -4, 0], 119)  # September moved 3 months earlier
        np.testing.assert_array_equal(out[1, -3:, 2], [1, 1, 1])


class NetworkTests(unittest.TestCase):
    def test_forward_shapes(self):
        seq, xs = torch.randn(4, 3, 120), torch.randn(4, 7)
        for net in (DualCNN(7), WideDeepCNN(7)):
            self.assertEqual(tuple(net(seq, xs).shape), (4,))


class PUTests(unittest.TestCase):
    def test_unweight_inverts_class_weight(self):
        p = np.array([0.01, 0.2, 0.7])
        w = 9.0
        weighted = w * p / (w * p + 1 - p)
        np.testing.assert_allclose(pu.unweight(weighted, w), p)

    def test_table_and_ranking(self):
        rng = np.random.default_rng(0)
        n = 500
        y = (rng.random(n) < 0.1).astype(int)
        oof = pd.DataFrame(
            {
                "client_id": [f"c{i}" for i in range(n)],
                "fold": np.arange(n) % 5,
                "score": np.clip(0.3 * y + rng.random(n) * 0.5, 0.01, 0.99),
            }
        )
        labels = pd.DataFrame({"client_id": oof["client_id"], "target": y})
        ids, folds, yy, scores, table = pu.pu_from_oof(oof, labels)
        self.assertEqual(list(table["fold"]), [0, 1, 2, 3, 4, "mean"])
        self.assertTrue((table["c_label_frequency"] > 0).all())
        # One constant per fold: the order inside each fold is M3's.
        for k in range(5):
            m = folds == k
            self.assertTrue(
                (
                    np.argsort(scores[m], kind="stable")
                    == np.argsort(oof["score"][m], kind="stable")
                ).all()
            )


def metered(inv, start=500):
    # One meter per client and counter type, readings chained so idx_diff = cons.
    inv = inv.sort_values(["client_id", "invoice_date"]).reset_index(drop=True)
    inv["counter_number"] = "1"
    inv["counter_coefficient"] = 1
    inv["new_index"] = start + inv.groupby(["client_id", "counter_type"])["cons"].cumsum()
    inv["old_index"] = inv["new_index"] - inv["cons"]
    inv["idx_diff"] = inv["cons"]
    return inv


class AttackTests(unittest.TestCase):
    def test_budget_and_cons_pm(self):
        inv = metered(
            invoices(
                [("a", f"201{i}-01-01", 100 + 50 * i, 4, "ELEC") for i in range(6)]
                + [("b", "2015-01-01", 80, np.nan, "ELEC")]
            )
        )
        for attack in (adversarial.attack_uniform, adversarial.attack_smooth):
            out = attack(inv)
            a = out["client_id"] == "a"
            drop = 1 - out.loc[a, "cons"].sum() / inv.loc[inv["client_id"] == "a", "cons"].sum()
            self.assertLessEqual(drop, adversarial.BUDGET + 1e-9)
            self.assertGreater(drop, adversarial.BUDGET - 0.02)
            np.testing.assert_allclose(out["cons_pm"][a], (out["cons"] / out["months_valid"])[a])
            self.assertTrue(out.loc[out["client_id"] == "b", "cons_pm"].isna().all())

    def test_meter_readings_follow_the_attack(self):
        # The meter reads lower too: idx_diff matches the new cons, each old_index equals the
        # previous new_index, and no reading goes below zero even from a low start.
        inv = metered(invoices([("a", f"201{i}-01-01", 400, 4, "ELEC") for i in range(5)]), 50)
        for attack in (adversarial.attack_uniform, adversarial.attack_smooth):
            out = attack(inv).sort_values("invoice_date")
            np.testing.assert_array_equal(out["idx_diff"], out["cons"])
            np.testing.assert_array_equal(out["new_index"] - out["old_index"], out["idx_diff"])
            np.testing.assert_array_equal(out["old_index"].iloc[1:], out["new_index"].iloc[:-1])
            self.assertGreaterEqual(out[["old_index", "new_index"]].min().min(), 0)

    def test_smooth_is_a_decline(self):
        f = adversarial._smooth_factor(np.full(10, 100.0))
        self.assertEqual(f[0], 1.0)
        self.assertTrue((np.diff(f) < 0).all())
        self.assertAlmostEqual((100 * f).sum(), 0.7 * 1000)

    def test_recall_and_swap(self):
        # Top 5% of 40 clients = 2 clients, holding 1 of the 20 frauds.
        y = np.array([1, 0] * 20)
        self.assertEqual(adversarial.recall_at(y, np.arange(40)[::-1], 0.05), 0.05)
        X = pd.DataFrame({"cat": pd.Series(["x", "y"], dtype="string"), "v": [1.0, 2.0]})
        F = pd.DataFrame({"cat": ["x"], "v": [9.0]}, index=pd.Index(["c2"], name="client_id"))
        out = adversarial.swap_frauds(X, np.array([1]), F, np.array(["c2"]))
        self.assertEqual(out.loc[1, "v"], 9.0)
        self.assertEqual(out["cat"].dtype, X["cat"].dtype)


class SaveRunTests(unittest.TestCase):
    def test_writes_oof_and_replaces_row(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                n = 100
                y = np.r_[np.ones(20), np.zeros(80)]
                folds = np.arange(n) % 5
                ids = np.array([f"c{i}" for i in range(n)])
                for score in (np.linspace(0, 1, n), np.linspace(1, 0, n)):
                    save_run(ids, folds, y, score, "M4", "v0", run_id="M4_test", loss="bce")
                table = pd.read_csv("results/tables/main_results.csv")
                oof = pd.read_parquet("results/oof/M4_v0.parquet")
            finally:
                os.chdir(cwd)
        self.assertEqual(len(table), 1)  # the rerun replaced its own row
        self.assertEqual(list(oof.columns), OOF_COLUMNS)
        self.assertIn("recall@5%_mean", table)


if __name__ == "__main__":
    unittest.main()
