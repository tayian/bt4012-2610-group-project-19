# Hand-made invoices for clean.py and the consumption/billing features.

import unittest

import numpy as np
import pandas as pd

from src.data import clean
from src.features import behavioural


def invoice(
    client,
    date,
    cons,
    months=4,
    old=1000,
    counter="m1",
    kind="ELEC",
    statue="0",
    remarque="6",
    coef=1,
    levels=None,
):
    levels = levels or [cons, 0, 0, 0]
    return {
        "client_id": client,
        "invoice_date": pd.Timestamp(date),
        "tarif_type": 11,
        "counter_number": counter,
        "counter_statue": statue,
        "counter_code": 203,
        "reading_remarque": remarque,
        "counter_coefficient": coef,
        **{f"consommation_level_{i + 1}": v for i, v in enumerate(levels)},
        "old_index": old,
        "new_index": old + cons,
        "months_number": months,
        "counter_type": kind,
    }


class CleanTests(unittest.TestCase):
    def setUp(self):
        rows = [
            invoice("A", "2018-01-01", 400),
            invoice("A", "2018-05-01", 400),  # exact copy below
            invoice("A", "2018-09-01", 200, months=0, statue="A"),  # impossible period, junk status
            invoice("A", "2019-01-01", 0, remarque="413"),
        ]
        rows.append(dict(rows[1]))
        rollover = invoice("B", "2018-01-01", 1518, old=98482)
        rollover["new_index"] = 0  # 98482 -> 99999 -> 0
        backwards = invoice("B", "2018-05-01", 50, old=500)
        backwards["new_index"] = 400
        rows += [rollover, backwards]
        self.raw = pd.DataFrame(rows)

    def test_dedup_codes_and_flags(self):
        report = []
        out = clean.clean_invoices(self.raw, report)
        self.assertEqual(len(out), len(self.raw) - 1)
        self.assertEqual(report[0]["count"], 1)
        self.assertEqual(list(out.counter_statue), ["0", "0", "other", "0", "0", "0"])
        self.assertEqual(out.reading_remarque.iloc[3], "other")
        self.assertTrue(np.isnan(out.cons_pm.iloc[2]))
        self.assertTrue(out.flag_bad_months_number.iloc[2])
        self.assertEqual(out.cons_pm.iloc[0], 100)
        b = out[out.client_id == "B"]
        self.assertEqual(list(b.flag_index_backwards), [True, True])
        self.assertEqual(list(b.flag_index_rollover), [True, False])
        self.assertEqual(list(b.flag_recon_mismatch), [False, True])
        self.assertEqual(b.recon_gap.iloc[1], 150)
        self.assertEqual(list(out.columns), clean.COLUMNS)

    def test_final_year_status(self):
        dates = pd.Series(
            pd.to_datetime(
                [f"2018-{m:02d}-01" for m in range(1, 13)] * 4
                + [f"2019-{m:02d}-01" for m in range(1, 9)] * 4
                + ["2019-09-01"]
            )
        )
        final, complete, month, ratio = clean.final_year_status(dates)
        self.assertEqual((final, complete), (2019, False))
        self.assertEqual(ratio, 0)


def clients(ids):
    return pd.DataFrame(
        {
            "client_id": ids,
            "client_catg": 11,
            "district": 60,
            "region": 101,
            "creation_date": pd.Timestamp("2010-01-01"),
        }
    )


class BehaviouralTests(unittest.TestCase):
    # Client C has six invoices: four ELEC on one meter, two GAZ on another. D has none.

    def setUp(self):
        rows = [
            invoice("C", "2016-01-01", 400),
            invoice("C", "2016-05-01", 800, levels=[600, 200, 0, 0]),
            invoice("C", "2016-09-01", 0, statue="1"),
            invoice("C", "2017-09-01", 1200, levels=[600, 300, 300, 0]),  # 12 months late
            invoice("C", "2016-01-01", 80, months=2, counter="g1", kind="GAZ"),
            invoice("C", "2016-03-01", 40, months=2, old=1003, counter="g1", kind="GAZ"),
        ]
        self.inv = clean.clean_invoices(pd.DataFrame(rows))
        # In this tiny table 2017 looks like an incomplete final year; treat both years as full
        # so the year-on-year feature can be checked (the partial case is tested separately).
        self.inv["flag_partial_year"] = False
        self.out = self.build(self.inv)

    def build(self, inv, ids=("C", "D")):
        out = behavioural.build_features(inv, clients(list(ids)), inspection=True)
        return out.set_index("client_id")

    def test_consumption_stats(self):
        c = self.out.loc["C"]
        x = pd.Series([100, 200, 0, 300, 40, 20], dtype=float)
        self.assertAlmostEqual(c.cons_pm_mean, x.mean())
        self.assertAlmostEqual(c.cons_pm_median, 70)
        self.assertAlmostEqual(c.cons_pm_mad, (x - 70).abs().median())
        self.assertAlmostEqual(c.cons_pm_skew, x.skew())
        self.assertAlmostEqual(c.cons_pm_kurt, x.kurt())
        self.assertAlmostEqual(c.cons_pm_mean_elec, 150)
        self.assertAlmostEqual(c.cons_pm_max_gaz, 40)
        self.assertAlmostEqual(c.share_level_2_plus, 800 / 2520)
        self.assertAlmostEqual(c.share_level_2_plus_gaz, 0)

    def test_billing_stats(self):
        c = self.out.loc["C"]
        self.assertEqual(c.n_invoices, 6)
        self.assertEqual(c.months_number_max, 4)
        self.assertAlmostEqual(c.days_since_last_invoice_max, 365)
        self.assertAlmostEqual(c.days_since_last_invoice_mean, np.mean([121, 123, 365, 60]))
        self.assertEqual(c.has_missing_months, 1)
        self.assertAlmostEqual(c.invoices_per_year, 6 / (609 / 365.25))

    def test_client_without_invoices(self):
        d = self.out.loc["D"]
        self.assertEqual((d.n_invoices, d.has_missing_months), (0, 0))
        self.assertTrue(np.isnan(d.cons_pm_mean))
        numeric = self.out.select_dtypes(include="number").to_numpy(dtype=float)
        self.assertFalse(np.isinf(numeric).any())

    def test_trajectory(self):
        c = self.out.loc["C"]
        # ELEC yearly means: 2016 = (100 + 200 + 0) / 3 = 100, 2017 = 300, so the "drop" is -2.
        self.assertAlmostEqual(c.yoy_drop_max, -2)
        # Only the ELEC series has three or more points for a slope.
        t = np.array([0, 121, 244, 609]) / 365.25
        self.assertAlmostEqual(c.slope_last3y, np.polyfit(t, [100, 200, 0, 300], 1)[0])
        self.assertEqual((c.zero_run_max, c.zero_share), (1, 1 / 6))
        self.assertEqual(c.sudden_spike_share, 0)

    def test_partial_final_year_is_left_out_of_year_on_year(self):
        inv = self.inv.copy()
        inv["flag_partial_year"] = inv["invoice_date"].dt.year.eq(2017)
        self.assertTrue(np.isnan(self.build(inv).loc["C"].yoy_drop_max))

    def test_robust_z(self):
        # ELEC: median 150, MAD 100; GAZ: median 30, MAD 10. Largest |z| is 150 / (1.4826 * 100).
        c = self.out.loc["C"]
        self.assertAlmostEqual(c.robust_z_max, 150 / (1.4826 * 100))
        self.assertEqual(c.n_robust_z_gt3, 0)

    def test_validity_counts_skip_rollovers(self):
        rows = [
            invoice("B", "2018-01-01", 1518, old=98482),
            invoice("B", "2018-05-01", 50, old=500),
        ]
        rows[0]["new_index"] = 0  # an exact meter rollover
        rows[1]["new_index"] = 400  # a real backwards reading, 150 short of cons
        b = self.build(clean.clean_invoices(pd.DataFrame(rows)), ["B"]).loc["B"]
        self.assertEqual(
            (b.n_index_backwards, b.n_recon_mismatch, b.n_bad_months_number), (1, 1, 0)
        )
        self.assertAlmostEqual(b.recon_gap_mean, 75)

    def test_digits(self):
        c = self.out.loc["C"]
        self.assertTrue(np.isnan(c.benford_mad))  # only 5 positive readings
        self.assertAlmostEqual(c.round_reading_share, 5 / 6)  # new_index 1043 is not round
        rows = [invoice("E", f"2015-{m:02d}-01", 100 + m) for m in range(1, 13)]
        rows += [invoice("E", f"2016-{m:02d}-01", 900 + m) for m in range(1, 13)]
        e = self.build(clean.clean_invoices(pd.DataFrame(rows)), ["E"]).loc["E"]
        shares = np.zeros(9)
        shares[[0, 8]] = 0.5
        expected = np.abs(shares - np.log10(1 + 1 / np.arange(1, 10))).mean()
        self.assertAlmostEqual(e.benford_mad, expected)

    def test_inspection_features_only_in_with_inspection(self):
        c = self.out.loc["C"]
        self.assertAlmostEqual(c.statue_share_0, 5 / 6)
        self.assertAlmostEqual(c.statue_share_1, 1 / 6)
        self.assertEqual(c.n_statue_changes, 2)  # ELEC meter: 0, 0, 1, 0
        self.assertEqual(c.remarque_share_6, 1)
        plain = behavioural.build_features(self.inv, clients(["C", "D"]))
        self.assertFalse(set(behavioural.INSPECTION_COLUMNS) & set(plain.columns))
        self.assertEqual(self.out.loc["D"].n_statue_changes, 0)

    def test_profile_columns(self):
        c = self.out.loc["C"]
        self.assertEqual((c.client_catg, c.has_elec, c.has_gaz, c.n_counters), ("11", 1, 1, 2))

    def test_kurtosis_matches_pandas_on_random_groups(self):
        rng = np.random.default_rng(4012)
        key = pd.Series(rng.integers(0, 30, 400))
        x = pd.Series(rng.lognormal(3, 1, 400))
        x[x.index % 50 == 0] = np.nan
        ours = behavioural._kurtosis(x, key)
        ref = x.groupby(key).apply(lambda s: s.kurt())
        np.testing.assert_allclose(ours.sort_index(), ref.sort_index(), rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
