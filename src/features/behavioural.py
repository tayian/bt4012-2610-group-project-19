# Features v1: one row per client with every feature group that is known before an inspection.
#
#   F1 profile (from src/features/aggregate.py), F2 consumption level, F3 billing regularity,
#   F4 trajectory, F5 robust self-deviation, F6 data validity, F7 Benford and digits.
#
# features_v1_with_inspection adds F9 (meter status and reading remark shares), which come from
# the inspector-written columns counter_statue and reading_remarque, so it is only used for the
# leakage comparison. Everything reads the clean invoice table from src/data/clean.py, so
# impossible periods are already NaN in months_valid and cons_pm. NaN means a feature cannot be
# computed for the client (no usable invoices, too few readings, no variation).

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import REMARQUE_STATES, STATUE_STATES
from src.data.load import LEVELS
from src.features import aggregate

DAYS_PER_MONTH = 30.44
MISSING_GAP_RATIO = 2.0  # a gap over twice the billed period means at least one missed reading
SCOPES = {"": None, "_elec": "ELEC", "_gaz": "GAZ"}
DAY_NS = 86_400_000_000_000
PROFILE_COLUMNS = [
    "client_catg",
    "district",
    "region",
    "tenure_years",
    "has_elec",
    "has_gaz",
    "n_counters",
]
SPIKE_RATIO = 2.0  # F4: a bill above twice the client's median is a spike
SLOPE_YEARS = 3  # F4: the trend uses the last three years of each meter type
MAD_SCALE = 1.4826  # F5: makes the MAD comparable to a standard deviation for normal data
ROBUST_Z_LIMIT = 3.0
BENFORD_MIN_READINGS = 20  # F7: fewer positive readings give a meaningless digit distribution
BENFORD = np.log10(1 + 1 / np.arange(1, 10))
INVOICE_COLUMNS = [
    "client_id",
    "invoice_date",
    "counter_number",
    "counter_type",
    "counter_statue",
    "reading_remarque",
    "new_index",
    "months_valid",
    "cons",
    "cons_pm",
    "recon_gap",
    "flag_index_backwards",
    "flag_index_rollover",
    "flag_recon_mismatch",
    "flag_bad_months_number",
    "flag_partial_year",
    *LEVELS,
]
INSPECTION_COLUMNS = [
    *[f"statue_share_{s}" for s in [*STATUE_STATES, "other"]],
    "n_statue_changes",
    *[f"remarque_share_{s}" for s in [*REMARQUE_STATES, "other"]],
]


def _kurtosis(x, key):
    # Bias-corrected excess kurtosis per group, matching pandas Series.kurt().
    d = x - x.groupby(key).transform("mean")
    g = pd.DataFrame({"n": x.notna().astype(float), "m2": d**2, "m4": d**4}).groupby(key).sum()
    n = g["n"]
    m2, m4 = g["m2"] / n, g["m4"] / n
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = m4 / m2**2 - 3
        out = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    # A constant series has zero variance: pandas returns 0 rather than NaN.
    return out.where(n >= 4).where(m2 > 1e-14, np.where(n >= 4, 0.0, np.nan))


def consumption_features(x, key, upper, total, suffix=""):
    # F2: robust and classic statistics of cons_pm (x), plus the share above tariff level 1.
    g = x.groupby(key)
    median = g.transform("median")
    out = pd.DataFrame(
        {
            "cons_pm_mean": g.mean(),
            "cons_pm_median": g.median(),
            "cons_pm_std": g.std(),
            "cons_pm_min": g.min(),
            "cons_pm_max": g.max(),
            "cons_pm_mad": (x - median).abs().groupby(key).median(),
            "cons_pm_skew": g.skew(),
            "cons_pm_kurt": _kurtosis(x, key),
        }
    )
    upper, total = upper.groupby(key).sum(), total.groupby(key).sum()
    out["share_level_2_plus"] = (upper / total).where(total > 0)
    return out.add_suffix(suffix)


def _meter_order(inv, key):
    # Row order that sorts invoices by meter, then date, and a mask for "same meter as the row
    # before". Meters are (client, counter_number, counter_type).
    meter = (
        inv.groupby(["client_id", "counter_number", "counter_type"], sort=False, dropna=False)
        .ngroup()
        .to_numpy()
    )
    day = inv["invoice_date"].to_numpy().view("i8")
    order = np.lexsort((day, meter))
    meter = meter[order]
    same = np.r_[False, meter[1:] == meter[:-1]]
    return order, same, key.to_numpy()[order]


def billing_features(inv, key):
    # F3: how often and how regularly the client is billed.
    months = inv["months_valid"]
    dates = inv["invoice_date"].groupby(key)
    years = ((dates.max() - dates.min()).dt.days / 365.25).clip(lower=1.0)
    out = pd.DataFrame({"n_invoices": key.groupby(key).size()})
    out["invoices_per_year"] = out["n_invoices"] / years
    by_month = months.groupby(key)
    out["months_number_mean"] = by_month.mean()
    out["months_number_max"] = by_month.max()

    # Gaps are measured per meter: two meters billed on different days are not gaps.
    meter = (
        inv.groupby(["client_id", "counter_number", "counter_type"], sort=False, dropna=False)
        .ngroup()
        .to_numpy()
    )
    day = inv["invoice_date"].to_numpy().view("i8")
    first = np.flatnonzero(~pd.DataFrame({"meter": meter, "day": day}).duplicated().to_numpy())
    order = first[np.lexsort((day[first], meter[first]))]
    meter, day = meter[order], day[order]
    same = np.r_[False, meter[1:] == meter[:-1]]
    gap = np.where(same, np.r_[0, np.diff(day)] // DAY_NS, np.nan)
    billed = months.to_numpy(np.float64, na_value=np.nan)[order]
    missing = gap > MISSING_GAP_RATIO * billed * DAYS_PER_MONTH
    by_client = pd.DataFrame({"gap": gap, "missing": missing}).groupby(key.to_numpy()[order])
    out["days_since_last_invoice_mean"] = by_client["gap"].mean()
    out["days_since_last_invoice_max"] = by_client["gap"].max()
    out["has_missing_months"] = by_client["missing"].any().astype("int64")
    return out


def trajectory_features(inv, key, series):
    # F4 and F5. Each client's own history is judged per meter type (series = client x
    # counter_type), because ELEC and GAZ bills have different scales; the per-series values
    # are then combined per client (worst drop, steepest decline, largest deviation).
    x = inv["cons_pm"]
    ok = x.notna()
    client_of = key.groupby(series).first()

    # yoy_drop_max: largest fall between consecutive calendar years in mean cons_pm, as
    # 1 - this year / last year. The incomplete final year is left out.
    full = ok & ~inv["flag_partial_year"]
    year = inv["invoice_date"].dt.year
    annual = x[full].groupby([series[full], year[full]]).mean()
    s, y = annual.index.get_level_values(0), annual.index.get_level_values(1)
    v = annual.to_numpy()
    prev = np.r_[False, (s[1:] == s[:-1]) & (y[1:] - y[:-1] == 1)]
    prev_v = np.r_[np.nan, v[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = np.where(prev & (prev_v > 0), 1 - v / prev_v, np.nan)
    yoy = pd.Series(drop).groupby(client_of.reindex(s).to_numpy()).max()

    # slope_last3y: least-squares slope of cons_pm per year over the last three years of the
    # series; the steepest decline (minimum) across the client's meter types.
    t = (inv["invoice_date"] - inv["invoice_date"].min()).dt.days / 365.25
    recent = ok & (t >= t.groupby(series).transform("max") - SLOPE_YEARS)
    ts, xs, gs = t[recent], x[recent], series[recent]
    sums = pd.DataFrame({"n": 1.0, "t": ts, "x": xs, "tt": ts * ts, "tx": ts * xs}).groupby(gs)
    sums = sums.sum()
    denom = sums["n"] * sums["tt"] - sums["t"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (sums["n"] * sums["tx"] - sums["t"] * sums["x"]) / denom
    slope = slope.where((sums["n"] >= 3) & (denom > 1e-9))
    slope = slope.groupby(client_of.reindex(slope.index).to_numpy()).min()

    # sudden_spike_share and F5 robust z, both against the series' own median.
    median = x.groupby(series).transform("median")
    mad = (x - median).abs().groupby(series).transform("median")
    spike = (x > SPIKE_RATIO * median) & ok
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((x - median) / (MAD_SCALE * mad)).abs().where(mad > 0)
    per_client = pd.DataFrame(
        {"n_ok": ok, "spike": spike, "z": z, "z_gt": z > ROBUST_Z_LIMIT, "has_z": z.notna()}
    ).groupby(key)
    sums = per_client[["n_ok", "spike", "z_gt", "has_z"]].sum()
    out = pd.DataFrame(index=sums.index)
    out["yoy_drop_max"] = yoy
    out["slope_last3y"] = slope
    out["sudden_spike_share"] = (sums["spike"] / sums["n_ok"]).where(sums["n_ok"] > 0)
    out["robust_z_max"] = per_client["z"].max()
    out["n_robust_z_gt3"] = sums["z_gt"].where(sums["has_z"] > 0)

    # zero_run_max / zero_share: bills with zero consumption, runs counted per meter in date
    # order.
    zero = inv["cons"].eq(0).to_numpy()
    order, same, client = _meter_order(inv, key)
    z_sorted = zero[order]
    starts = ~same | np.r_[True, z_sorted[1:] != z_sorted[:-1]]
    run_id = np.cumsum(starts)
    run_len = pd.Series(z_sorted.astype(int)).groupby(run_id).transform("sum").to_numpy()
    run_len = np.where(z_sorted, run_len, 0)
    out["zero_run_max"] = pd.Series(run_len).groupby(client).max()
    out["zero_share"] = pd.Series(zero).groupby(key.to_numpy()).mean()
    return out


def validity_features(inv, key):
    # F6: physically impossible or inconsistent readings. Exact meter rollovers (a 5-digit
    # meter passing 99999) are not counted as backwards readings.
    flags = pd.DataFrame(
        {
            "n_index_backwards": inv["flag_index_backwards"] & ~inv["flag_index_rollover"],
            "n_recon_mismatch": inv["flag_recon_mismatch"],
            "n_bad_months_number": inv["flag_bad_months_number"],
        }
    ).astype("int64")
    out = flags.groupby(key).sum()
    out["recon_gap_mean"] = inv["recon_gap"].groupby(key).mean()
    return out


def digit_features(inv, key):
    # F7: first-digit Benford deviation of cons, and the share of round meter readings.
    cons = inv["cons"].to_numpy(np.float64)
    positive = cons > 0
    first = np.zeros(len(cons), dtype=np.int64)
    p = cons[positive]
    first[positive] = (p // 10 ** np.floor(np.log10(p))).astype(np.int64)
    counts = (
        pd.crosstab(key.to_numpy()[positive], first[positive])
        .reindex(columns=range(1, 10), fill_value=0)
        .astype(float)
    )
    n = counts.sum(axis=1)
    mad = (counts.div(n, axis=0) - BENFORD).abs().mean(axis=1)
    out = pd.DataFrame(index=pd.RangeIndex(key.max() + 1 if len(key) else 0))
    out["benford_mad"] = mad.where(n >= BENFORD_MIN_READINGS)
    out["round_reading_share"] = inv["new_index"].mod(5).eq(0).groupby(key).mean()
    return out


def inspection_features(inv, key):
    # F9: shares of each standardised meter status and reading remark, and how often the
    # status changes on the same meter.
    out = []
    for field, states, prefix in (
        ("counter_statue", STATUE_STATES, "statue_share_"),
        ("reading_remarque", REMARQUE_STATES, "remarque_share_"),
    ):
        shares = pd.crosstab(key.to_numpy(), inv[field].to_numpy(), normalize="index")
        out.append(shares.reindex(columns=[*states, "other"], fill_value=0.0).add_prefix(prefix))
    status = inv["counter_statue"].to_numpy()
    order, same, client = _meter_order(inv, key)
    s = status[order]
    changed = same & np.r_[False, s[1:] != s[:-1]]
    out.append(pd.Series(changed.astype("int64"), name="n_statue_changes").groupby(client).sum())
    return pd.concat(out, axis=1)[INSPECTION_COLUMNS]


def build_features(inv, clients, inspection=False):
    # One row per client in `clients` order. `inv` is invoice_clean.parquet and `clients` is
    # client.parquet (profile columns are taken from it, never the target).
    if inv["client_id"].isna().any():
        raise ValueError("invoices with a missing client_id cannot be grouped")
    missing = set(INVOICE_COLUMNS) - set(inv.columns)
    if missing:
        raise ValueError(f"Missing invoice columns: {sorted(missing)}")
    # Group on integer client codes; the IDs are mapped back before the final reindex.
    codes, ids = pd.factorize(inv["client_id"])
    key = pd.Series(codes, index=inv.index)
    series = pd.Series(
        pd.factorize(pd.MultiIndex.from_arrays([codes, inv["counter_type"].to_numpy()]))[0],
        index=inv.index,
    )
    upper = inv[LEVELS[1:]].sum(axis=1)
    parts = []
    for suffix, counter_type in SCOPES.items():
        rows = (
            slice(None)
            if counter_type is None
            else inv["counter_type"].eq(counter_type).to_numpy(bool, na_value=False)
        )
        parts.append(
            consumption_features(
                inv["cons_pm"][rows], key[rows], upper[rows], inv["cons"][rows], suffix
            )
        )
    parts += [
        billing_features(inv, key),
        trajectory_features(inv, key, series),
        validity_features(inv, key),
        digit_features(inv, key),
    ]
    if inspection:
        parts.append(inspection_features(inv, key))
    out = pd.concat(parts, axis=1).reindex(pd.RangeIndex(len(ids)))
    out.index = pd.Index(ids, name="client_id")
    client_ids = pd.Index(clients["client_id"], name="client_id")
    out = out.reindex(client_ids)
    # Clients without invoices have zero counts, not missing ones.
    counts = ["n_invoices", "has_missing_months", "n_index_backwards", "n_recon_mismatch"]
    counts += ["n_bad_months_number"] + (["n_statue_changes"] if inspection else [])
    out[counts] = out[counts].fillna(0).astype("int64")

    profile = aggregate.build_features(inv, clients).set_index("client_id")[PROFILE_COLUMNS]
    return pd.concat([profile.reindex(client_ids), out], axis=1).reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Build features_v1 (F1 to F7) and features_v1_with_inspection (+ F9)."
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/features_v1.parquet"))
    args = parser.parse_args()
    inspection_out = args.out.with_name(args.out.stem + "_with_inspection.parquet")
    try:
        inv = pd.read_parquet(args.interim / "invoice_clean.parquet", columns=INVOICE_COLUMNS)
        clients = pd.read_parquet(
            args.interim / "client.parquet", columns=aggregate.PROFILE_COLUMNS
        )
        features = build_features(inv, clients, inspection=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        features.drop(columns=INSPECTION_COLUMNS).to_parquet(args.out, index=False)
        features.to_parquet(inspection_out, index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"behavioural: {exc}\n")
    n = features.shape[1] - 1
    print(f"Saved {len(features):,} clients x {n - len(INSPECTION_COLUMNS)} features to {args.out}")
    print(f"Saved {len(features):,} clients x {n} features to {inspection_out}")


if __name__ == "__main__":
    main()
