# Features v1: consumption level (F2) and billing regularity (F3).
#
# Reads the clean invoice table from src/data/clean.py, so impossible periods are already NaN
# in months_valid and cons_pm. F2 is computed for all counters together and separately for
# ELEC and GAZ (suffixes _elec, _gaz). F3 describes the client's billing as a whole. One row
# per client, NaN where a client has no usable invoices.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import LEVELS

DAYS_PER_MONTH = 30.44
MISSING_GAP_RATIO = 2.0  # a gap over twice the billed period means at least one missed reading
SCOPES = {"": None, "_elec": "ELEC", "_gaz": "GAZ"}
DAY_NS = 86_400_000_000_000


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


def build_features(inv, client_ids):
    # Group on integer client codes; the IDs are mapped back before the final reindex.
    if inv["client_id"].isna().any():
        raise ValueError("invoices with a missing client_id cannot be grouped")
    codes, ids = pd.factorize(inv["client_id"])
    key = pd.Series(codes, index=inv.index)
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
    parts.append(billing_features(inv, key))
    out = pd.concat(parts, axis=1)
    out.index = pd.Index(ids.take(out.index.to_numpy()), name="client_id")
    out = out.reindex(pd.Index(client_ids, name="client_id"))
    out["n_invoices"] = out["n_invoices"].fillna(0).astype("int64")
    out["has_missing_months"] = out["has_missing_months"].fillna(0).astype("int64")
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Build consumption (F2) and billing (F3) features."
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/features_v1.parquet"))
    args = parser.parse_args()
    try:
        inv = pd.read_parquet(
            args.interim / "invoice_clean.parquet",
            columns=[
                "client_id",
                "invoice_date",
                "counter_number",
                "counter_type",
                "months_valid",
                "cons",
                "cons_pm",
                *LEVELS,
            ],
        )
        clients = pd.read_parquet(args.interim / "client.parquet", columns=["client_id"])
        features = build_features(inv, clients["client_id"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.out, index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"behavioural: {exc}\n")
    print(f"Saved {len(features):,} clients x {features.shape[1] - 1} features to {args.out}")


if __name__ == "__main__":
    main()
