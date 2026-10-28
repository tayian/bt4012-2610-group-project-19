"""W1-B4: build 15 quick client features from exact-deduplicated raw invoices."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LEVELS = [f"consommation_level_{i}" for i in range(1, 5)]
INVOICE_COLUMNS = [
    "client_id", "invoice_date", "counter_number", "counter_type",
    "months_number", *LEVELS,
]
FEATURE_COLUMNS = [
    # F1: profile and meters (7)
    "client_catg", "district", "region", "tenure_years",
    "has_elec", "has_gaz", "n_counters",
    # F2: consumption per month (4)
    "cons_pm_mean", "cons_pm_std", "cons_pm_min", "cons_pm_max",
    # F3: billing (4)
    "n_invoices", "invoices_per_year", "months_number_mean", "months_number_max",
]


def _dates(values):
    """Handle STEG's ISO invoice dates and day/month/year creation dates."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    text = values.astype("string").str.strip()
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(dmy)


def _ids(frame, name, unique=False):
    ids = frame["client_id"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{name}: client_id must be nonempty and nonmissing.")
    if unique and ids.duplicated().any():
        raise ValueError(f"{name}: client_id must be unique.")
    return ids


def build_features(invoices, clients):
    """Return one row per client, with client_id plus 15 predictors.

    No labels, split assignments, global statistics, encoders or scalers are
    used. Exact duplicates are removed across ALL original invoice columns.
    Consumption channels are combined for this small v0 feature set.
    """
    missing = set(INVOICE_COLUMNS) - set(invoices.columns)
    if missing:
        raise ValueError(f"Missing invoice columns: {sorted(missing)}")

    district = "district" if "district" in clients.columns else "disrict"
    profile_columns = ["client_id", "client_catg", district, "region", "creation_date"]
    missing = set(profile_columns) - set(clients.columns)
    if missing:
        raise ValueError(f"Missing client columns: {sorted(missing)}")

    # Select profile fields explicitly: target is never included.
    profile = clients[profile_columns].rename(columns={district: "district"}).copy()
    profile["client_id"] = _ids(profile, "clients", unique=True)

    # Do this before selecting columns, so invoices differing in other raw
    # fields (for example reading_remarque) are not incorrectly collapsed.
    inv = invoices.drop_duplicates()[INVOICE_COLUMNS].copy()
    inv["client_id"] = _ids(inv, "invoices")
    unknown = ~inv["client_id"].isin(profile["client_id"])
    if unknown.any():
        raise ValueError(
            "Invoice client IDs are missing from the client table. "
            "Check that both files belong to the same dataset (train or test)."
        )

    for column in ("client_catg", "district", "region"):
        # Preserve these as categories for encoding INSIDE each CV fold later.
        profile[column] = profile[column].astype("string")
    profile["creation_date"] = _dates(profile["creation_date"])
    inv["invoice_date"] = _dates(inv["invoice_date"])

    levels = inv[LEVELS].apply(pd.to_numeric, errors="coerce")
    cons = levels.sum(axis=1, min_count=4)
    months = pd.to_numeric(inv["months_number"], errors="coerce")
    inv["months_number"] = months.where(np.isfinite(months) & months.gt(0))
    inv["cons_pm"] = (cons / inv["months_number"]).replace([np.inf, -np.inf], np.nan)

    inv["counter_number"] = inv["counter_number"].astype("string").replace("", pd.NA)
    kind = inv["counter_type"].astype("string").str.strip().str.upper()
    inv["_has_elec"] = kind.eq("ELEC").fillna(False)
    inv["_has_gaz"] = kind.eq("GAZ").fillna(False)

    stats = inv.groupby("client_id", sort=False).agg(
        has_elec=("_has_elec", "max"),
        has_gaz=("_has_gaz", "max"),
        n_counters=("counter_number", "nunique"),
        cons_pm_mean=("cons_pm", "mean"),
        cons_pm_std=("cons_pm", "std"),
        cons_pm_min=("cons_pm", "min"),
        cons_pm_max=("cons_pm", "max"),
        n_invoices=("client_id", "size"),
        months_number_mean=("months_number", "mean"),
        months_number_max=("months_number", "max"),
        _first_invoice=("invoice_date", "min"),
        _last_invoice=("invoice_date", "max"),
    ).reset_index()

    features = profile.merge(stats, on="client_id", how="left", validate="one_to_one", sort=False)
    for column in ("has_elec", "has_gaz"):
        features[column] = features[column].fillna(False).astype("int64")
    for column in ("n_counters", "n_invoices"):
        features[column] = features[column].fillna(0).astype("int64")

    tenure = (features["_last_invoice"] - features["creation_date"]).dt.days / 365.25
    features["tenure_years"] = tenure.where(tenure.ge(0))
    history_years = (
        (features["_last_invoice"] - features["_first_invoice"]).dt.days / 365.25
    ).clip(lower=1.0)
    features["invoices_per_year"] = features["n_invoices"] / history_years

    return features[["client_id", *FEATURE_COLUMNS]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invoices", type=Path, default=Path("data/raw/invoice_train.csv"))
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/features_v0.parquet"))
    args = parser.parse_args()
    try:
        if args.out.suffix.lower() != ".parquet":
            raise ValueError("--out must end in .parquet.")
        # Read all invoice columns to identify exact raw-row duplicates.
        invoices = pd.read_csv(
            args.invoices, low_memory=False,
            dtype={"client_id": "string", "counter_number": "string",
                   "counter_statue": "string", "reading_remarque": "string"},
        )
        clients = pd.read_csv(
            args.clients, dtype={"client_id": "string", "client_catg": "string",
                                 "disrict": "string", "district": "string", "region": "string"},
        )
        features = build_features(invoices, clients)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.out, index=False, engine="pyarrow")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"features_v0: {exc}\n")

    deduplicated = int(features["n_invoices"].sum())
    no_invoices = int(features["n_invoices"].eq(0).sum())
    print(f"Read {len(invoices):,} invoice rows; removed {len(invoices) - deduplicated:,} exact duplicates.")
    print(f"Saved {len(features):,} clients x 15 features (+ client_id) to {args.out}")
    if no_invoices:
        print(f"NOTE: {no_invoices:,} clients have no invoices in this input; "
              "check whether you used a mini/sample file.")


if __name__ == "__main__":
    main()
