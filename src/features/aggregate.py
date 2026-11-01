# Build 15 quick client features (features_v0) from the loaded clients and cleaned invoices.

import argparse
from pathlib import Path

import pandas as pd

from src.data.ids import check_ids

INVOICE_COLUMNS = [
    "client_id",
    "invoice_date",
    "counter_number",
    "counter_type",
    "months_valid",
    "cons_pm",
]
PROFILE_COLUMNS = ["client_id", "client_catg", "district", "region", "creation_date"]
FEATURE_COLUMNS = [
    # F1: profile and meters (7)
    "client_catg",
    "district",
    "region",
    "tenure_years",
    "has_elec",
    "has_gaz",
    "n_counters",
    # F2: consumption per month (4)
    "cons_pm_mean",
    "cons_pm_std",
    "cons_pm_min",
    "cons_pm_max",
    # F3: billing (4)
    "n_invoices",
    "invoices_per_year",
    "months_number_mean",
    "months_number_max",
]


def build_features(invoices, clients):
    # Return one row per client, with client_id plus 15 predictors.
    #
    # `invoices` is invoice_clean.parquet (exact duplicates dropped, months_valid and cons_pm
    # already NaN for impossible periods) and `clients` is client.parquet. No labels, split
    # assignments, global statistics, encoders or scalers are used. Consumption channels are
    # combined for this small v0 feature set.
    for name, frame, columns in (
        ("invoice", invoices, INVOICE_COLUMNS),
        ("client", clients, PROFILE_COLUMNS),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")

    # Select profile fields explicitly: target is never included.
    profile = clients[PROFILE_COLUMNS].copy()
    profile["client_id"] = check_ids(profile["client_id"], "clients", unique=True)
    inv = invoices[INVOICE_COLUMNS].copy()
    inv["client_id"] = check_ids(inv["client_id"], "invoices")
    unknown = ~inv["client_id"].isin(profile["client_id"])
    if unknown.any():
        raise ValueError(
            "Invoice client IDs are missing from the client table. "
            "Check that both files belong to the same dataset (train or test)."
        )

    for column in ("client_catg", "district", "region"):
        # Preserve these as categories for encoding INSIDE each CV fold later.
        profile[column] = profile[column].astype("string")
    # Nullable booleans, so clients without invoices become NA (then 0) after the merge.
    inv["_has_elec"] = inv["counter_type"].eq("ELEC").fillna(False).astype("boolean")
    inv["_has_gaz"] = inv["counter_type"].eq("GAZ").fillna(False).astype("boolean")

    stats = (
        inv.groupby("client_id", sort=False)
        .agg(
            has_elec=("_has_elec", "max"),
            has_gaz=("_has_gaz", "max"),
            n_counters=("counter_number", "nunique"),
            cons_pm_mean=("cons_pm", "mean"),
            cons_pm_std=("cons_pm", "std"),
            cons_pm_min=("cons_pm", "min"),
            cons_pm_max=("cons_pm", "max"),
            n_invoices=("client_id", "size"),
            months_number_mean=("months_valid", "mean"),
            months_number_max=("months_valid", "max"),
            _first_invoice=("invoice_date", "min"),
            _last_invoice=("invoice_date", "max"),
        )
        .reset_index()
    )

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
    parser = argparse.ArgumentParser(
        description="Build the 15 baseline client features (features_v0)."
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/features_v0.parquet"))
    args = parser.parse_args()
    try:
        if args.out.suffix.lower() != ".parquet":
            raise ValueError("--out must end in .parquet.")
        invoices = pd.read_parquet(args.interim / "invoice_clean.parquet", columns=INVOICE_COLUMNS)
        clients = pd.read_parquet(args.interim / "client.parquet", columns=PROFILE_COLUMNS)
        features = build_features(invoices, clients)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.out, index=False, engine="pyarrow")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"features_v0: {exc}\n")

    no_invoices = int(features["n_invoices"].eq(0).sum())
    print(
        f"Saved {len(features):,} clients x 15 features (+ client_id) from "
        f"{len(invoices):,} clean invoices to {args.out}"
    )
    if no_invoices:
        print(
            f"NOTE: {no_invoices:,} clients have no invoices in this input; "
            "check whether you used a mini/sample file."
        )


if __name__ == "__main__":
    main()
