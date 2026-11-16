# F12 meter, tariff and recency features, and features_v3 = features_v1 + F12.
#
# v1 summarises consumption over a client's whole history. The time-robustness check showed
# that recent behaviour carries much of the signal, and v1 never uses the tariff, meter-code,
# coefficient or meter-index fields. F12 adds, per client:
#   recency    cons_pm mean over the last 1 and 2 years of the client's own history, their
#              ratio to the lifetime mean, the latest invoice's cons_pm, invoices in the last
#              2 years, the first and last invoice year and the years between them
#   tariff     number of tariff types and the share of invoices on the common ones (11, 40,
#              10, 15), number of meter codes, the most common meter code, and whether any
#              meter has a coefficient other than 1
#   meter      number of distinct meter numbers per counter type, the largest new_index, and
#              meter index growth per month (idx_diff / months_valid, mean and max)
#   bands      mean consumption per month in tariff levels 1 to 4 and the share of invoices
#              that reach level 3 or 4
# Only invoice fields are used (no labels, no inspector-written columns), so F12 needs no
# fold handling. "Last N years" is relative to each client's own last invoice, so clients
# whose billing stopped early are compared with themselves.
#
#   python3 -m src.features.meter_tariff
#     -> data/processed/meter_tariff.parquet, data/processed/features_v3.parquet
#   python3 -m src.features.meter_tariff --audit
#     -> results/tables/f12_ablation.csv      tuned M3 on folds 0 and 1: v1, v3, v1 plus each
#                                             F12 group and v3 minus each group
#        results/tables/f12_change_timing.csv when meter-code, meter-number and tariff changes
#                                             happen in a client's history, by class
# The timing table is the leakage check: meter replacements and tariff changes could follow a
# detected fraud, and labels have no date. If they did, fraud clients' changes would cluster
# at the end of their history; the table shows whether they do.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = [f"consommation_level_{i}" for i in range(1, 5)]
TARIFFS = [11, 40, 10, 15]
COLUMNS = [
    "cons_pm_mean_last1y",
    "cons_pm_mean_last2y",
    "ratio_last1y_lifetime",
    "ratio_last2y_lifetime",
    "cons_pm_last_invoice",
    "n_invoices_last2y",
    "first_invoice_year",
    "last_invoice_year",
    "active_years",
    "n_tarif_types",
    *[f"tarif_share_{t}" for t in TARIFFS],
    "n_counter_codes",
    "counter_code_mode",
    "has_coefficient_not_1",
    "n_counter_numbers_elec",
    "n_counter_numbers_gaz",
    "new_index_max",
    "index_growth_pm_mean",
    "index_growth_pm_max",
    *[f"level_{i}_pm_mean" for i in range(1, 5)],
    "share_level_3_plus",
]


def build(inv):
    inv = inv.sort_values(["client_id", "invoice_date"]).reset_index(drop=True)
    g = inv.groupby("client_id", sort=False)
    last = g["invoice_date"].transform("max")
    age = (last - inv["invoice_date"]).dt.days / 365.25
    out = pd.DataFrame(index=g.size().index)
    lifetime = g["cons_pm"].mean()
    for years in (1, 2):
        recent = inv.loc[age < years].groupby("client_id")["cons_pm"].mean()
        out[f"cons_pm_mean_last{years}y"] = recent
        out[f"ratio_last{years}y_lifetime"] = recent / lifetime.replace(0, np.nan)
    out["cons_pm_last_invoice"] = g["cons_pm"].last()
    out["n_invoices_last2y"] = (age < 2).groupby(inv["client_id"]).sum()
    out["first_invoice_year"] = g["invoice_date"].min().dt.year
    out["last_invoice_year"] = g["invoice_date"].max().dt.year
    out["active_years"] = (g["invoice_date"].max() - g["invoice_date"].min()).dt.days / 365.25
    out["n_tarif_types"] = g["tarif_type"].nunique()
    for t in TARIFFS:
        out[f"tarif_share_{t}"] = (inv["tarif_type"] == t).groupby(inv["client_id"]).mean()
    out["n_counter_codes"] = g["counter_code"].nunique()
    out["counter_code_mode"] = g["counter_code"].agg(lambda s: s.value_counts().index[0])
    out["has_coefficient_not_1"] = (inv["counter_coefficient"] != 1).groupby(inv["client_id"]).max()
    for kind in ("elec", "gaz"):
        rows = inv["counter_type"].str.lower() == kind
        out[f"n_counter_numbers_{kind}"] = (
            inv.loc[rows].groupby("client_id")["counter_number"].nunique()
        )
    out["new_index_max"] = g["new_index"].max()
    growth = inv["idx_diff"] / inv["months_valid"]
    growth = growth.where(np.isfinite(growth))
    out["index_growth_pm_mean"] = growth.groupby(inv["client_id"]).mean()
    out["index_growth_pm_max"] = growth.groupby(inv["client_id"]).max()
    for i, level in enumerate(LEVELS, 1):
        per_month = (inv[level] / inv["months_valid"]).where(lambda s: np.isfinite(s))
        out[f"level_{i}_pm_mean"] = per_month.groupby(inv["client_id"]).mean()
    high = (inv[LEVELS[2]] + inv[LEVELS[3]]) > 0
    out["share_level_3_plus"] = high.groupby(inv["client_id"]).mean()
    for kind in ("elec", "gaz"):
        out[f"n_counter_numbers_{kind}"] = out[f"n_counter_numbers_{kind}"].fillna(0)
    out["has_coefficient_not_1"] = out["has_coefficient_not_1"].astype(float)
    return out[COLUMNS].rename_axis("client_id").reset_index()


GROUPS = {
    "recency": COLUMNS[:9],
    "tariff": COLUMNS[9:17],
    "meter": COLUMNS[17:22],
    "bands": COLUMNS[22:],
}


def change_timing(inv, labels):
    inv = inv.sort_values(["client_id", "counter_type", "invoice_date"])
    by_meter = inv.groupby(["client_id", "counter_type"])
    first = inv.groupby("client_id")["invoice_date"].transform("min")
    last = inv.groupby("client_id")["invoice_date"].transform("max")
    rows = []
    for field in ("counter_code", "counter_number", "tarif_type"):
        previous = by_meter[field].shift()
        changed = previous.notna() & (previous != inv[field])
        events = pd.DataFrame(
            {
                "client_id": inv.loc[changed, "client_id"],
                "position": ((inv["invoice_date"] - first) / (last - first))[changed],
                "years_before_last": ((last - inv["invoice_date"]).dt.days / 365.25)[changed],
            }
        ).merge(labels, on="client_id")
        n_clients = labels["target"].value_counts()
        for target, e in events.groupby("target"):
            rows.append(
                {
                    "field": field,
                    "target": target,
                    "changes": len(e),
                    "changes_per_client": len(e) / n_clients[target],
                    "median_position_in_history": e["position"].median(),
                    "share_in_last_year": (e["years_before_last"] < 1).mean(),
                    "share_in_last_2_years": (e["years_before_last"] < 2).mean(),
                }
            )
    return pd.DataFrame(rows)


def audit(n_jobs=4):
    import json

    from sklearn.metrics import average_precision_score

    from src.eval.oof import write_frames
    from src.models.test_set import load_split
    from src.models.trees import fit_m3

    (ids, X, y), _ = load_split("v3")
    folds = pd.read_csv("data/splits/folds.csv", dtype={"client_id": "string"})
    fold = folds.set_index("client_id")["fold"].loc[ids].to_numpy()
    params = json.loads(Path("results/tables/M3_tuned_params_v1.json").read_text())
    v1 = [c for c in X.columns if c not in COLUMNS]

    def score(cols):
        values = []
        for k in (0, 1):
            m = fit_m3(X.loc[fold != k, cols], y[fold != k], **params, n_jobs=n_jobs)
            values.append(
                average_precision_score(y[fold == k], m.predict_proba(X.loc[fold == k, cols])[:, 1])
            )
        return float(np.mean(values))

    variants = {"v1": v1, "v3": list(X.columns)}
    for name, cols in GROUPS.items():
        variants[f"v1 + {name}"] = v1 + cols
        variants[f"v3 - {name}"] = [c for c in X.columns if c not in cols]
    ablation = pd.DataFrame(
        [
            {"features": k, "n_features": len(v), "pr_auc_folds_0_1": score(v)}
            for k, v in variants.items()
        ]
    )
    inv = pd.read_parquet(
        "data/interim/invoice_clean.parquet",
        columns=[
            "client_id",
            "invoice_date",
            "counter_type",
            "counter_number",
            "counter_code",
            "tarif_type",
        ],
    )
    inv["client_id"] = inv["client_id"].astype(str)
    train = set(pd.read_csv("data/splits/train_ids.csv", dtype=str)["client_id"])
    labels = pd.read_parquet("data/interim/client.parquet")[["client_id", "target"]]
    labels["client_id"] = labels["client_id"].astype(str)
    labels = labels[labels["client_id"].isin(train)]
    timing = change_timing(inv[inv["client_id"].isin(train)], labels)
    write_frames(
        {
            Path("results/tables/f12_ablation.csv"): ablation,
            Path("results/tables/f12_change_timing.csv"): timing,
        }
    )
    print(ablation.round(4).to_string(index=False))
    print(timing.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Build F12 and features_v3 (v1 + F12).")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--audit", action="store_true", help="write the F12 ablation and timing check")
    p.add_argument("--n-jobs", type=int, default=4)
    a = p.parse_args()
    if a.audit:
        audit(a.n_jobs)
        return
    inv = pd.read_parquet("data/interim/invoice_clean.parquet")
    inv["client_id"] = inv["client_id"].astype(str)
    f12 = build(inv)
    v1 = pd.read_parquet(a.processed / "features_v1.parquet")
    v1["client_id"] = v1["client_id"].astype(str)
    v3 = v1.merge(f12, on="client_id", how="left", validate="one_to_one")
    f12.to_parquet(a.processed / "meter_tariff.parquet", index=False)
    v3.to_parquet(a.processed / "features_v3.parquet", index=False)
    print(f"F12: {len(COLUMNS)} features for {len(f12)} clients; features_v3 has {v3.shape[1] - 1}")


if __name__ == "__main__":
    main()
