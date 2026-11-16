# Peer-residual features (F8): does a client declare less than similar customers?
#
# A LightGBM regressor predicts each invoice's log1p(cons_pm) from tarif_type, client_catg,
# region, invoice month and counter_type, trained on non-fraud clients only. The residual is
# actual minus predicted (on the log scale, so it reads as a ratio to the peer level), averaged
# and minimised per client. To keep labels out of a client's own features, the model for a
# training client in fold k is fitted on non-fraud training clients of the other folds; test
# clients use a model fitted on every non-fraud training client.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import N_FOLDS, SEED, read_splits

PREDICTORS = ["tarif_type", "client_catg", "region", "month", "counter_type"]
PARAMS = {
    "n_estimators": 200,
    "num_leaves": 63,
    "learning_rate": 0.1,
    "min_child_samples": 100,
    "random_state": SEED,
    "n_jobs": 4,
    "verbosity": -1,
    # Same result on every run (otherwise threads can change the last digits).
    "deterministic": True,
    "force_row_wise": True,
}
COLUMNS = ["peer_resid_mean", "peer_resid_min"]


def invoice_table(inv, clients):
    # One row per invoice with a valid cons_pm: categorical predictors and the log target.
    inv = inv[inv["cons_pm"].notna()]
    profile = clients.set_index("client_id")[["client_catg", "region"]]
    data = pd.DataFrame(
        {
            "client_id": inv["client_id"].to_numpy(),
            "tarif_type": inv["tarif_type"].to_numpy(),
            "client_catg": profile["client_catg"].reindex(inv["client_id"]).to_numpy(),
            "region": profile["region"].reindex(inv["client_id"]).to_numpy(),
            "month": inv["invoice_date"].dt.month.to_numpy(),
            "counter_type": inv["counter_type"].to_numpy(),
            "y": np.log1p(inv["cons_pm"].to_numpy(np.float64)),
        }
    )
    # Fixed category vocabularies from the codes themselves (no labels involved).
    for column in PREDICTORS:
        data[column] = data[column].astype("string").astype("category")
    return data


def residual_features(data, groups, params=None):
    # groups: list of (clients to score, non-fraud clients to fit on). Returns one row per
    # scored client.
    from lightgbm import LGBMRegressor

    out = []
    for score_ids, fit_ids in groups:
        fit = data[data["client_id"].isin(fit_ids)]
        model = LGBMRegressor(**{**PARAMS, **(params or {})}).fit(fit[PREDICTORS], fit["y"])
        rows = data[data["client_id"].isin(score_ids)]
        resid = rows["y"].to_numpy() - model.predict(rows[PREDICTORS])
        out.append(
            pd.DataFrame({"client_id": rows["client_id"].to_numpy(), "r": resid})
            .groupby("client_id")["r"]
            .agg(peer_resid_mean="mean", peer_resid_min="min")
        )
    return pd.concat(out)


def fold_groups(folds, labels, test_ids):
    # (clients scored, non-fraud clients fitted on) for each fold, then for the test set.
    normal = set(labels.loc[labels["target"].eq(0), "client_id"])
    groups = []
    for k in range(N_FOLDS):
        in_fold = folds["fold"].eq(k)
        fit = set(folds.loc[~in_fold, "client_id"]) & normal
        groups.append((set(folds.loc[in_fold, "client_id"]), fit))
    groups.append((set(test_ids), set(folds["client_id"]) & normal))
    return groups


def main():
    parser = argparse.ArgumentParser(description="Build the out-of-fold peer-residual features.")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/peer_residual.parquet"))
    args = parser.parse_args()
    try:
        clients = pd.read_parquet(args.interim / "client.parquet")
        clients["client_id"] = clients["client_id"].astype("string")
        _, test, folds = read_splits(args.split_dir, clients)
        inv = pd.read_parquet(
            args.interim / "invoice_clean.parquet",
            columns=["client_id", "invoice_date", "tarif_type", "counter_type", "cons_pm"],
        )
        inv["client_id"] = inv["client_id"].astype("string")
        data = invoice_table(inv, clients)
        features = residual_features(data, fold_groups(folds, clients, test["client_id"]))
        features = features.reindex(pd.Index(clients["client_id"], name="client_id"))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        features.reset_index().to_parquet(args.out, index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"peer: {exc}\n")
    print(f"Saved peer residuals for {features.notna().all(axis=1).sum():,} clients to {args.out}")


if __name__ == "__main__":
    main()
