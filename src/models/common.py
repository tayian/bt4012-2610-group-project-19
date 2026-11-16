# Shared helpers for the neural and anomaly models: data loading on the saved splits,
# label-free preprocessing fitted on the training fold only, and the writer for OOF scores and
# main_results rows.

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import read_splits
from src.eval.oof import oof_frame, write_frames

SEED = 4012
# v1 is the default for every model. The leakage comparison scores v1 against
# v1_with_inspection, which adds features built from the inspector-written columns
# (counter_statue, reading_remarque) once they exist.
FEATURES = {
    "v0": "data/processed/features_v0.parquet",
    "v1": "data/processed/features_v1.parquet",
    "v1_with_inspection": "data/processed/features_v1_with_inspection.parquet",
    "v2": "data/processed/features_v2.parquet",
    "v2_with_inspection": "data/processed/features_v2_with_inspection.parquet",
    "v3": "data/processed/features_v3.parquet",
    "v4": "data/processed/features_v4.parquet",
}
CAT_ONEHOT = ["client_catg"]
CAT_FREQ = ["district", "region"]


def load_train(feature_version="v1"):
    # Training clients in fold order: (ids, X DataFrame, y, folds).
    cli = pd.read_parquet("data/interim/client.parquet")[["client_id", "target"]]
    # read_splits checks the saved split files against the client table.
    _, _, folds = read_splits("data/splits", cli)
    folds["client_id"] = folds["client_id"].astype(str)
    cli["client_id"] = cli["client_id"].astype(str)
    X = pd.read_parquet(FEATURES[feature_version])
    X["client_id"] = X["client_id"].astype(str)
    df = folds.merge(cli, on="client_id").merge(X, on="client_id", how="left")
    ids = df["client_id"].values
    y = df["target"].values.astype(np.float32)
    f = df["fold"].values
    X = df.drop(columns=["client_id", "target", "fold"])
    return ids, X, y, f


class TabularPrep:
    # Fit on training fold only: one-hot client_catg, frequency-encode district/region, signed
    # log1p, median impute + missing flags, standardise.

    def fit(self, X: pd.DataFrame):
        self.num = [c for c in X.columns if c not in CAT_ONEHOT + CAT_FREQ]
        self.cats = {c: sorted(X[c].dropna().unique()) for c in CAT_ONEHOT if c in X}
        self.freq = {c: X[c].value_counts(normalize=True) for c in CAT_FREQ if c in X}
        Z = self._num(X)
        self.med = np.nanmedian(Z, 0)
        self.med = np.where(np.isnan(self.med), 0, self.med)
        self.miss_cols = np.where(np.isnan(Z).any(0))[0]
        Z = self._impute(Z)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        return self

    def _num(self, X):
        Z = X[self.num].to_numpy(np.float64)
        Z = np.sign(Z) * np.log1p(np.abs(Z))
        Z[~np.isfinite(Z)] = np.nan
        return Z

    def _impute(self, Z):
        flags = np.isnan(Z[:, self.miss_cols]).astype(np.float64)
        Z = np.where(np.isnan(Z), self.med, Z)
        return np.hstack([Z, flags])

    def transform(self, X):
        Z = (self._impute(self._num(X)) - self.mu) / self.sd
        parts = [Z]
        for c, levels in self.cats.items():
            parts.append(
                np.stack([(X[c].values == level) for level in levels], 1).astype(np.float64)
            )
        for c, fr in self.freq.items():
            parts.append(X[c].map(fr).fillna(0).to_numpy()[:, None] * 10)
        return np.clip(np.hstack(parts), -10, 10).astype(np.float32)


def save_run(
    ids, folds, y, scores, model, feature_version, run_id, oof_name=None, probabilities=True, **meta
):
    # Write one model's OOF scores and its row of results/tables/main_results.csv together.
    #
    # Metrics come from the shared src.eval.metrics.budget_metrics (tie-aware, with Brier and
    # net recovery), per saved fold, as mean and sample std, so the rows line up with M1 and
    # M3. A rerun replaces the row with the same run_id and keeps every other row. oof_name sets
    # the OOF file name when one model has variants (M4focal_v0 next to M4_v0).
    from src.eval.metrics import budget_metrics

    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    folds = np.asarray(folds)
    if not np.isfinite(scores).all():
        raise ValueError(f"{run_id}: every training client needs a finite OOF score.")
    table = pd.DataFrame([budget_metrics(y[folds == k], scores[folds == k]) for k in range(5)])
    if not probabilities:
        # Anomaly scores in [0, 1] are not fraud probabilities, so no Brier score.
        table["brier"] = np.nan
    row = {
        "model": model,
        "feature_version": feature_version,
        "n_clients": len(y),
        "n_folds": 5,
    }
    for metric in table.columns:
        row[f"{metric}_mean"] = float(table[metric].mean())
        row[f"{metric}_std"] = float(table[metric].std(ddof=1))
    row.update(
        {
            "run_id": run_id,
            "feature_set": f"features_{feature_version}",
            "evaluation_split": "train_cv",
            "seed": SEED,
            **meta,
        }
    )
    table_path = Path("results/tables/main_results.csv")
    previous = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    if "run_id" in previous:
        previous = previous.loc[previous["run_id"].ne(run_id)]
    results = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    oof = oof_frame(ids, folds, scores, model, feature_version)
    oof_path = Path(f"results/oof/{oof_name or model}_{feature_version}.parquet")
    write_frames({oof_path: oof, table_path: results})
    print(
        f"{run_id}: PR-AUC {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f}, "
        f"recall@5% {row['recall@5%_mean']:.4f}  -> {oof_path}",
        flush=True,
    )
    return row
