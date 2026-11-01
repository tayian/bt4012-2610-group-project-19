# Shared helpers for the neural and anomaly models: data loading on the saved splits,
# label-free preprocessing fitted on the training fold only, and the OOF writer.

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.split import read_splits
from src.eval.oof import write_frames

SEED = 4012
# v1 is the default for every model. The leakage comparison scores v1 against
# v1_with_inspection, which adds features built from the inspector-written columns
# (counter_statue, reading_remarque) once they exist.
FEATURES = {
    "v0": "data/processed/features_v0.parquet",
    "v1": "data/processed/features_v1.parquet",
    "v1_with_inspection": "data/processed/features_v1_with_inspection.parquet",
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


def write_oof(ids, folds, scores, model, feature_version, path=None):
    # The columns of src.eval.oof.OOF_COLUMNS, written with its safe writer. The frame is built
    # here rather than with oof_frame, which would store client_id as a pandas string column
    # instead of the plain object column these files have always used.
    out = pd.DataFrame(
        {
            "client_id": ids,
            "fold": folds,
            "score": scores,
            "model": model,
            "feature_version": feature_version,
        }
    )
    path = path or f"results/oof/{model}_{feature_version}.parquet"
    write_frames({path: out})
    return out


def fold_report(y, s, folds):
    ap = [average_precision_score(y[folds == k], s[folds == k]) for k in np.unique(folds)]
    return float(np.mean(ap)), float(np.std(ap))


def budget_metrics(y_true, scores, budgets=(0.01, 0.05, 0.10)):
    # Precision and recall in the top 1%, 5% and 10% plus PR-AUC and ROC-AUC. Deliberately not
    # src.eval.metrics.budget_metrics: this one counts the positives among the first k clients
    # after a sort, while the shared one takes the expected count over scores tied at the cut-off
    # (and adds Brier and net-recovery columns), so the numbers can differ.
    y_true = np.asarray(y_true)
    y_sorted = y_true[np.argsort(-np.asarray(scores))]
    out = {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
    }
    for b in budgets:
        k = max(1, int(len(y_sorted) * b))
        tp = y_sorted[:k].sum()
        out[f"precision@{int(b * 100)}%"] = tp / k
        out[f"recall@{int(b * 100)}%"] = tp / y_true.sum()
    return out


def oof_metrics_row(y, s, folds, model, fv):
    # Per-fold mean of budget metrics (as in eval protocol 5.4).
    rows = [budget_metrics(y[folds == k], s[folds == k]) for k in np.unique(folds)]
    d = pd.DataFrame(rows)
    r = {"model": model, "feature_version": fv}
    for c in d.columns:
        r[c] = d[c].mean()
        r[c + "_std"] = d[c].std()
    return r
