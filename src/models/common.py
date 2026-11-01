"""Shared helpers for C's models: data loading on the plan's splits, label-free
preprocessing fitted on the training fold only, OOF writer in the 4.4 contract."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SEED = 4012
FEATURES = {
    "v0": "data/processed/features_v0.parquet",
    "v1": "data/processed/features_v1_noleak.parquet",   # chosen variant, see docs/decisions.md
    "v1_full": "data/processed/features_v1_full.parquet",
    "v1_noleak": "data/processed/features_v1_noleak.parquet",
}

def _chosen_v1():
    """W3-B1 leakage A/B decides which v1 variant is 'v1' for every model."""
    import json, os
    # >>> STAND-IN DEPENDENCY: this JSON is written by the stand-in M3 run.
    # Before merging: read B's W3-B1 decision (docs/decisions.md) instead, or
    # hard-code FEATURES["v1"] to the variant B chose.
    p = "results/tables/leakage_ab_standin.json"
    if os.path.exists(p):
        v = json.load(open(p))["headline_variant"]
        return f"data/processed/features_v1_{v}.parquet"
    return FEATURES["v1_noleak"]


FEATURES["v1"] = _chosen_v1()
CAT_ONEHOT = ["client_catg"]
CAT_FREQ = ["district", "region"]


def load_train(feature_version="v1"):
    """Training clients in fold order: (ids, X DataFrame, y, folds)."""
    folds = pd.read_csv("data/splits/folds.csv", dtype={"client_id": str})
    cli = pd.read_parquet("data/interim/client.parquet")[["client_id", "target"]]
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
    """Fit on training fold only: one-hot client_catg, frequency-encode
    district/region, signed log1p, median impute + missing flags, standardise."""

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
            parts.append(np.stack([(X[c].values == l) for l in levels], 1).astype(np.float64))
        for c, fr in self.freq.items():
            parts.append(X[c].map(fr).fillna(0).to_numpy()[:, None] * 10)
        return np.clip(np.hstack(parts), -10, 10).astype(np.float32)


def write_oof(ids, folds, scores, model, feature_version, path=None):
    out = pd.DataFrame({"client_id": ids, "fold": folds, "score": scores,
                        "model": model, "feature_version": feature_version})
    path = path or f"results/oof/{model}_{feature_version}.parquet"
    out.to_parquet(path, index=False)
    return out


def fold_report(y, s, folds):
    ap = [average_precision_score(y[folds == k], s[folds == k]) for k in np.unique(folds)]
    return float(np.mean(ap)), float(np.std(ap))


def budget_metrics(y_true, scores, budgets=(0.01, 0.05, 0.10)):
    """Same as B's src/eval/metrics.py (section 16.5).
    >>> STAND-IN: copy of B's code. Before merging replace the body with
        `from src.eval.metrics import budget_metrics` so there is one source of truth."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    y_sorted = y_true[np.argsort(-np.asarray(scores))]
    out = {"pr_auc": average_precision_score(y_true, scores), "roc_auc": roc_auc_score(y_true, scores)}
    for b in budgets:
        k = max(1, int(len(y_sorted) * b))
        tp = y_sorted[:k].sum()
        out[f"precision@{int(b * 100)}%"] = tp / k
        out[f"recall@{int(b * 100)}%"] = tp / y_true.sum()
    return out


def oof_metrics_row(y, s, folds, model, fv):
    """Per-fold mean of budget metrics (as in eval protocol 5.4)."""
    rows = [budget_metrics(y[folds == k], s[folds == k]) for k in np.unique(folds)]
    d = pd.DataFrame(rows)
    r = {"model": model, "feature_version": fv}
    for c in d.columns:
        r[c] = d[c].mean()
        r[c + "_std"] = d[c].std()
    return r
