import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def budget_metrics(y_true, scores, budgets=(0.01, 0.05, 0.10)):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_sorted = y_true[np.argsort(-scores)]
    out = {"pr_auc": average_precision_score(y_true, scores),
           "roc_auc": roc_auc_score(y_true, scores)}
    for b in budgets:
        k = max(1, int(len(y_sorted) * b))
        tp = y_sorted[:k].sum()
        out[f"precision@{int(b*100)}%"] = tp / k
        out[f"recall@{int(b*100)}%"] = tp / y_true.sum()
    return out

def net_recovery_curve(y_true, scores, value_per_fraud=10.0, cost_per_visit=1.0):
    y_sorted = np.asarray(y_true)[np.argsort(-np.asarray(scores))]
    k = np.arange(1, len(y_sorted) + 1)
    net = value_per_fraud * np.cumsum(y_sorted) - cost_per_visit * k
    return k, net  # optimal budget: k[net.argmax()]

def bootstrap_ci(y_true, scores, metric_fn, n_boot=1000, seed=4012):
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        vals.append(metric_fn(y_true[idx], scores[idx]))
    return np.percentile(vals, [2.5, 97.5])


def cv_evaluate(
    model_fn,
    X,
    y,
    folds,
    *,
    model="model",
    feature_version="v0",
):
    """
    Evaluate a classifier using the five saved training folds.

    Parameters
    ----------
    model_fn : callable
        Returns a NEW unfitted classifier or Pipeline on each call.
        It must support fit(), predict_proba(), and classes_.

    X : pandas.DataFrame
        Training clients only.
        Contains client_id and predictor columns, but no target column.

    y : array-like
        Binary labels in exactly the same row order as X.
        If a pandas Series, its index must also match X.index.

    folds : pandas.DataFrame
        Saved folds.csv containing client_id and fold (0 through 4).
        Must cover exactly the clients in X.

    Returns
    -------
    oof : pandas.DataFrame
        client_id, fold, score, model, feature_version.

    metrics_row : dict
        Model metadata and each metric's mean and sample standard deviation.
    """
    # Validate inputs before fitting any models.
    if not isinstance(X, pd.DataFrame) or "client_id" not in X:
        raise ValueError("X must be a DataFrame with a client_id column.")

    if "target" in X:
        raise ValueError("Remove target from X and pass it separately as y.")

    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ValueError("y must have the same index and row order as X.")

    y = np.asarray(y)

    if y.ndim != 1 or len(y) != len(X) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain one binary 0/1 label per X row.")

    ids = X["client_id"].astype("string")

    f = folds[["client_id", "fold"]].copy()
    f["client_id"] = f["client_id"].astype("string")

    for values in (ids, f["client_id"]):
        if (
            values.isna().any()
            or values.str.strip().eq("").any()
            or values.duplicated().any()
        ):
            raise ValueError("Client IDs must be nonempty, nonmissing and unique.")

    if set(ids) != set(f["client_id"]):
        raise ValueError(
            "X must contain exactly the IDs in folds.csv; exclude test clients."
        )

    if set(f["fold"]) != set(range(5)):
        raise ValueError(
            "folds must contain all five saved fold numbers, 0 through 4."
        )

    # Align folds by client ID, regardless of CSV row order.
    fold_ids = ids.map(
        f.set_index("client_id")["fold"]
    ).to_numpy(dtype=int)

    # IDs identify clients; they are not model predictors.
    features = X.drop(columns="client_id")

    for fold in range(5):
        valid = fold_ids == fold

        if (
            len(np.unique(y[valid])) != 2
            or len(np.unique(y[~valid])) != 2
        ):
            raise ValueError(
                "Every training and validation partition must contain both classes."
            )

    scores = np.full(len(X), np.nan)
    fold_metrics = []

    for fold in range(5):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)

        # A fresh model/Pipeline for this fold.
        estimator = model_fn()

        estimator.fit(
            features.iloc[train_idx],
            y[train_idx],
        )

        positive = np.flatnonzero(
            np.asarray(estimator.classes_) == 1
        )

        if len(positive) != 1:
            raise ValueError("The estimator must expose positive class 1.")

        predicted = np.asarray(
            estimator.predict_proba(features.iloc[valid_idx])
        )[:, positive[0]]

        if (
            not np.isfinite(predicted).all()
            or ((predicted < 0) | (predicted > 1)).any()
        ):
            raise ValueError(
                "predict_proba must return finite probabilities in [0, 1]."
            )

        scores[valid_idx] = predicted

        fold_metrics.append(
            budget_metrics(y[valid_idx], predicted)
        )

    oof = pd.DataFrame({
        "client_id": ids.to_numpy(),
        "fold": fold_ids,
        "score": scores,
        "model": model,
        "feature_version": feature_version,
    })

    table = pd.DataFrame(fold_metrics)

    metrics_row = {
        "model": model,
        "feature_version": feature_version,
        "n_clients": len(X),
        "n_folds": 5,
    }

    for metric in table.columns:
        metrics_row[f"{metric}_mean"] = float(table[metric].mean())
        metrics_row[f"{metric}_std"] = float(table[metric].std(ddof=1))

    return oof, metrics_row