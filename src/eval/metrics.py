# Shared ranking and budget metrics, bootstrap intervals and the saved-fold CV evaluator.

from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.ids import check_ids
from src.data.split import N_FOLDS
from src.eval.oof import oof_frame

VALUE_COST_RATIOS = (5, 10, 20)


def expected_tp_curve(y_true, scores):
    # Expected true positives in the top k clients, for k = 0..n. Tied scores have no natural
    # order, so a tie group of size m with p positives adds p / m per client taken from it.
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    n = len(s)
    if n == 0:
        return np.zeros(1)
    new_group = np.r_[True, s[1:] != s[:-1]]
    group = np.cumsum(new_group) - 1
    starts = np.flatnonzero(new_group)
    sizes = np.diff(np.r_[starts, n])
    positives = np.add.reduceat(y, starts)
    before = np.r_[0.0, np.cumsum(positives)[:-1]]
    taken = np.arange(1, n + 1) - starts[group]
    tp = before[group] + positives[group] * taken / sizes[group]
    return np.r_[0.0, tp]


def budget_size(n, budget):
    # k = floor(n * budget), but always inspect at least one client.
    # The budget is read as the decimal it prints as, so 0.29 * 100 is exactly 29, not 28.999...
    return max(1, int(Fraction(str(budget)) * n))


def budget_metrics(y_true, scores, budgets=(0.01, 0.05, 0.10), value_cost_ratios=VALUE_COST_RATIOS):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    tp = expected_tp_curve(y_true, scores)
    out = {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
    }
    for b in budgets:
        k = budget_size(n, b)
        out[f"precision@{b * 100:g}%"] = tp[k] / k
        out[f"recall@{b * 100:g}%"] = tp[k] / y_true.sum()
    # Brier score only makes sense for probabilities.
    in_range = np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    out["brier"] = float(np.mean((scores - y_true) ** 2)) if in_range else np.nan
    # Best net recovery with C = 1 and V = ratio, over budgets k = 1..n, as in net_recovery_curve.
    k = np.arange(1, n + 1)
    for ratio in value_cost_ratios:
        net = ratio * tp[1:] - k
        best = int(np.argmax(net))
        out[f"net_opt_vc{ratio:g}"] = float(net[best])
        out[f"k_opt_share_vc{ratio:g}"] = k[best] / n
    return out


def net_recovery_curve(y_true, scores, value_per_fraud=10.0, cost_per_visit=1.0):
    # Expected Net(k) = V * TP(k) - C * k for k = 1..n, with ties handled as in
    # expected_tp_curve. The optimal budget is k[net.argmax()].
    tp = expected_tp_curve(y_true, scores)[1:]
    k = np.arange(1, len(tp) + 1)
    net = value_per_fraud * tp - cost_per_visit * k
    return k, net


def tie_groups(scores):
    # Index of each client's distinct score, 0 for the highest, and the number of distinct scores.
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    distinct, group = np.unique(-scores, return_inverse=True)
    return group, len(distinct)


def ranking_metric(tp, fp, metric):
    # PR-AUC or ROC-AUC from the positives and negatives at each distinct score, highest first.
    # Matches sklearn's average_precision_score and roc_auc_score, ties included.
    seen_tp, seen_fp = np.cumsum(tp), np.cumsum(fp)
    if metric == "roc_auc":
        # Trapezoids under the ROC curve; a tie group adds one sloped segment.
        return float(np.sum(fp * (seen_tp - tp / 2)) / (seen_tp[-1] * seen_fp[-1]))
    # Average precision: each tie group's share of recall times the precision at its threshold.
    hit = tp > 0
    return float(np.sum(tp[hit] * seen_tp[hit] / (seen_tp[hit] + seen_fp[hit])) / seen_tp[-1])


def bootstrap_values(y_true, scores, metric, n_boot=1000, seed=4012):
    # The metric on each resample of the clients. metric is "pr_auc", "roc_auc" or a function
    # metric(y_true, scores). Resamples with only one class are skipped, because PR-AUC and
    # ROC-AUC are undefined there.
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if callable(metric):

        def value(idx):
            y = y_true[idx]
            return None if y.min() == y.max() else metric(y, scores[idx])
    elif metric in ("pr_auc", "roc_auc"):
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("y_true must contain binary 0/1 labels.")
        # Scores are ranked once. Each client's distinct score and class share one code, so a
        # single bincount gives a resample's negatives and positives at every distinct score.
        group, n_groups = tie_groups(scores)
        codes = 2 * group + (y_true == 1)

        def value(idx):
            fp, tp = np.bincount(codes[idx], minlength=2 * n_groups).reshape(-1, 2).T
            return None if tp.sum() in (0, len(idx)) else ranking_metric(tp, fp, metric)
    else:
        raise ValueError(f"Unknown metric {metric!r}; use 'pr_auc', 'roc_auc' or a function.")
    vals = (value(rng.integers(0, len(y_true), len(y_true))) for _ in range(n_boot))
    return np.array([v for v in vals if v is not None], dtype=float)


def bootstrap_ci(y_true, scores, metric, n_boot=1000, seed=4012):
    # Percentile 95% interval over resampled clients; see bootstrap_values.
    vals = bootstrap_values(y_true, scores, metric, n_boot, seed)
    if not len(vals):
        return np.array([np.nan, np.nan])
    return np.percentile(vals, [2.5, 97.5])


def fold_mean_std(table):
    # Each metric's mean and sample std over a table with one row per fold.
    out = {}
    for metric in table.columns:
        out[f"{metric}_mean"] = float(table[metric].mean())
        out[f"{metric}_std"] = float(table[metric].std(ddof=1))
    return out


def cv_evaluate(model_fn, X, y, folds, *, model="model", feature_version="v0"):
    # Fit a fresh model_fn() per saved fold on training clients (X has client_id, no target) and
    # return the OOF frame and a row of each metric's mean and sample std.
    # Inputs are validated before any model is fitted.
    if not isinstance(X, pd.DataFrame) or "client_id" not in X:
        raise ValueError("X must be a DataFrame with a client_id column.")
    if "target" in X:
        raise ValueError("Remove target from X and pass it separately as y.")
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ValueError("y must have the same index and row order as X.")
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain one binary 0/1 label per X row.")

    ids = check_ids(X["client_id"], "X", unique=True)
    f = folds[["client_id", "fold"]].copy()
    f["client_id"] = check_ids(f["client_id"], "folds", unique=True)
    if set(ids) != set(f["client_id"]):
        raise ValueError("X must contain exactly the IDs in folds.csv; exclude test clients.")
    if set(f["fold"]) != set(range(N_FOLDS)):
        raise ValueError("folds must contain all five saved fold numbers, 0 through 4.")
    fold_ids = ids.map(f.set_index("client_id")["fold"]).to_numpy(dtype=int)
    features = X.drop(columns="client_id")
    for fold in range(N_FOLDS):
        valid = fold_ids == fold
        if len(np.unique(y[valid])) != 2 or len(np.unique(y[~valid])) != 2:
            raise ValueError("Every training and validation partition must contain both classes.")

    scores = np.full(len(X), np.nan)
    fold_metrics = []
    for fold in range(N_FOLDS):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        estimator = model_fn()
        estimator.fit(features.iloc[train_idx], y[train_idx])
        positive = np.flatnonzero(np.asarray(estimator.classes_) == 1)
        if len(positive) != 1:
            raise ValueError("The estimator must expose positive class 1.")
        predicted = np.asarray(estimator.predict_proba(features.iloc[valid_idx]))[:, positive[0]]
        if not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 1)).any():
            raise ValueError("predict_proba must return finite probabilities in [0, 1].")
        scores[valid_idx] = predicted
        fold_metrics.append(budget_metrics(y[valid_idx], predicted))

    oof = oof_frame(ids, fold_ids, scores, model, feature_version)
    metrics_row = {
        "model": model,
        "feature_version": feature_version,
        "n_clients": len(X),
        "n_folds": N_FOLDS,
        **fold_mean_std(pd.DataFrame(fold_metrics)),
    }
    return oof, metrics_row
