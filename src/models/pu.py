# PU learning (M11): Elkan and Noto (2008) on the LightGBM (M3) out-of-fold scores.
#
# Our labels are positive-unlabelled: target = 1 means an inspection confirmed fraud, while
# target = 0 means "not caught", not "proven honest". If the confirmed frauds were a random
# sample of all frauds, P(labelled | x) = c * P(fraud | x), where c = P(labelled | fraud).
#
# For each held-out fold k:
#   1. M3 is fitted with scale_pos_weight w = negatives / positives in its training folds,
#      which multiplies its odds by w. Undo that first: p = s / (s + w * (1 - s)).
#      Without this step the weighted scores overstate P(labelled | x), so c and the
#      estimated fraud rate are both wrong.
#   2. c = mean of p over the known frauds of fold k (Elkan and Noto's estimator e1).
#   3. The estimated true fraud rate is reported two ways: labelled rate / c (Elkan and Noto's
#      estimate of P(fraud)) and the mean of min(p / c, 1) over the fold.
#
# The OOF score is p / c without the cap of 1. Dividing by one constant per fold keeps the
# order inside each fold, so the ranking metrics equal M3's; capping would tie every client
# above c at 1 and lose ranking. Values above 1 are not probabilities (Brier is left empty).
# The fraud-rate estimate is the result that matters, and it has two opposite biases:
#   * a weak scorer gives a low c (its scores for known frauds are low), which inflates it;
#   * inspections target suspicious clients, so confirmed frauds are easier to score than
#     frauds in general, which raises c and deflates it.
# Read it as a rough range, and rerun on the tuned M3 before quoting it.
#
#   python3 -m src.models.pu --m3 v0              (python run.py pu)
#   python3 -m src.models.pu --m3 v1 --weight 3   if the tuned M3 used scale_pos_weight = 3
#
# Writes results/oof/M11_<version>.parquet, its main_results.csv row and
# results/tables/pu_estimate.csv (one row per fold plus their mean, per M3 version).

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import read_splits
from src.eval.oof import write_frames
from src.models.common import save_run


def unweight(scores, w):
    # Probabilities of a model trained with positive-class weight w, mapped back to the
    # unweighted scale: odds / w.
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, 1 - 1e-12)
    return s / (s + w * (1 - s))


def elkan_noto(p, labelled):
    # c = mean P(labelled | x) over labelled positives; returns (c, p / c uncapped).
    labelled = np.asarray(labelled).astype(bool)
    if not labelled.any():
        raise ValueError("Each fold needs at least one labelled fraud to estimate c.")
    c = float(np.mean(p[labelled]))
    return c, p / c


def pu_from_oof(oof, labels, weight="fold"):
    # oof: client_id, fold, score; labels: client_id, target (training clients).
    # weight: "fold" (negatives / positives of each training fold, as M3 uses), "none", or
    # a number (a tuned scale_pos_weight).
    d = oof[["client_id", "fold", "score"]].merge(labels, on="client_id", validate="1:1")
    if len(d) != len(oof):
        raise ValueError("Every OOF client needs a label.")
    y = d["target"].to_numpy()
    folds = d["fold"].to_numpy()
    pu = np.full(len(d), np.nan)
    rows = []
    for k in sorted(np.unique(folds)):
        va, tr = folds == k, folds != k
        if weight == "fold":
            w = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
        elif weight == "none":
            w = 1.0
        else:
            w = float(weight)
        p = unweight(d["score"].to_numpy()[va], w)
        c, pu[va] = elkan_noto(p, y[va] == 1)
        n, n_pos = int(va.sum()), int((y[va] == 1).sum())
        est = float(np.minimum(pu[va], 1.0).mean())
        rows.append(
            {
                "fold": int(k),
                "n_clients": n,
                "n_labelled_frauds": n_pos,
                "labelled_fraud_rate": n_pos / n,
                "scale_pos_weight": w,
                # Should sit near the labelled rate if the weight correction is right.
                "mean_unweighted_score": float(p.mean()),
                "c_label_frequency": c,
                "est_true_fraud_rate_labelled_over_c": min(1.0, n_pos / n / c),
                "est_true_fraud_rate": est,
                "est_hidden_fraud_rate": est - n_pos / n,
                "est_fraud_rate_among_unlabelled": (est * n - n_pos) / (n - n_pos),
            }
        )
    table = pd.DataFrame(rows)
    mean = table.drop(columns="fold").mean().to_dict()
    table = pd.concat([table, pd.DataFrame([{"fold": "mean", **mean}])], ignore_index=True)
    return d["client_id"].to_numpy(), folds, y, pu, table


def main():
    p = argparse.ArgumentParser(description="PU-adjust the M3 OOF scores (Elkan and Noto).")
    p.add_argument("--m3", default="v0", help="version of results/oof/M3_<version>.parquet")
    p.add_argument("--weight", default="fold", help='"fold", "none" or a number')
    a = p.parse_args()
    oof = pd.read_parquet(f"results/oof/M3_{a.m3}.parquet")
    clients = pd.read_parquet("data/interim/client.parquet", columns=["client_id", "target"])
    _, _, folds = read_splits("data/splits", clients)
    if set(oof["client_id"].astype(str)) != set(folds["client_id"].astype(str)):
        raise SystemExit("M3 OOF clients do not match data/splits/folds.csv.")
    labels = clients.assign(client_id=clients["client_id"].astype(str))
    oof = oof.assign(client_id=oof["client_id"].astype(str))
    ids, f, y, pu, table = pu_from_oof(oof, labels, a.weight)

    table.insert(0, "m3_version", a.m3)
    path = Path("results/tables/pu_estimate.csv")
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[old["m3_version"] != a.m3], table], ignore_index=True)
    write_frames({path: table})
    save_run(
        ids,
        f,
        y,
        pu,
        "M11",
        a.m3,
        run_id=f"M11_from_M3_{a.m3}_elkan_noto_train_cv",
        preprocessing=f"M3_{a.m3}_oof_unweighted_then_divided_by_c_per_fold_uncapped",
    )
    m = table[(table["m3_version"] == a.m3) & (table["fold"].astype(str) == "mean")].iloc[0]
    print(
        f"c = {m['c_label_frequency']:.3f}; labelled fraud rate {m['labelled_fraud_rate']:.2%}, "
        f"estimated true rate {m['est_true_fraud_rate']:.2%} (mean capped score) or "
        f"{m['est_true_fraud_rate_labelled_over_c']:.2%} (labelled rate / c)"
    )


if __name__ == "__main__":
    main()
