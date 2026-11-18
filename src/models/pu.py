# M11 PU learning: Elkan and Noto (2008) applied to the M3 OOF scores, to estimate the true fraud
# rate when target = 0 means "not caught" rather than "honest". Run: python3 -m src.models.pu --m3 v0

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import read_splits
from src.eval.oof import write_frames
from src.models.common import save_run


def unweight(scores, w):
    # M3 is trained with positive-class weight w, which multiplies its odds by w; undo that
    # first, or c and the fraud-rate estimate are both overstated.
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
    # weight is "fold" (negatives / positives of each training fold, as M3 uses), "none" or a
    # number. The OOF score p / c is left uncapped: one constant per fold keeps M3's ranking,
    # while capping at 1 would tie every client above c.
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
