# M4 MLP (Deep Learning I): 3 layers (256, 128, 64), ReLU, dropout 0.3, batch norm; focal loss
# (gamma 2) vs weighted BCE; Adam lr 1e-3; early stop on validation PR-AUC. Run on v0 with
# both losses, and on v1 with the chosen loss.
#
#   python3 -m src.models.mlp --features v0 --loss both
#   python3 -m src.models.mlp --features v1 --loss bce

import argparse

import numpy as np
import pandas as pd
import torch.nn as nn

from src.models.common import TabularPrep, load_train, oof_metrics_row, write_oof
from src.models.dl import predict, train_model


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(256, 128, 64), p=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run(fv, loss):
    ids, X, y, folds = load_train(fv)
    oof = np.zeros(len(y))
    for k in range(5):
        tr, va = folds != k, folds == k
        prep = TabularPrep().fit(X[tr])
        Xtr, Xva = prep.transform(X[tr]), prep.transform(X[va])
        print(f"  M4 {fv} {loss} fold {k}  d_in={Xtr.shape[1]}", flush=True)
        m, _ = train_model(
            lambda: MLP(Xtr.shape[1]),
            [Xtr],
            y[tr],
            loss=loss,
            bs=512,
            max_epochs=40,
            patience=5,
            verbose=False,
        )
        oof[va] = predict(m, [Xva])
    return ids, y, folds, oof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="v1")
    p.add_argument("--loss", default="bce", choices=["bce", "focal", "both"])
    a = p.parse_args()
    rows = []
    losses = ["bce", "focal"] if a.loss == "both" else [a.loss]
    for loss in losses:
        ids, y, folds, oof = run(a.features, loss)
        name = "M4" if len(losses) == 1 or loss == "bce" else "M4focal"
        write_oof(ids, folds, oof, name, a.features)
        r = oof_metrics_row(y, oof, folds, f"M4 ({loss})", a.features)
        rows.append(r)
        print(
            f"M4 {a.features} {loss}: PR-AUC {r['pr_auc']:.4f} ± {r['pr_auc_std']:.4f}", flush=True
        )
    pd.DataFrame(rows).to_csv(f"results/tables/m4_loss_compare_{a.features}.csv", index=False)
