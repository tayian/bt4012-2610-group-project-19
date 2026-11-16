# M4 MLP (Deep Learning I): 3 layers (256, 128, 64), ReLU, dropout 0.3, batch norm; focal loss
# (gamma 2) vs weighted BCE; Adam lr 1e-3; early stop on validation PR-AUC. Run on v0 with
# both losses, and on v1 with the chosen loss.
#
#   python3 -m src.models.mlp --features v0 --loss both
#   python3 -m src.models.mlp --features v1 --loss bce

import argparse

import numpy as np
import torch.nn as nn

from src.models.common import TabularPrep, load_train, save_run
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


def main():
    p = argparse.ArgumentParser(description="Train the MLP (M4) on the saved folds.")
    p.add_argument("--features", default="v1", choices=["v0", "v1", "v2"])
    p.add_argument("--loss", default="bce", choices=["bce", "focal", "both"])
    a = p.parse_args()
    for loss in ["bce", "focal"] if a.loss == "both" else [a.loss]:
        ids, y, folds, oof = run(a.features, loss)
        # Weighted BCE keeps the plain M4 file name; focal loss gets its own file.
        save_run(
            ids,
            folds,
            y,
            oof,
            "M4",
            a.features,
            run_id=f"M4_features_{a.features}_{loss}_train_cv",
            oof_name="M4" if loss == "bce" else "M4focal",
            loss=loss,
            preprocessing="fold_fitted_log1p_impute_standardise",
        )


if __name__ == "__main__":
    main()
