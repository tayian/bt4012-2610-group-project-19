"""M7 dual-input 1D-CNN (Deep Learning II), section 7:
  sequence branch  Conv1D(32,k=3) -> Conv1D(64,k=3) -> Conv1D(64,k=12) -> global max pool
  static branch    Dense(64)
  head             concat -> Dense(64) -> dropout 0.3 -> sigmoid, weighted BCE

    python -m src.models.cnn --features v0           # W2-C4
    python -m src.models.cnn --features v1 --save    # W3-C1 (saves fold models for attacks)
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.models.common import load_train, TabularPrep, write_oof, oof_metrics_row
from src.models.dl import train_model, predict

SEQ_SCALE = 8.0   # log1p kWh/month rarely exceeds ~12; keeps inputs near [0, 1.5]


class DualCNN(nn.Module):
    def __init__(self, d_static, c_in=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 12, padding=6), nn.ReLU(),   # yearly seasonality
        )
        self.static = nn.Sequential(nn.Linear(d_static, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, seq, xs):
        h = self.conv(seq).amax(dim=-1)
        return self.head(torch.cat([h, self.static(xs)], 1)).squeeze(-1)


def load_seq(ids):
    seq = np.load("data/processed/seq.npy", mmap_mode="r")
    sid = np.load("data/processed/seq_client_ids.npy")
    pos = pd.Series(np.arange(len(sid)), index=sid)[ids].values
    s = np.array(seq[pos], dtype=np.float32)
    return to_model_seq(s)


def to_model_seq(s):
    """(n, 120, 3) stored tensor -> (n, 3, 120) scaled model input."""
    s = s.copy()
    s[:, :, :2] /= SEQ_SCALE
    return np.ascontiguousarray(s.transpose(0, 2, 1))


def run(fv, save=False, folds_to_run=range(5), max_epochs=15):
    ids, X, y, folds = load_train(fv)
    S = load_seq(ids)
    oof = np.full(len(y), np.nan)
    os.makedirs("results/models", exist_ok=True)
    for k in folds_to_run:
        tr, va = folds != k, folds == k
        prep = TabularPrep().fit(X[tr])
        Xtr, Xva = prep.transform(X[tr]), prep.transform(X[va])
        print(f"  M7 {fv} fold {k}", flush=True)
        m, best = train_model(lambda: DualCNN(Xtr.shape[1]), [S[tr], Xtr], y[tr],
                              loss="bce", bs=512, max_epochs=max_epochs, patience=3)
        oof[va] = predict(m, [S[va], Xva])
        if save:
            torch.save(m.state_dict(), f"results/models/M7_{fv}_fold{k}.pt")
            pickle.dump(prep, open(f"results/models/prep_{fv}_fold{k}.pkl", "wb"))
    return ids, y, folds, oof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="v1")
    p.add_argument("--save", action="store_true")
    a = p.parse_args()
    ids, y, folds, oof = run(a.features, a.save)
    write_oof(ids, folds, oof, "M7", a.features)
    r = oof_metrics_row(y, oof, folds, "M7", a.features)
    pd.DataFrame([r]).to_csv(f"results/tables/m7_{a.features}.csv", index=False)
    print(f"M7 {a.features}: PR-AUC {r['pr_auc']:.4f} ± {r['pr_auc_std']:.4f}")
