# M7 dual-input 1D-CNN (Deep Learning II):
#   sequence branch  Conv1D(32, k=3) -> Conv1D(64, k=3) -> Conv1D(64, k=12) -> global max pool
#   static branch    Dense(64)
#   head             concat -> Dense(64) -> dropout 0.3 -> sigmoid, weighted BCE
#
# M7wd wide-and-deep variant (after Zheng et al. 2018, docs/reading_notes.md): the same two
# branches plus a Conv2D branch over the 120 months reshaped to a 10 x 12 year-by-month grid,
# so a kernel compares the same calendar month across years. The grid is calendar-aligned:
# each client's window is shifted to end in December of its last invoice year, padding the
# shift with masked months, so column j is always the same calendar month. Both variants use
# the same saved folds, seed and training settings, so their OOF files compare directly.
#
#   python3 -m src.models.cnn --features v0 --save          # saves fold models for the attacks
#   python3 -m src.models.cnn --features v1 --arch widedeep

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.features.sequence import COLUMNS, last_invoice_month
from src.models.common import TabularPrep, load_train, save_run
from src.models.dl import predict, train_model

SEQ_SCALE = 8.0  # log1p kWh/month rarely exceeds ~12; keeps inputs near [0, 1.5]


class DualCNN(nn.Module):
    def __init__(self, d_static, c_in=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 12, padding=6),
            nn.ReLU(),  # yearly seasonality
        )
        self.static = nn.Sequential(nn.Linear(d_static, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, seq, xs):
        h = self.conv(seq).amax(dim=-1)
        return self.head(torch.cat([h, self.static(xs)], 1)).squeeze(-1)


class WideDeepCNN(nn.Module):
    # DualCNN plus a Conv2D branch on the (3, 10, 12) year-by-month grid.
    def __init__(self, d_static, c_in=3):
        super().__init__()
        self.conv = DualCNN(d_static, c_in).conv
        self.grid = nn.Sequential(
            nn.Conv2d(c_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.static = nn.Sequential(nn.Linear(d_static, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, seq, xs):
        h1 = self.conv(seq).amax(dim=-1)
        g = seq.reshape(seq.shape[0], seq.shape[1], 10, 12)
        h2 = self.grid(g).amax(dim=(-2, -1))
        return self.head(torch.cat([h1, h2, self.static(xs)], 1)).squeeze(-1)


ARCHS = {"cnn1d": ("M7", DualCNN), "widedeep": ("M7wd", WideDeepCNN)}


def load_seq(ids, calendar=False):
    seq = np.load("data/processed/seq.npy", mmap_mode="r")
    sid = np.load("data/processed/seq_client_ids.npy")
    pos = pd.Series(np.arange(len(sid)), index=sid)[ids].values
    s = np.array(seq[pos], dtype=np.float32)
    if calendar:
        inv = pd.read_parquet("data/interim/invoice_clean.parquet", columns=COLUMNS)
        s = to_calendar(s, last_invoice_month(inv, ids))
    return to_model_seq(s)


def to_calendar(s, last_month):
    # Shift each (120, 3) window so it ends in December: drop the oldest (11 - month) months
    # and append that many masked months. Clients without invoices (month -1) are unchanged.
    out = s.copy()
    shift = np.where(last_month >= 0, 11 - last_month, 0)
    for k in np.unique(shift[shift > 0]):
        rows = shift == k
        out[rows, :-k] = s[rows, k:]
        out[rows, -k:, :2] = 0.0
        out[rows, -k:, 2] = 1.0
    return out


def to_model_seq(s):
    # (n, 120, 3) stored tensor -> (n, 3, 120) scaled model input.
    s = s.copy()
    s[:, :, :2] /= SEQ_SCALE
    return np.ascontiguousarray(s.transpose(0, 2, 1))


def run(fv, save=False, folds_to_run=range(5), max_epochs=15, arch="cnn1d"):
    name, net = ARCHS[arch]
    ids, X, y, folds = load_train(fv)
    S = load_seq(ids, calendar=arch == "widedeep")
    oof = np.full(len(y), np.nan)
    os.makedirs("results/models", exist_ok=True)
    for k in folds_to_run:
        tr, va = folds != k, folds == k
        prep = TabularPrep().fit(X[tr])
        Xtr, Xva = prep.transform(X[tr]), prep.transform(X[va])
        print(f"  {name} {fv} fold {k}", flush=True)
        m, best = train_model(
            lambda: net(Xtr.shape[1]),
            [S[tr], Xtr],
            y[tr],
            loss="bce",
            bs=512,
            max_epochs=max_epochs,
            patience=3,
        )
        oof[va] = predict(m, [S[va], Xva])
        if save:
            torch.save(m.state_dict(), f"results/models/{name}_{fv}_fold{k}.pt")
            with open(f"results/models/prep_{fv}_fold{k}.pkl", "wb") as fh:
                pickle.dump(prep, fh)
    return ids, y, folds, oof


def main():
    p = argparse.ArgumentParser(description="Train the 1D-CNN (M7) or its wide-and-deep variant.")
    p.add_argument("--features", default="v1", choices=["v0", "v1", "v2", "v3"])
    p.add_argument("--arch", default="cnn1d", choices=sorted(ARCHS))
    p.add_argument("--save", action="store_true", help="keep fold models for the attacks")
    a = p.parse_args()
    name = ARCHS[a.arch][0]
    ids, y, folds, oof = run(a.features, a.save, arch=a.arch)
    save_run(
        ids,
        folds,
        y,
        oof,
        name,
        a.features,
        run_id=f"{name}_features_{a.features}_{a.arch}_train_cv",
        loss="bce",
        preprocessing=(
            "seq120_last_invoice_aligned"
            if a.arch == "cnn1d"
            else "seq120_calendar_aligned_10x12_grid"
        )
        + "_plus_fold_fitted_static",
    )


if __name__ == "__main__":
    main()
