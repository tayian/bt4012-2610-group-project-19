"""Unsupervised anomaly scorers, out-of-fold (W2-C3 on v0, W3-C1 on v1).

M5 Isolation forest: features (target never used), 300 trees,
   contamination 'auto', fitted on the training fold, scored on held-out fold.
M6 Autoencoder: 120-month total cons_pm sequence (log1p) flattened + F2-F4
   features; Dense in-64-16-64-in; trained on NON-FRAUD training-fold clients
   only; score = reconstruction MSE on the held-out fold.

    python -m src.models.anomaly --features v1
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from src.models.common import load_train, TabularPrep, write_oof, oof_metrics_row, SEED
from src.models.dl import set_seed

F2_F4_PREFIX = ("cons_pm_", "share_level", "level", "n_invoices", "invoices_per_year",
                "days_since", "gap_days", "has_missing", "months_number",
                "yoy_drop", "slope_last3y", "sudden_spike", "zero_")


def m5_iforest(X, folds):
    s = np.zeros(len(X))
    for k in range(5):
        tr, va = folds != k, folds == k
        prep = TabularPrep().fit(X[tr])
        f = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=2)
        f.fit(prep.transform(X[tr]))
        s[va] = -f.score_samples(prep.transform(X[va]))   # higher = more anomalous
    return s


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def total_seq(ids):
    seq = np.load("data/processed/seq.npy", mmap_mode="r")
    sid = np.load("data/processed/seq_client_ids.npy")
    pos = pd.Series(np.arange(len(sid)), index=sid)[ids].values
    s = np.array(seq[pos], dtype=np.float32)
    tot = np.log1p(np.expm1(s[:, :, 0]) + np.expm1(s[:, :, 1]))
    return tot / 8.0


def m6_autoencoder(ids, X, y, folds, epochs=25):
    cols = [c for c in X.columns if c.startswith(F2_F4_PREFIX)]
    seq = total_seq(ids)
    s = np.zeros(len(y))
    for k in range(5):
        set_seed()
        tr, va = folds != k, folds == k
        prep = TabularPrep().fit(X.loc[tr, cols])
        Z = np.hstack([seq, prep.transform(X[cols]) / 3]).astype(np.float32)
        honest = np.where(tr & (y == 0))[0]
        rng = np.random.default_rng(SEED)
        cut = int(0.9 * len(honest))
        perm = rng.permutation(honest)
        fit_i, val_i = perm[:cut], perm[cut:]
        model = AE(Z.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_state, bad = 1e9, None, 0
        Zt = torch.as_tensor(Z)
        for ep in range(epochs):
            model.train()
            for b in np.array_split(rng.permutation(fit_i), max(1, len(fit_i) // 512)):
                opt.zero_grad()
                l = ((model(Zt[b]) - Zt[b]) ** 2).mean()
                l.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = ((model(Zt[val_i]) - Zt[val_i]) ** 2).mean().item()
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k_: v.clone() for k_, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 3:
                    break
        model.load_state_dict(best_state)
        with torch.no_grad():
            s[va] = ((model(Zt[va]) - Zt[va]) ** 2).mean(1).numpy()
        print(f"  M6 fold {k}: honest val MSE {best:.4f}", flush=True)
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="v1")
    a = p.parse_args()
    ids, X, y, folds = load_train(a.features)
    rows = []
    s5 = m5_iforest(X, folds)
    write_oof(ids, folds, s5, "M5", a.features)
    rows.append(oof_metrics_row(y, s5, folds, "M5", a.features))
    s6 = m6_autoencoder(ids, X, y, folds)
    write_oof(ids, folds, s6, "M6", a.features)
    rows.append(oof_metrics_row(y, s6, folds, "M6", a.features))
    d = pd.DataFrame(rows)
    d.to_csv(f"results/tables/anomaly_{a.features}.csv", index=False)
    print(d[["model", "pr_auc", "pr_auc_std", "roc_auc", "recall@5%"]].round(4).to_string(index=False))
