# Unsupervised anomaly scorers, out-of-fold (run on v0 and v1).
#
# M5 Isolation forest: consumption and billing features (target never used), 300 trees,
#    contamination 'auto', fitted on the training fold, scored on the held-out fold. Profile
#    columns are left out: rare categories and missing-value patterns are "anomalous" without
#    saying anything about fraud.
# M6 Autoencoder: 120-month total cons_pm sequence (log1p) flattened + consumption and billing
#    features; Dense in-64-16-64-in; trained on NON-FRAUD training-fold clients only; score =
#    reconstruction MSE on the held-out fold.
#
#   python3 -m src.models.anomaly --features v1

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

from src.models.common import SEED, TabularPrep, load_train, save_run
from src.models.dl import set_seed

CONSUMPTION_BILLING_PREFIX = (
    "cons_pm_",
    "share_level",
    "level",
    "n_invoices",
    "invoices_per_year",
    "days_since",
    "gap_days",
    "has_missing",
    "months_number",
    "yoy_drop",
    "slope_last3y",
    "sudden_spike",
    "zero_",
    "robust_z",
    "n_robust_z",
)


def behaviour_columns(X):
    return [c for c in X.columns if c.startswith(CONSUMPTION_BILLING_PREFIX)]


def m5_iforest(X, folds):
    X = X[behaviour_columns(X)]
    s = np.zeros(len(X))
    for k in range(5):
        tr, va = folds != k, folds == k
        prep = TabularPrep().fit(X[tr])
        f = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=2)
        f.fit(prep.transform(X[tr]))
        s[va] = -f.score_samples(prep.transform(X[va]))  # higher = more anomalous
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
    cols = behaviour_columns(X)
    seq = total_seq(ids)
    s = np.zeros(len(y))
    for k in range(5):
        set_seed()
        tr, va = folds != k, folds == k
        # Scaling is fitted on the same non-fraud clients the autoencoder learns from.
        prep = TabularPrep().fit(X.loc[tr & (y == 0), cols])
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
                loss = ((model(Zt[b]) - Zt[b]) ** 2).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = ((model(Zt[val_i]) - Zt[val_i]) ** 2).mean().item()
            if vl < best - 1e-5:
                best, best_state, bad = (
                    vl,
                    {k_: v.clone() for k_, v in model.state_dict().items()},
                    0,
                )
            else:
                bad += 1
                if bad >= 3:
                    break
        model.load_state_dict(best_state)
        with torch.no_grad():
            s[va] = ((model(Zt[va]) - Zt[va]) ** 2).mean(1).numpy()
        print(f"  M6 fold {k}: honest val MSE {best:.4f}", flush=True)
    return s


def main():
    p = argparse.ArgumentParser(description="Score the isolation forest (M5) and autoencoder (M6).")
    p.add_argument("--features", default="v1", choices=["v0", "v1", "v2"])
    a = p.parse_args()
    ids, X, y, folds = load_train(a.features)
    # Anomaly scores are not probabilities, so their Brier score is left empty.
    save_run(
        ids,
        folds,
        y,
        m5_iforest(X, folds),
        "M5",
        a.features,
        run_id=f"M5_features_{a.features}_iforest300_train_cv",
        probabilities=False,
        n_estimators=300,
        preprocessing="fold_fitted_log1p_impute_standardise_unsupervised",
    )
    save_run(
        ids,
        folds,
        y,
        m6_autoencoder(ids, X, y, folds),
        "M6",
        a.features,
        run_id=f"M6_features_{a.features}_autoencoder_nonfraud_train_cv",
        probabilities=False,
        preprocessing="seq120_plus_F2_F3_trained_on_nonfraud_training_clients",
    )


if __name__ == "__main__":
    main()
