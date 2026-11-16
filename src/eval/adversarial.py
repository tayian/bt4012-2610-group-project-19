# Adversarial robustness study (Deep Learning II). Threat model: docs/threat_model.md.
#
# A fraudulent client lowers the consumption declared on their own invoices by at most 30% in
# total and tries to drop out of the top-5% inspection list. Every number is recall@5% on the
# held-out CV fold (training clients only, test set untouched), using the shared tie-aware
# metric, averaged over the 5 saved folds. Non-fraud clients keep their clean inputs; every
# fraud client in the held-out fold is attacked.
#
# Attacks on the invoices, after which features and sequences are rebuilt with the same
# pipeline code as everyone else's (src/features/aggregate.py for v0, behavioural.py for v1,
# sequence.py):
#   uniform  every invoice's consumption x 0.70
#   smooth   a multiplier that ramps linearly from 1 to f_end over the client's invoices (an
#            honest-looking slow decline), with f_end set so the total falls exactly 30%
# and, for the CNN only, a white-box attack on its input:
#   pgd      10 PGD steps on the sequence, each month only lowered, projected back onto the
#            30% budget; the static branch is held fixed
#
# Defences: adversarial augmentation for M3 (one attacked copy of each training-fold fraud),
# PGD adversarial training and mimicry augmentation for M7.
#
#   python3 -m src.eval.adversarial trees --features v0   # M3 study and defence
#   python3 -m src.eval.adversarial cnn --features v0     # M7 study; needs python run.py cnn first
#   python3 -m src.eval.adversarial all --features v0     (python run.py robustness)
#
# Writes results/tables/robustness_by_fold.csv and the summary results/tables/robustness.csv
# (mean and std over folds per model and attack). Rows of other feature versions are kept.

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.load import LEVELS
from src.eval.metrics import budget_size, expected_tp_curve
from src.eval.oof import write_frames
from src.features.sequence import build_sequence
from src.models.cnn import SEQ_SCALE, DualCNN, load_seq, to_model_seq
from src.models.common import SEED, TabularPrep, load_train

BUDGET = 0.30
TOP = 0.05
# Features an attacker moves by lowering declared consumption: F2, and in v1 also the F4/F5
# trajectory and self-deviation groups.
CONSUMPTION_PREFIX = (
    "cons_pm",
    "share_level",
    "yoy_drop",
    "slope_last3y",
    "sudden_spike",
    "zero_",
    "robust_z",
    "n_robust_z",
)
BY_FOLD = Path("results/tables/robustness_by_fold.csv")
SUMMARY = Path("results/tables/robustness.csv")


def build_features(fv, inv, cli):
    # The feature step for this version, run on (attacked) invoices of the given clients.
    # Returns one row per client indexed by client_id, with the columns of features_<fv>.
    if fv == "v0":
        from src.features.aggregate import build_features as v0

        out = v0(inv, cli)
    elif fv == "v1":
        from src.features.behavioural import build_features as v1

        out = v1(inv, cli)
    else:
        raise ValueError(f"No feature builder for {fv}.")
    return out.set_index("client_id")


# Invoice attacks
def _scale_levels(inv, factor):
    # Lower the four consumption levels and recompute cons and cons_pm the way clean.py does
    # (cons_pm = cons / months_valid, missing where the billing period is impossible).
    out = inv.copy()
    for level in LEVELS:
        out[level] = np.ceil(out[level].to_numpy() * factor).astype(np.int64)
    out["cons"] = out[LEVELS].sum(axis=1)
    out["cons_pm"] = out["cons"] / out["months_valid"]
    return out


def attack_uniform(inv):
    return _scale_levels(inv, np.full(len(inv), 1 - BUDGET))


def _smooth_factor(cons, budget=BUDGET):
    # Linear ramp 1 -> f_end over the client's invoices (date order) such that sum(cons *
    # factor) = (1 - budget) * sum(cons). Falls back to uniform if a ramp alone cannot reach
    # the budget.
    n = len(cons)
    if n == 1 or cons.sum() == 0:
        return np.full(n, 1 - budget)
    w = np.arange(n) / (n - 1)  # 0..1
    # factor = 1 - a*w  ->  a = budget * sum(c) / sum(c*w)
    a = budget * cons.sum() / max((cons * w).sum(), 1e-9)
    if a > 1:
        return np.full(n, 1 - budget)
    return 1 - a * w


def attack_smooth(inv):
    inv = inv.sort_values(["client_id", "invoice_date"]).reset_index(drop=True)
    f = np.empty(len(inv))
    cons = inv["cons"].to_numpy(float)
    for idx in inv.groupby("client_id", sort=False).indices.values():
        f[idx] = _smooth_factor(cons[idx])
    return _scale_levels(inv, f)


ATTACKS = {"uniform": attack_uniform, "smooth": attack_smooth}


def load_fraud_invoices():
    # Clean invoices and client rows of the training-split frauds (the only clients attacked).
    clients = pd.read_parquet("data/interim/client.parquet")
    clients["client_id"] = clients["client_id"].astype(str)
    folds = pd.read_csv("data/splits/folds.csv", dtype={"client_id": str})
    fraud = set(folds.merge(clients, on="client_id").query("target == 1")["client_id"])
    inv = pd.read_parquet("data/interim/invoice_clean.parquet")
    inv["client_id"] = inv["client_id"].astype(str)
    inv = inv[inv["client_id"].isin(fraud)].reset_index(drop=True)
    return inv, clients[clients["client_id"].isin(fraud)].reset_index(drop=True)


def attacked_inputs(fv, cache="data/processed/attacked_{fv}_{attack}"):
    # Features and sequences of every training fraud under each attack, cached on disk.
    out = {}
    missing = [a for a in ATTACKS if not os.path.exists(cache.format(fv=fv, attack=a) + ".parquet")]
    if missing:
        inv, cli = load_fraud_invoices()
        for a in missing:
            t = time.time()
            base = cache.format(fv=fv, attack=a)
            ainv = ATTACKS[a](inv)
            build_features(fv, ainv, cli).reset_index().to_parquet(base + ".parquet", index=False)
            seq, ids = build_sequence(ainv, cli["client_id"].values)
            np.save(base + "_seq.npy", seq)
            np.save(base + "_ids.npy", ids)
            drop = 1 - ainv["cons"].sum() / inv["cons"].sum()
            print(f"  attack {a}: declared consumption -{drop:.1%} ({time.time() - t:.0f}s)")
    for a in ATTACKS:
        base = cache.format(fv=fv, attack=a)
        feats = pd.read_parquet(base + ".parquet").set_index("client_id")
        sid = np.load(base + "_ids.npy")
        seq = to_model_seq(np.load(base + "_seq.npy"))
        out[a] = (feats, pd.Series(np.arange(len(sid)), index=sid), seq)
    return out


def recall_at(y, s, top=TOP):
    # Shared tie-aware recall in the top `top` share of clients (src/eval/metrics.py).
    y = np.asarray(y, dtype=float)
    return float(expected_tp_curve(y, s)[budget_size(len(y), top)] / y.sum())


def swap_frauds(X, rows, F, ids):
    # Copy of X with the given rows replaced by the attacked features of those clients.
    X = X.copy()
    idx = X.index[rows]
    X.loc[idx, :] = F.loc[ids, X.columns].set_axis(idx).astype(X.dtypes.to_dict())
    return X


def _log(rows, study, fv):
    r = rows[-1]
    print(
        f"  {r['model']} fold {r['fold']}: "
        + "  ".join(f"{k}={r[k]:.3f}" for k in ("clean", *ATTACKS, "pgd") if k in r),
        flush=True,
    )
    frame = pd.DataFrame(rows).assign(study=study, feature_version=fv)
    old = pd.read_csv(BY_FOLD) if BY_FOLD.exists() else pd.DataFrame()
    if not old.empty:
        old = old[~((old["study"] == study) & (old["feature_version"] == fv))]
    write_frames({BY_FOLD: pd.concat([old, frame], ignore_index=True)})


# Trees (M3)
def tree_study(fv="v0"):
    # M3 as is, M3 without the consumption group (which group carries the vulnerability), and
    # M3 with adversarial augmentation: each training-fold fraud gets one attacked copy
    # (uniform or smooth at random, 1:1), label 1. fit_m3 recomputes scale_pos_weight after
    # augmentation. Only training-fold frauds are augmented, so the held-out fold stays unseen.
    from src.models.trees import fit_m3

    ids, X, y, folds = load_train(fv)
    att = attacked_inputs(fv)
    consumption = [c for c in X.columns if c.startswith(CONSUMPTION_PREFIX)]
    rng = np.random.default_rng(SEED)
    rows = []
    for name in ("M3", "M3 without consumption features", "M3 + adv. augmentation"):
        cols = [c for c in X.columns if c not in consumption] if "without" in name else list(X)
        for k in range(5):
            tr, va = folds != k, folds == k
            Xtr, ytr = X.loc[tr, cols], y[tr]
            if "augmentation" in name:
                fr = ids[tr][ytr == 1]
                pick = rng.integers(0, 2, len(fr))
                copies = pd.concat(
                    [
                        att["uniform"][0].loc[fr[pick == 0], cols],
                        att["smooth"][0].loc[fr[pick == 1], cols],
                    ]
                ).astype(Xtr.dtypes.to_dict())
                Xtr = pd.concat([Xtr, copies], ignore_index=True)
                ytr = np.concatenate([ytr, np.ones(len(copies), ytr.dtype)])
            m = fit_m3(Xtr, ytr)
            Xva, yv = X.loc[va, cols], y[va]
            fr_pos, fr_ids = np.flatnonzero(yv == 1), ids[va][yv == 1]
            r = {"model": name, "fold": k, "clean": recall_at(yv, m.predict_proba(Xva)[:, 1])}
            for a, (F, _, _) in att.items():
                Xa = swap_frauds(Xva, fr_pos, F, fr_ids)
                r[a] = recall_at(yv, m.predict_proba(Xa)[:, 1])
            rows.append(r)
            _log(rows, "trees", fv)
    return pd.DataFrame(rows)


# CNN (M7)
def project_budget(x_adv, x0, mask):
    # Keep x_adv <= x0, >= 0, unchanged on masked months, and total raw declared consumption
    # >= (1 - BUDGET) * original (per client).
    x_adv = torch.minimum(torch.clamp(x_adv, min=0.0), x0)
    x_adv = torch.where(mask, x0, x_adv)
    raw0 = torch.expm1(x0 * SEQ_SCALE)
    raw = torch.expm1(x_adv * SEQ_SCALE)
    D = (raw0 - raw).sum(dim=(1, 2))
    cap = BUDGET * raw0.sum(dim=(1, 2))
    scale = torch.where(D > cap, cap / D.clamp(min=1e-9), torch.ones_like(D))
    raw = raw0 - (raw0 - raw) * scale[:, None, None]
    return torch.log1p(raw.clamp(min=0)) / SEQ_SCALE


def pgd_seq(model, S, Xs, steps=10, alpha=0.05):
    # White-box PGD on the consumption channels of the sequence (channels 0, 1).
    # S: (n, 3, 120) model input. Returns attacked S.
    model.eval()
    S = torch.as_tensor(S)
    Xs = torch.as_tensor(Xs)
    x0 = S[:, :2].clone()
    mask = (S[:, 2:3] > 0.5).expand_as(x0)
    x = x0.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        logit = model(torch.cat([x, S[:, 2:]], 1), Xs)
        (g,) = torch.autograd.grad(logit.sum(), x)
        with torch.no_grad():
            x = project_budget(x - alpha * g.sign(), x0, mask)
    return torch.cat([x.detach(), S[:, 2:]], 1).numpy()


def predict_batched(model, S, Xs, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(S), bs):
            batch = model(torch.as_tensor(S[i : i + bs]), torch.as_tensor(Xs[i : i + bs]))
            out.append(torch.sigmoid(batch).numpy())
    return np.concatenate(out)


def pgd_augment(steps=5, alpha=0.05):
    # Adversarial training hook for dl.train_model: PGD copies of the fraud examples in each
    # batch, appended 1:1 with label 1.
    def aug(xb, yb, model):
        pos = yb > 0.5
        if pos.sum() == 0:
            return None, None
        S, Xs = xb[0][pos], xb[1][pos]
        Sa = torch.as_tensor(pgd_seq(model, S.numpy(), Xs.numpy(), steps, alpha))
        return [Sa, Xs], torch.ones(len(Sa))

    return aug


def _evaluate_cnn(name, model, k, prep, X, S, y, ids, att):
    # Recall@5% of one fold model, clean and under each attack.
    Xva, Sva, yv = prep.transform(X), S, y
    fr_pos, fr_ids = np.flatnonzero(yv == 1), ids[yv == 1]
    r = {"model": name, "fold": k, "clean": recall_at(yv, predict_batched(model, Sva, Xva))}
    for a, (F, pos_map, seq) in att.items():
        Sa = Sva.copy()
        Sa[fr_pos] = seq[pos_map[fr_ids].to_numpy()]
        Xa = prep.transform(swap_frauds(X, fr_pos, F, fr_ids))
        r[a] = recall_at(yv, predict_batched(model, Sa, Xa))
    Sa = Sva.copy()
    Sa[fr_pos] = pgd_seq(model, Sva[fr_pos], Xva[fr_pos], steps=10)
    r["pgd"] = recall_at(yv, predict_batched(model, Sa, Xva))
    return r


def cnn_study(fv="v0", folds_to_run=range(5)):
    # The saved M7 fold models (python run.py cnn), then two defences trained per fold: PGD
    # adversarial training, and mimicry augmentation (one uniform or smooth copy of each
    # training-fold fraud, with sequence and static features rebuilt from the attacked
    # invoices).
    from src.models.dl import train_model

    ids, X, y, folds = load_train(fv)
    S = load_seq(ids)
    att = attacked_inputs(fv)
    # Resume: keep folds already finished by an earlier, interrupted run (three models each).
    rows = []
    if BY_FOLD.exists():
        old = pd.read_csv(BY_FOLD)
        old = old[(old["study"] == "cnn") & (old["feature_version"] == fv)]
        done = old.groupby("fold")["model"].nunique()
        done = set(done[done == 3].index)
        rows = old[old["fold"].isin(done)].drop(columns=["study", "feature_version"])
        rows = rows.dropna(axis=1, how="all").to_dict("records")
    else:
        done = set()
    for k in folds_to_run:
        if k in done:
            print(f"  fold {k} already done", flush=True)
            continue
        rng = np.random.default_rng(SEED + k)
        tr, va = folds != k, folds == k
        with open(f"results/models/prep_{fv}_fold{k}.pkl", "rb") as fh:
            prep = pickle.load(fh)
        assert isinstance(prep, TabularPrep)
        Xtr = prep.transform(X[tr])
        base = DualCNN(Xtr.shape[1])
        base.load_state_dict(torch.load(f"results/models/M7_{fv}_fold{k}.pt"))

        print(f"  PGD adversarial training, fold {k}", flush=True)
        pgd_model, _ = train_model(
            lambda: DualCNN(Xtr.shape[1]),
            [S[tr], Xtr],
            y[tr],
            loss="bce",
            bs=512,
            max_epochs=12,
            patience=3,
            augment=pgd_augment(),
        )

        print(f"  mimicry augmentation, fold {k}", flush=True)
        fr = ids[tr][y[tr] == 1]
        uni = rng.integers(0, 2, len(fr)) == 0
        parts = [("uniform", fr[uni]), ("smooth", fr[~uni])]
        Sa = np.concatenate([att[a][2][att[a][1][c].to_numpy()] for a, c in parts])
        Xa = pd.concat([att[a][0].loc[c, X.columns] for a, c in parts], ignore_index=True).astype(
            X.dtypes.to_dict()
        )
        X_aug = prep.transform(pd.concat([X[tr], Xa], ignore_index=True))
        mimic_model, _ = train_model(
            lambda: DualCNN(X_aug.shape[1]),
            [np.concatenate([S[tr], Sa]), X_aug],
            np.concatenate([y[tr], np.ones(len(fr), y.dtype)]),
            loss="bce",
            bs=512,
            max_epochs=12,
            patience=3,
        )

        for name, model in (
            ("M7", base),
            ("M7 + PGD adv. training", pgd_model),
            ("M7 + mimicry augmentation", mimic_model),
        ):
            rows.append(_evaluate_cnn(name, model, k, prep, X[va], S[va], y[va], ids[va], att))
            _log(rows, "cnn", fv)


def summarise():
    d = pd.read_csv(BY_FOLD)
    attacks = [c for c in ("clean", *ATTACKS, "pgd") if c in d]
    long = d.melt(
        id_vars=["study", "feature_version", "model", "fold"],
        value_vars=attacks,
        var_name="attack",
        value_name="recall@5%",
    ).dropna()
    keys = ["study", "feature_version", "model", "attack"]
    t = long.groupby(keys, sort=False)["recall@5%"].agg(["mean", "std"]).reset_index()
    clean = t[t["attack"] == "clean"].set_index(["feature_version", "model"])["mean"]
    t["drop_vs_clean"] = (
        t["mean"] - pd.Series(list(zip(t["feature_version"], t["model"]))).map(clean).to_numpy()
    )
    write_frames({SUMMARY: t})
    wide = t.pivot_table(
        index=["feature_version", "model"], columns="attack", values="mean", sort=False
    )
    print(wide[[a for a in attacks if a in wide]].round(3).to_string())


def main():
    p = argparse.ArgumentParser(description="Attack M3 and M7 and test the defences.")
    p.add_argument("study", nargs="?", default="all", choices=["trees", "cnn", "all"])
    p.add_argument("--features", default="v0", choices=["v0", "v1"])
    a = p.parse_args()
    if a.study in ("trees", "all"):
        tree_study(a.features)
    if a.study in ("cnn", "all"):
        cnn_study(a.features)
    summarise()


if __name__ == "__main__":
    main()
