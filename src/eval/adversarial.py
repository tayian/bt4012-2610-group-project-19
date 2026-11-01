"""W3-C3 / W4-C2 / W4-C3 adversarial robustness study (Deep Learning II).
Threat model: docs/threat_model.md.

Every number is recall@5% on the held-out CV fold (training clients only,
test set untouched), averaged over the 5 folds. Non-fraud clients keep their
clean inputs; every fraud client in the held-out fold is attacked.

    python -m src.eval.adversarial trees     # M3 x {clean, uniform, smooth, index-consistent} x feature groups
    python -m src.eval.adversarial cnn       # M7 x {clean, uniform, smooth, PGD}, before/after adversarial training
"""
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
# >>> STAND-IN DEPENDENCY (A's feature builder) ---------------------------------
# build_v1 / LEAKY_PREFIX come from the stand-in src/features/aggregate.py.
# Before merging: change this import to A's real v1 builder (src/features/aggregate.py
# + behavioural.py from W2-A1) so attacked clients are re-featurised exactly like
# everyone else. Function needed: build_v1(invoices_df, clients_df) -> one row per client.
from src.features.aggregate import build_v1, LEAKY_PREFIX
# <<< ------------------------------------------------------------------------------
from src.features.sequence import build_sequence
from src.models.common import load_train, TabularPrep, SEED

BUDGET = 0.30
TOP = 0.05
LEVELS = ["consommation_level_1", "consommation_level_2",
          "consommation_level_3", "consommation_level_4"]


# --------------------------------------------------------------- invoice attacks
def _scale_levels(inv, factor):
    out = inv.copy()
    for l in LEVELS:
        out[l] = np.ceil(out[l].to_numpy() * factor).astype(np.int64)
    out["cons"] = out[LEVELS].sum(axis=1)
    out["cons_pm"] = out["cons"] / out["months_number"].clip(lower=1)
    return out


def attack_uniform(inv):
    return _scale_levels(inv, np.full(len(inv), 1 - BUDGET))


def _smooth_factor(cons, budget=BUDGET):
    """Linear ramp 1 -> f_end over the client's invoices (date order) such that
    sum(cons * factor) = (1 - budget) * sum(cons). Falls back to uniform if a
    ramp alone cannot reach the budget."""
    n = len(cons)
    if n == 1 or cons.sum() == 0:
        return np.full(n, 1 - budget)
    w = np.arange(n) / (n - 1)                       # 0..1
    # factor = 1 - a*w  ->  a = budget * sum(c) / sum(c*w)
    a = budget * cons.sum() / max((cons * w).sum(), 1e-9)
    if a > 1:
        return np.full(n, 1 - budget)
    return 1 - a * w


def attack_smooth(inv):
    inv = inv.sort_values(["client_id", "invoice_date"]).reset_index(drop=True)
    f = np.empty(len(inv))
    for _, idx in inv.groupby("client_id", observed=True).indices.items():
        f[idx] = _smooth_factor(inv["cons"].to_numpy(float)[idx])
    return _scale_levels(inv, f)


def attack_index_consistent(inv):
    """Smooth mimicry + readings rewritten so new_index - old_index matches the
    lowered consumption (attacker also controls the meter reading)."""
    a = attack_smooth(inv)
    coef = a["counter_coefficient"].clip(lower=1)
    a["new_index"] = a["old_index"] + np.round(a["cons"] / coef).astype(np.int64)
    return a


ATTACKS = {"uniform": attack_uniform, "smooth": attack_smooth,
           "index_consistent": attack_index_consistent}


def load_fraud_invoices():
    folds = pd.read_csv("data/splits/folds.csv", dtype={"client_id": str})
    cli = pd.read_parquet("data/interim/client.parquet")
    cli["client_id"] = cli["client_id"].astype(str)
    fr = set(folds.merge(cli, on="client_id").query("target == 1")["client_id"])
    inv = pd.read_parquet("data/interim/invoice_clean.parquet")
    inv["client_id"] = inv["client_id"].astype(str)
    inv["counter_type"] = inv["counter_type"].astype(str)
    inv = inv[inv["client_id"].isin(fr)].reset_index(drop=True)
    return inv, cli[cli["client_id"].isin(fr)].reset_index(drop=True)


def attacked_features(cache="data/processed/attacked_v1_{}.parquet"):
    """v1_full features for every training fraud client under each attack (cached)."""
    import os
    out = {}
    need = [a for a in ATTACKS if not os.path.exists(cache.format(a))]
    if need:
        inv, cli = load_fraud_invoices()
        for a in need:
            t = time.time()
            ainv = ATTACKS[a](inv)
            f = build_v1(ainv, cli).reset_index()
            f.to_parquet(cache.format(a), index=False)
            seq, ids = build_sequence(ainv, cli["client_id"].values)
            np.save(cache.format(a).replace(".parquet", "_seq.npy"), seq)
            np.save(cache.format(a).replace(".parquet", "_ids.npy"), ids)
            drop = 1 - ainv["cons"].sum() / inv["cons"].sum()
            print(f"  attack {a}: total declared consumption -{drop:.1%} ({time.time() - t:.0f}s)")
    for a in ATTACKS:
        out[a] = pd.read_parquet(cache.format(a)).set_index("client_id")
    return out


def recall_at(y, s, top=TOP):
    k = max(1, int(len(y) * top))
    return y[np.argsort(-s)][:k].sum() / y.sum()


# --------------------------------------------------------------- trees (M3, mimicry)
GROUPS = {
    "F6 data validity": ("n_index_backwards", "n_recon_mismatch", "recon_gap_mean", "n_bad_months_number"),
    "F9 meter status": LEAKY_PREFIX,
}


def tree_study():
    # >>> STAND-IN DEPENDENCY: swap for B's src/models/trees.py (tuned M3) before merging
    from src.models.trees_standin import fit_m3
    ids, X, y, folds = load_train("v1_full")
    att = attacked_features()
    variants = {
        "M3 v1_full (all groups)": [],
        "M3 without F9 (= v1_noleak)": ["F9 meter status"],
        "M3 without F6": ["F6 data validity"],
        "M3 without F6 and F9": ["F6 data validity", "F9 meter status"],
    }
    rows = []
    for vname, drop in variants.items():
        cols = [c for c in X.columns
                if not any(c.startswith(GROUPS[g]) for g in drop)]
        for k in range(5):
            tr, va = folds != k, folds == k
            m = fit_m3(X.loc[tr, cols], y[tr])
            Xva = X.loc[va, cols]
            yv = y[va]
            clean = m.predict_proba(Xva)[:, 1]
            r = {"model": vname, "fold": k, "clean": recall_at(yv, clean)}
            fr_ids = ids[va][yv == 1]
            fr_pos = np.where(yv == 1)[0]
            for a, F in att.items():
                Xa = Xva.copy()
                Xa.iloc[fr_pos] = F.loc[fr_ids, cols].to_numpy()
                r[a] = recall_at(yv, m.predict_proba(Xa)[:, 1])
            rows.append(r)
            pd.DataFrame(rows).to_csv("results/tables/robustness_trees_by_fold.csv", index=False)
            print(f"  {vname} fold {k}: " + "  ".join(f"{c}={float(v):.3f}" for c, v in r.items()
                                                       if c not in ("model", "fold")), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/tables/robustness_trees_by_fold.csv", index=False)
    return d


def tree_defence():
    """Two cheap defences for M3 (W4-C3, tree side):
    * monotone: LightGBM monotone_constraints = +1 on F6 data-validity features,
      so more reconciliation failures can never lower the risk score
    * augment: adversarial data augmentation, each training-fold fraud gets one
      attacked copy (uniform or smooth at random, 1:1), label 1. Only frauds of
      the training fold are used, so the held-out fold stays unseen."""
    import lightgbm as lgb
    # >>> STAND-IN DEPENDENCY: swap for B's tuned M3 params from src/models/trees.py
    from src.models.trees_standin import PARAMS
    ids, X, y, folds = load_train("v1_full")
    att = attacked_features()
    cols = list(X.columns)
    mono = [1 if c.startswith(GROUPS["F6 data validity"]) else 0 for c in cols]
    rng = np.random.default_rng(SEED)
    rows = []
    for vname in ["M3 + monotone F6", "M3 + adv. augmentation", "M3 + both"]:
        for k in range(5):
            tr, va = folds != k, folds == k
            Xtr, ytr = X.loc[tr, cols], y[tr]
            if "augment" in vname or "both" in vname:
                fr = ids[tr][ytr == 1]
                pick = rng.integers(0, 2, len(fr))
                aug = np.where(pick[:, None] == 0, att["uniform"].loc[fr, cols].to_numpy(),
                               att["smooth"].loc[fr, cols].to_numpy())
                Xtr = pd.concat([Xtr, pd.DataFrame(aug, columns=cols)], ignore_index=True)
                ytr = np.concatenate([ytr, np.ones(len(fr), np.float32)])
            extra = {"monotone_constraints": mono} if ("monotone" in vname or "both" in vname) else {}
            spw = (ytr == 0).sum() / (ytr == 1).sum()
            m = lgb.LGBMClassifier(scale_pos_weight=spw, **PARAMS, **extra).fit(Xtr, ytr)
            Xva, yv = X.loc[va, cols], y[va]
            r = {"model": vname, "fold": k, "clean": recall_at(yv, m.predict_proba(Xva)[:, 1])}
            fr_ids, fr_pos = ids[va][yv == 1], np.where(yv == 1)[0]
            for a, F in att.items():
                Xa = Xva.copy()
                Xa.iloc[fr_pos] = F.loc[fr_ids, cols].to_numpy()
                r[a] = recall_at(yv, m.predict_proba(Xa)[:, 1])
            rows.append(r)
            pd.DataFrame(rows).to_csv("results/tables/robustness_treedef_by_fold.csv", index=False)
            print(f"  {vname} fold {k}: " + "  ".join(f"{c}={float(v):.3f}" for c, v in r.items()
                                                       if c not in ("model", "fold")), flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- CNN (M7, PGD)
from src.models.cnn import DualCNN, load_seq, to_model_seq, SEQ_SCALE  # noqa: E402


def project_budget(x_adv, x0, mask):
    """Keep x_adv <= x0, >= 0, unchanged on masked months, and total raw
    declared consumption >= (1 - BUDGET) * original (per client)."""
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
    """White-box PGD on the consumption channels of the sequence (channels 0, 1).
    S: (n, 3, 120) model input. Returns attacked S."""
    model.eval()
    S = torch.as_tensor(S)
    Xs = torch.as_tensor(Xs)
    x0 = S[:, :2].clone()
    mask = (S[:, 2:3] > 0.5).expand_as(x0)
    x = x0.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        logit = model(torch.cat([x, S[:, 2:]], 1), Xs)
        g, = torch.autograd.grad(logit.sum(), x)
        with torch.no_grad():
            x = project_budget(x - alpha * g.sign(), x0, mask)
    return torch.cat([x.detach(), S[:, 2:]], 1).numpy()


def predict_batched(model, S, Xs, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(S), bs):
            out.append(torch.sigmoid(model(torch.as_tensor(S[i:i + bs]),
                                           torch.as_tensor(Xs[i:i + bs]))).numpy())
    return np.concatenate(out)


def pgd_augment(steps=5, alpha=0.05):
    """Adversarial training hook for dl.train_model: PGD copies of the fraud
    examples in each batch, appended 1:1 with label 1."""
    def aug(xb, yb, model):
        pos = yb > 0.5
        if pos.sum() == 0:
            return None, None
        S, Xs = xb[0][pos], xb[1][pos]
        Sa = torch.as_tensor(pgd_seq(model, S.numpy(), Xs.numpy(), steps, alpha))
        return [Sa, Xs], torch.ones(len(Sa))
    return aug


def cnn_study(fv="v1", defend=True, folds_to_run=range(5)):
    from src.models.dl import train_model
    ids, X, y, folds = load_train(fv)
    S = load_seq(ids)
    att = attacked_features()
    att_seq = {}
    for a in att:
        base = f"data/processed/attacked_v1_{a}"
        sq, sid = np.load(base + "_seq.npy"), np.load(base + "_ids.npy")
        att_seq[a] = (pd.Series(np.arange(len(sid)), index=sid), to_model_seq(sq))
    rows = []
    for k in folds_to_run:
        tr, va = folds != k, folds == k
        prep = pickle.load(open(f"results/models/prep_{fv}_fold{k}.pkl", "rb"))
        Xva = prep.transform(X[va])
        Sva, yv = S[va], y[va]
        fr_pos = np.where(yv == 1)[0]
        fr_ids = ids[va][yv == 1]
        models = {}
        m = DualCNN(Xva.shape[1])
        m.load_state_dict(torch.load(f"results/models/M7_{fv}_fold{k}.pt"))
        models["M7"] = m
        if defend:
            Xtr = prep.transform(X[tr])
            print(f"  adversarial training M7 fold {k}", flush=True)
            md, _ = train_model(lambda: DualCNN(Xtr.shape[1]), [S[tr], Xtr], y[tr],
                                loss="bce", bs=512, max_epochs=12, patience=3,
                                augment=pgd_augment())
            torch.save(md.state_dict(), f"results/models/M7adv_{fv}_fold{k}.pt")
            models["M7 + adv. training"] = md
        for name, mdl in models.items():
            r = {"model": name, "fold": k}
            r["clean"] = recall_at(yv, predict_batched(mdl, Sva, Xva))
            for a in ["uniform", "smooth"]:
                pos_map, sq = att_seq[a]
                Sa, Xa = Sva.copy(), X[va].copy()
                Sa[fr_pos] = sq[pos_map[fr_ids].values]
                Xa.iloc[fr_pos] = att[a].loc[fr_ids, X.columns].to_numpy()
                r[a] = recall_at(yv, predict_batched(mdl, Sa, prep.transform(Xa)))
            Sa = Sva.copy()
            Sa[fr_pos] = pgd_seq(mdl, Sva[fr_pos], Xva[fr_pos], steps=10)
            r["pgd"] = recall_at(yv, predict_batched(mdl, Sa, Xva))
            rows.append(r)
            pd.DataFrame(rows).to_csv("results/tables/robustness_cnn_by_fold.csv", index=False)
            print(f"  {name} fold {k}: " + "  ".join(f"{c}={float(v):.3f}" for c, v in r.items()
                                                    if c not in ("model", "fold")), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/tables/robustness_cnn_by_fold.csv", index=False)
    return d


def cnn_mimicry_defence(fv="v1", folds_to_run=range(5)):
    """M7 trained with one mimicry-attacked copy (uniform or smooth, random) of
    each training-fold fraud, label 1, sequence and static inputs both rebuilt
    from the attacked invoices. Evaluated like cnn_study()."""
    from src.models.dl import train_model
    ids, X, y, folds = load_train(fv)
    S = load_seq(ids)
    att = attacked_features()
    att_seq = {}
    for a in ("uniform", "smooth"):
        base = f"data/processed/attacked_v1_{a}"
        sq, sid = np.load(base + "_seq.npy"), np.load(base + "_ids.npy")
        att_seq[a] = (pd.Series(np.arange(len(sid)), index=sid), to_model_seq(sq))
    rng = np.random.default_rng(SEED)
    rows = []
    for k in folds_to_run:
        tr, va = folds != k, folds == k
        prep = pickle.load(open(f"results/models/prep_{fv}_fold{k}.pkl", "rb"))
        fr = ids[tr][y[tr] == 1]
        pick = rng.integers(0, 2, len(fr))
        Sa_tr = np.stack([att_seq["uniform" if p == 0 else "smooth"][1][
            att_seq["uniform" if p == 0 else "smooth"][0][c]] for c, p in zip(fr, pick)])
        Xa_tr = pd.DataFrame(np.where(pick[:, None] == 0, att["uniform"].loc[fr, X.columns].to_numpy(),
                                      att["smooth"].loc[fr, X.columns].to_numpy()), columns=X.columns)
        S_tr = np.concatenate([S[tr], Sa_tr])
        X_tr = prep.transform(pd.concat([X[tr], Xa_tr], ignore_index=True))
        y_tr = np.concatenate([y[tr], np.ones(len(fr), np.float32)])
        print(f"  mimicry-augmented M7 fold {k}", flush=True)
        m, _ = train_model(lambda: DualCNN(X_tr.shape[1]), [S_tr, X_tr], y_tr,
                           loss="bce", bs=512, max_epochs=12, patience=3)
        Xva, Sva, yv = prep.transform(X[va]), S[va], y[va]
        fr_pos, fr_ids = np.where(yv == 1)[0], ids[va][yv == 1]
        r = {"model": "M7 + mimicry augmentation", "fold": k,
             "clean": recall_at(yv, predict_batched(m, Sva, Xva))}
        for a in ("uniform", "smooth"):
            pos_map, sq = att_seq[a]
            Sa, Xa = Sva.copy(), X[va].copy()
            Sa[fr_pos] = sq[pos_map[fr_ids].values]
            Xa.iloc[fr_pos] = att[a].loc[fr_ids, X.columns].to_numpy()
            r[a] = recall_at(yv, predict_batched(m, Sa, prep.transform(Xa)))
        Sa = Sva.copy()
        Sa[fr_pos] = pgd_seq(m, Sva[fr_pos], Xva[fr_pos], steps=10)
        r["pgd"] = recall_at(yv, predict_batched(m, Sa, Xva))
        rows.append(r)
        pd.DataFrame(rows).to_csv("results/tables/robustness_cnndef_by_fold.csv", index=False)
        print("  " + "  ".join(f"{c}={float(v):.3f}" for c, v in r.items() if c not in ("model", "fold")), flush=True)
    return pd.DataFrame(rows)


def summarise():
    import os
    parts = [pd.read_csv(f"results/tables/robustness_{p}_by_fold.csv")
             for p in ("trees", "treedef", "cnn", "cnndef") if os.path.exists(f"results/tables/robustness_{p}_by_fold.csv")]
    d = pd.concat(parts)
    long = d.melt(id_vars=["model", "fold"], var_name="attack", value_name="recall@5%").dropna()
    t = long.groupby(["model", "attack"])["recall@5%"].agg(["mean", "std"]).reset_index()
    clean = t[t.attack == "clean"].set_index("model")["mean"]
    t["drop_vs_clean"] = t["mean"] - t["model"].map(clean)
    t.to_csv("results/tables/robustness.csv", index=False)
    wide = t.pivot(index="model", columns="attack", values="mean")
    order = [c for c in ["clean", "uniform", "smooth", "index_consistent", "pgd"] if c in wide]
    print(wide[order].round(3).to_string())
    return t


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("trees", "all"):
        tree_study()
    if what in ("treedef", "all"):
        tree_defence()
    if what in ("cnndef", "all"):
        cnn_mimicry_defence()
    if what in ("cnn", "all"):
        import os
        fl = [int(f) for f in os.environ.get("CNN_FOLDS", "0,1,2,3,4").split(",")]
        cnn_study(folds_to_run=fl)
    summarise()
