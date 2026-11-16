# Fit each final-table model once on all training clients and score the held-out test clients.
#
# This is the only place the test set is used. Every model keeps the settings it had in 5-fold
# CV (the tuned M3 reads results/tables/M3_tuned_params_<version>.json), is refitted on the 108,394
# training clients, and scores the 27,099 test clients once. Writes
# results/test/<model>_<version>.parquet with the OOF columns (fold = -1).
#
#   python3 -m src.models.test_set trees        # M1, M3 default, M3 tuned, M9 (minutes)
#   python3 -m src.models.test_set widedeep     # M7wd wide-and-deep (about an hour on CPU)
#   python3 -m src.models.test_set shap         # top-5 SHAP drivers of the tuned M3 for test
#                                               # clients, joined with the training OOF drivers
#                                               # into results/shap_top5_final.parquet
#
# The fusion, metrics and bootstrap intervals are computed from these files by src.models.fusion.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.oof import oof_frame, write_frames

SEED = 4012
OUT = Path("results/test")


def load_split(fv="v1"):
    # (train ids, X, y), (test ids, X, y) in the saved split order; X holds predictor columns only.
    from src.data.split import read_clients, read_splits

    clients = read_clients("data/raw/client_train.csv")
    train, test, _ = read_splits("data/splits", clients)
    X = pd.read_parquet(f"data/processed/features_{fv}.parquet")
    X["client_id"] = X["client_id"].astype("string")
    target = clients.set_index("client_id")["target"]
    out = []
    for part in (train, test):
        frame = part.merge(X, on="client_id", how="left", validate="one_to_one")
        ids = frame.pop("client_id")
        out.append((ids.to_numpy(), frame, target.loc[ids].to_numpy()))
    return out


def save_scores(ids, scores, name, fv):
    path = OUT / f"{name}_{fv}.parquet"
    write_frames({path: oof_frame(ids, np.full(len(ids), -1), scores, name, fv)})
    print(f"{name}: {len(ids)} test scores -> {path}", flush=True)


def run_trees(fv="v1", n_jobs=4):
    from src.models import linear, trees
    from src.models.catboost_model import M9Classifier

    (tr_ids, Xtr, ytr), (te_ids, Xte, _) = load_split(fv)
    tuned = json.loads(Path(f"results/tables/M3_tuned_params_{fv}.json").read_text())
    models = {
        "M1": lambda: linear.model_fn(),
        "M3": lambda: trees.model_fn(n_jobs=n_jobs),
        "M3_tuned": lambda: trees.model_fn(**tuned, n_jobs=n_jobs),
        "M9": lambda: M9Classifier(thread_count=n_jobs),
    }
    for name, make in models.items():
        model = make().fit(Xtr, ytr)
        save_scores(te_ids, model.predict_proba(Xte)[:, 1], name, fv)


def run_widedeep(fv="v1", max_epochs=15):
    import torch

    from src.models.cnn import WideDeepCNN, load_seq
    from src.models.common import TabularPrep
    from src.models.dl import predict, train_model

    (tr_ids, Xtr, ytr), (te_ids, Xte, _) = load_split(fv)
    prep = TabularPrep().fit(Xtr)
    Str, Ste = load_seq(tr_ids, calendar=True), load_seq(te_ids, calendar=True)
    Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)
    model, best = train_model(
        lambda: WideDeepCNN(Ztr.shape[1]),
        [Str, Ztr],
        ytr.astype(np.float32),
        loss="bce",
        bs=512,
        max_epochs=max_epochs,
        patience=3,
    )
    print(f"inner validation PR-AUC {best:.4f}", flush=True)
    Path("results/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"results/models/M7wd_{fv}_full.pt")
    save_scores(te_ids, predict(model, [Ste, Zte]), "M7wd", fv)


def run_shap(fv="v1", n_jobs=4, batch_size=4096):
    import shap

    from src.eval.explain import positive_shap, top_rows
    from src.models.trees import fit_m3

    (_, Xtr, ytr), (te_ids, Xte, _) = load_split(fv)
    tuned = json.loads(Path(f"results/tables/M3_tuned_params_{fv}.json").read_text())
    model = fit_m3(Xtr, ytr, **{**tuned, "n_jobs": n_jobs})
    explainer = shap.TreeExplainer(model.estimator_, feature_perturbation="tree_path_dependent")
    parts = []
    for start in range(0, len(Xte), batch_size):
        raw = Xte.iloc[start : start + batch_size]
        values = positive_shap(explainer, model._prepare(raw))
        parts.append(top_rows(te_ids[start : start + len(raw)], raw, values))
    train = pd.read_parquet(f"results/shap_top5_{fv}.parquet").assign(split="train_oof")
    final = pd.concat([train, pd.concat(parts).assign(split="test")], ignore_index=True)
    write_frames({Path("results/shap_top5_final.parquet"): final})
    print(f"SHAP top 5 for {len(te_ids)} test clients -> results/shap_top5_final.parquet")


def main():
    p = argparse.ArgumentParser(description="Refit the final-table models and score the test set.")
    p.add_argument("step", choices=["trees", "widedeep", "shap"])
    p.add_argument("--features", default="v3")
    p.add_argument("--n-jobs", type=int, default=4)
    a = p.parse_args()
    if a.step == "trees":
        run_trees(a.features, a.n_jobs)
    elif a.step == "shap":
        run_shap(a.features, a.n_jobs)
    else:
        run_widedeep(a.features)


if __name__ == "__main__":
    main()
