# Refit each final-table model on all training clients and score the held-out test set once.
# Run: python3 -m src.models.test_set trees | widedeep | shap

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import read_clients, read_splits
from src.eval.oof import oof_frame, write_frames

SEED = 4012
OUT = Path("results/test")


def load_split(fv="v1"):
    # (train ids, X, y), (test ids, X, y) in the saved split order; X holds predictor columns only.
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


# Model libraries are imported inside each step so a step loads only what it needs.
def run_trees(fv="v1", n_jobs=4):
    from src.models import linear, trees
    from src.models.catboost_model import M9Classifier

    (tr_ids, Xtr, ytr), (te_ids, Xte, _) = load_split(fv)
    tuned = json.loads(Path(f"results/tables/M3_tuned_params_{fv}.json").read_text())
    models = {
        "M1": linear.model_fn,
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
    p.add_argument("--features", default="v4")
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
