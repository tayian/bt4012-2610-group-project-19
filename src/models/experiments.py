# Shared experiment I/O for the tabular models: saved-fold evaluation with cached, resumable runs.

import argparse
import hashlib
import importlib.metadata
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.eval.metrics import budget_metrics, cv_evaluate, fold_mean_std
from src.eval.oof import feature_version, replace_run_row, write_frames
from src.models.linear import load_training


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@dataclass
class Context:
    X: pd.DataFrame
    y: pd.Series
    folds: pd.DataFrame
    version: str
    suffix: str
    feature_set: str
    results: Path
    fingerprint: str

    @property
    def fold_ids(self):
        return self.X.client_id.map(self.folds.set_index("client_id").fold).to_numpy(int)

    def cache(self, name, params):
        # A larger LR iteration cap can reuse already converged candidates.
        settings = {k: v for k, v in params.items() if k != "max_iter"}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        path = self.results / "runs" / self.feature_set / name / self.fingerprint / digest
        path.mkdir(parents=True, exist_ok=True)
        return path


def context(features, clients, splits, results):
    # The fingerprint covers the training data, the model code and library versions, so a cached
    # run is reused only when none of them changed.
    features = Path(features)
    X, y, folds = load_training(features, clients, splits)
    version, suffix = feature_version(features)
    digest = hashlib.sha256()
    for frame in (X, y.to_frame(), folds):
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        digest.update(str(frame.dtypes).encode())
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "models", root / "eval", root / "data"):
        for path in sorted(directory.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "imbalanced-learn"):
        digest.update(f"{package}={importlib.metadata.version(package)}".encode())
    return Context(
        X, y, folds, version, suffix, features.stem, Path(results), digest.hexdigest()[:20]
    )


def metric_row(ctx, scores, model):
    rows = [
        budget_metrics(ctx.y[ctx.fold_ids == f], np.asarray(scores)[ctx.fold_ids == f])
        for f in range(5)
    ]
    row = {"model": model, "feature_version": ctx.version, "n_clients": len(ctx.X), "n_folds": 5}
    row.update(fold_mean_std(pd.DataFrame(rows)))
    return row


def coefficients(pipeline, fold, scope):
    classifier = pipeline.named_steps["classifier"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.DataFrame(
        {
            "feature": names,
            "coefficient": classifier.coef_[0],
            "intercept": classifier.intercept_[0],
            "fold": fold,
            "fit_scope": scope,
        }
    )


class FitTracker:
    # Wraps a fold estimator to fail on non-convergence, keep its coefficients and log progress.
    def __init__(self, estimator, fold, name, collect):
        self.estimator, self.fold, self.name, self.collect = estimator, fold, name, collect

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.estimator.fit(X, y)
        if self.collect is not None:
            self.collect.append(coefficients(self.estimator, self.fold, "cv_train"))
        print(f"  {self.name}: fold {self.fold + 1}/5 fitted", flush=True)
        return self

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def evaluate(ctx, factory, name, params, collect_coefficients=False):
    cache = ctx.cache(name, params)
    complete = cache / "complete.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        oof = pd.read_parquet(cache / "oof.parquet")
        if (
            oof.client_id.tolist() != ctx.X.client_id.tolist()
            or oof.fold.tolist() != ctx.fold_ids.tolist()
        ):
            raise ValueError(f"Checkpoint IDs/folds do not match: {cache}")
        print(f"REUSE {name}", flush=True)
        coef = pd.read_csv(cache / "coefficients.csv") if collect_coefficients else None
        return oof, saved, coef
    print(f"FIT {name}", flush=True)
    fitted, index = [], 0

    def make():
        nonlocal index
        estimator = FitTracker(factory(), index, name, fitted if collect_coefficients else None)
        index += 1
        return estimator

    oof, row = cv_evaluate(make, ctx.X, ctx.y, ctx.folds, model=name, feature_version=ctx.version)
    row.update(params)
    row.update(
        run_id=f"{name}_{ctx.feature_set}_train_cv",
        feature_set=ctx.feature_set,
        evaluation_split="train_cv",
        data_fingerprint=ctx.fingerprint,
    )
    frames = {cache / "oof.parquet": oof}
    coef = pd.concat(fitted, ignore_index=True) if collect_coefficients else None
    if coef is not None:
        frames[cache / "coefficients.csv"] = coef
    write_frames(frames)
    write_json(complete, row)
    print(f"  PR-AUC {row['pr_auc_mean']:.6f} +/- {row['pr_auc_std']:.6f}", flush=True)
    return oof, row, coef


def publish(ctx, oof, row, name):
    row = dict(row, model=name, run_id=f"{name}_{ctx.feature_set}_train_cv")
    oof = oof.assign(model=name)
    table_path = ctx.results / "tables/main_results.csv"
    previous = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    table = replace_run_row(previous, row)
    write_frames({ctx.results / f"oof/{name}_{ctx.suffix}.parquet": oof, table_path: table})
    return row


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--features", type=Path, default=Path("data/processed/features_v0.parquet"))
    p.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    p.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    return p


def get_context(args):
    return context(args.features, args.clients, args.split_dir, args.results_dir)
