"""W1-B5: M1 L2 logistic regression, C=1, on the saved five training folds."""
import argparse
from functools import partial
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.prepare_baseline import read_clients, read_splits, validate_ids
from src.eval.metrics import cv_evaluate

SEED = 4012
CATEGORICAL = {"client_catg", "district", "region"}


def categorical_columns(frame):
    return [c for c in frame if c in CATEGORICAL or not pd.api.types.is_numeric_dtype(frame[c])]


def numeric_columns(frame):
    return [c for c in frame if c not in categorical_columns(frame)]


def normalize_missing_categories(frame):
    # sklearn imputers expect np.nan rather than pandas' nullable string pd.NA.
    return frame.astype(object).where(frame.notna(), np.nan)


def model_fn(max_iter=5000):
    """Return a fresh pipeline; all learned preprocessing is fitted per fold."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("normalize_missing", FunctionTransformer(normalize_missing_categories, validate=False)),
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numeric_columns),
        ("categorical", categorical, categorical_columns),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            penalty="l2", C=1.0, solver="saga", class_weight="balanced",
            random_state=SEED, max_iter=max_iter,
        )),
    ])


def load_training(features_path, clients_path, split_dir):
    clients = read_clients(clients_path)
    train, test, folds = read_splits(split_dir, clients)
    features = pd.read_parquet(features_path)
    if "client_id" not in features or "target" in features:
        raise ValueError("Feature table must include client_id and exclude target.")
    features["client_id"] = features["client_id"].astype("string")
    validate_ids(features, "features")
    if set(features.client_id) != set(clients.client_id):
        raise ValueError("Feature table must cover the raw client IDs exactly. Run make features.")

    # Select training IDs FIRST. No held-out feature row reaches the model.
    training = train.merge(features, on="client_id", how="left", validate="one_to_one")
    training = training.merge(clients, on="client_id", how="left", validate="one_to_one")
    training = training.reset_index(drop=True)
    X_train = training.drop(columns="target")
    y_train = training["target"].astype("int64")
    if set(X_train.client_id) & set(test.client_id):
        raise ValueError("Held-out clients must not enter CV.")
    return X_train, y_train, folds


def run_baseline(features_path, clients_path, split_dir, results_dir, max_iter=5000):
    features_path, results_dir = Path(features_path), Path(results_dir)
    match = re.fullmatch(r"features_(v\d+)(?:_(full|noleak))?", features_path.stem)
    if not match:
        raise ValueError("Feature filename must look like features_v0.parquet.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    version = match.group(1)
    feature_set = features_path.stem
    run_id = f"M1_{feature_set}_L2_C1_balanced_train_cv"
    table_path = results_dir / "tables/main_results.csv"
    previous = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()

    X_train, y_train, folds = load_training(features_path, clients_path, split_dir)
    missing_history = int(X_train["n_invoices"].eq(0).sum()) if "n_invoices" in X_train else 0
    if missing_history:
        print(f"NOTE: {missing_history:,} training clients have no invoices; check raw-data completeness.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        oof, row = cv_evaluate(
            partial(model_fn, max_iter=max_iter), X_train, y_train, folds,
            model="M1", feature_version=version,
        )
    row.update({
        "run_id": run_id, "feature_set": feature_set, "evaluation_split": "train_cv",
        "penalty": "l2", "C": 1.0, "solver": "saga", "class_weight": "balanced",
        "seed": SEED, "max_iter": max_iter,
        "n_clients_without_invoices": missing_history,
    })
    if "run_id" in previous:
        previous = previous.loc[previous["run_id"].ne(run_id)]
    table = pd.DataFrame([row]) if previous.empty else pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    suffix = features_path.stem.removeprefix("features_")
    oof_path = results_dir / f"oof/M1_{suffix}.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    oof_temp = oof_path.with_suffix(".parquet.tmp")
    table_temp = table_path.with_suffix(".csv.tmp")
    try:
        oof.to_parquet(oof_temp, index=False, engine="pyarrow")
        table.to_csv(table_temp, index=False)
        oof_temp.replace(oof_path)
        table_temp.replace(table_path)
    finally:
        oof_temp.unlink(missing_ok=True)
        table_temp.unlink(missing_ok=True)
    print(f"Saved training OOF predictions: {oof_path}")
    print(f"Updated results table: {table_path}")
    print(f"CV PR-AUC: {row['pr_auc_mean']:.6f} +/- {row['pr_auc_std']:.6f}")
    print(f"CV ROC-AUC: {row['roc_auc_mean']:.6f} +/- {row['roc_auc_std']:.6f}")
    return oof, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="features_v0")
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-iter", type=int, default=5000)
    args = parser.parse_args()
    features = Path(args.features)
    if features.suffix == "":
        features = Path("data/processed") / f"{features.name}.parquet"
    try:
        run_baseline(features, args.clients, args.split_dir, args.results_dir, args.max_iter)
    except ConvergenceWarning:
        parser.exit(2, "baseline: LR did not converge. Inspect the features or increase --max-iter / LR_MAX_ITER; no new results were saved.\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"baseline: {exc}\n")


if __name__ == "__main__":
    main()
