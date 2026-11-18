# M1 L2 logistic regression, C=1, on the saved five training folds.

import argparse
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.ids import check_ids
from src.data.split import read_clients, read_splits
from src.eval.metrics import cv_evaluate
from src.eval.oof import feature_version, replace_run_row, write_frames
from src.models.preprocess import as_object, categorical_columns, numeric_columns

SEED = 4012


def model_fn(max_iter=5000, penalty="l2", C=1.0, l1_ratio=None):
    # A fresh pipeline, so all learned preprocessing is fitted per fold.
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "normalize_missing",
                FunctionTransformer(as_object, validate=False, feature_names_out="one-to-one"),
            ),
            (
                "impute",
                SimpleImputer(
                    strategy="constant", fill_value="__MISSING__", keep_empty_features=True
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    penalty=penalty,
                    C=C,
                    l1_ratio=l1_ratio,
                    solver="saga",
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=max_iter,
                ),
            ),
        ]
    )


def load_training(features_path, clients_path, split_dir):
    clients = read_clients(clients_path)
    train, test, folds = read_splits(split_dir, clients)
    features = pd.read_parquet(features_path)
    if "client_id" not in features or "target" in features:
        raise ValueError("Feature table must include client_id and exclude target.")
    features["client_id"] = check_ids(features["client_id"], "features", unique=True)
    if set(features.client_id) != set(clients.client_id):
        raise ValueError(
            "Feature table must cover the raw client IDs exactly. Run python run.py features."
        )

    # Training IDs are selected first, so no held-out feature row reaches the model.
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
    version, suffix = feature_version(features_path)
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    feature_set = features_path.stem
    run_id = f"M1_{feature_set}_L2_C1_balanced_train_cv"
    table_path = results_dir / "tables/main_results.csv"
    previous = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()

    X_train, y_train, folds = load_training(features_path, clients_path, split_dir)
    missing_history = int(X_train["n_invoices"].eq(0).sum()) if "n_invoices" in X_train else 0
    if missing_history:
        print(
            f"NOTE: {missing_history:,} training clients have no invoices; check raw-data completeness."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        oof, row = cv_evaluate(
            partial(model_fn, max_iter=max_iter),
            X_train,
            y_train,
            folds,
            model="M1",
            feature_version=version,
        )
    row.update(
        {
            "run_id": run_id,
            "feature_set": feature_set,
            "evaluation_split": "train_cv",
            "penalty": "l2",
            "C": 1.0,
            "solver": "saga",
            "class_weight": "balanced",
            "seed": SEED,
            "max_iter": max_iter,
            "n_clients_without_invoices": missing_history,
        }
    )
    table = replace_run_row(previous, row)
    oof_path = results_dir / f"oof/M1_{suffix}.parquet"
    write_frames({oof_path: oof, table_path: table})
    print(f"Saved training OOF predictions: {oof_path}")
    print(f"Updated results table: {table_path}")
    print(f"CV PR-AUC: {row['pr_auc_mean']:.6f} +/- {row['pr_auc_std']:.6f}")
    print(f"CV ROC-AUC: {row['roc_auc_mean']:.6f} +/- {row['roc_auc_std']:.6f}")
    return oof, row


def main():
    parser = argparse.ArgumentParser(
        description="Train the M1 logistic regression baseline on the saved folds."
    )
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
        parser.exit(
            2,
            "baseline: LR did not converge. Inspect the features or increase --max-iter / LR_MAX_ITER; no new results were saved.\n",
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"baseline: {exc}\n")


if __name__ == "__main__":
    main()
