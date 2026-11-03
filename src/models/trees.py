# M3 LightGBM on the saved training folds, with a reusable fitting interface.

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.data.split import SEED
from src.eval.metrics import cv_evaluate
from src.eval.oof import feature_version, write_frames
from src.models.linear import load_training
from src.models.preprocess import categorical_columns

# Default model settings. Weighting is computed in fit(), never from validation labels.
# Keep scale_pos_weight and monotone_constraints out so callers can set fold-specific values.
PARAMS = {
    "objective": "binary",
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "min_child_samples": 20,
    "random_state": SEED,
    "n_jobs": 2,
    "verbosity": -1,
}


class M3Classifier(ClassifierMixin, BaseEstimator):
    # Native LightGBM missing values and categorical splits; no scaling is needed.
    # Category levels are learned from observed training-fold values only, even if the
    # input pandas Categorical carries a larger vocabulary in its dtype.

    def __init__(self, params=None):
        self.params = params

    def _frame(self, X):
        if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
            raise ValueError("M3 expects a nonempty DataFrame with unique feature columns.")
        if {"client_id", "target", "fold"} & set(X.columns):
            raise ValueError("Pass predictor columns only; exclude client_id, target and fold.")
        return X

    def _prepare(self, X):
        X = self._frame(X)
        if set(X.columns) != set(self.feature_names_in_):
            raise ValueError("Prediction columns must match the fitted M3 features.")
        out = pd.DataFrame(index=X.index)
        for column in self.feature_names_in_:
            if column in self.categories_:
                out[column] = pd.Categorical(
                    X[column].astype("string"), categories=self.categories_[column]
                )
            else:
                values = pd.to_numeric(X[column], errors="raise").to_numpy(
                    dtype=float, na_value=np.nan
                )
                out[column] = np.where(np.isfinite(values), values, np.nan)
        return out

    def fit(self, X, y):
        X = self._frame(X)
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("y must have the same row order and index as X.")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or set(np.unique(y)) != {0, 1}:
            raise ValueError("M3 needs one binary label per row and both classes.")
        params = {**PARAMS, **(self.params or {})}
        if {"scale_pos_weight", "class_weight", "is_unbalance"} & set(params):
            raise ValueError("M3 computes scale_pos_weight from its training labels in fit().")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.categories_ = {
            column: sorted(X[column].dropna().astype("string").unique().tolist())
            for column in categorical_columns(X)
        }
        self.scale_pos_weight_ = float(np.count_nonzero(y == 0) / np.count_nonzero(y == 1))
        self.estimator_ = LGBMClassifier(**params, scale_pos_weight=self.scale_pos_weight_).fit(
            self._prepare(X), y
        )
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(self._prepare(X), validate_features=True)

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self._prepare(X), validate_features=True)


def model_fn(**params):
    # cv_evaluate calls this factory once for each saved fold.
    return M3Classifier(params=params)


def fit_m3(X, y, **params):
    # X contains predictors only. Optional LightGBM overrides support the defence study,
    # including one monotone constraint per original feature; no columns are expanded.
    return model_fn(**params).fit(X, y)


def run_lightgbm(features_path, clients_path, split_dir, results_dir, n_jobs=2):
    features_path, results_dir = Path(features_path), Path(results_dir)
    version, suffix = feature_version(features_path)
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero.")
    run_id = f"M3_{features_path.stem}_default_weighted_train_cv"
    table_path = results_dir / "tables/main_results.csv"
    previous = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()

    # This shared loader uses read_splits() and selects training clients before fitting.
    X_train, y_train, folds = load_training(features_path, clients_path, split_dir)
    oof, row = cv_evaluate(
        partial(model_fn, n_jobs=n_jobs),
        X_train,
        y_train,
        folds,
        model="M3",
        feature_version=version,
    )
    row.update(
        {
            "run_id": run_id,
            "feature_set": features_path.stem,
            "evaluation_split": "train_cv",
            "seed": SEED,
            "n_estimators": PARAMS["n_estimators"],
            "num_leaves": PARAMS["num_leaves"],
            "learning_rate": PARAMS["learning_rate"],
            "min_child_samples": PARAMS["min_child_samples"],
            "n_jobs": n_jobs,
            "scale_pos_weight": "negatives/positives_in_training_fold",
            "preprocessing": "native_categorical_and_missing",
        }
    )
    if "run_id" in previous:
        previous = previous.loc[previous["run_id"].ne(run_id)]
    table = (
        pd.DataFrame([row])
        if previous.empty
        else pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    )
    oof_path = results_dir / f"oof/M3_{suffix}.parquet"
    write_frames({oof_path: oof, table_path: table})
    print(f"Saved training OOF predictions: {oof_path}")
    print(f"Updated results table: {table_path}")
    print(f"CV PR-AUC: {row['pr_auc_mean']:.6f} +/- {row['pr_auc_std']:.6f}")
    print(f"CV ROC-AUC: {row['roc_auc_mean']:.6f} +/- {row['roc_auc_std']:.6f}")
    return oof, row


def main():
    parser = argparse.ArgumentParser(description="Evaluate weighted M3 LightGBM on saved folds.")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_v0.parquet"))
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-jobs", type=int, default=PARAMS["n_jobs"])
    args = parser.parse_args()
    try:
        run_lightgbm(args.features, args.clients, args.split_dir, args.results_dir, args.n_jobs)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"lightgbm: {exc}\n")


if __name__ == "__main__":
    main()
