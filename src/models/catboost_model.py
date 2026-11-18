# M9 CatBoost with native categorical handling, the challenger to M3 LightGBM, on the saved folds.

from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.experiments import evaluate, get_context, parser, publish
from src.models.preprocess import categorical_columns

SEED = 4012
PARAMS = {
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "verbose": False,
    "allow_writing_files": False,
}


class M9Classifier(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, thread_count=4):
        self.params = params
        self.thread_count = thread_count

    def _frame(self, X):
        X = X.copy()
        for column in self.cat_columns_:
            X[column] = X[column].astype("string").fillna("missing").astype(str)
        for column in X.columns.difference(self.cat_columns_):
            X[column] = pd.to_numeric(X[column]).astype(float).replace([np.inf, -np.inf], np.nan)
        return X

    def fit(self, X, y):
        # Imported here so the other models run without catboost installed.
        from catboost import CatBoostClassifier

        y = np.asarray(y)
        self.cat_columns_ = categorical_columns(X)
        # Positive class weighted by negatives / positives of the training fold, as for M3.
        weight = (y == 0).sum() / max(1, (y == 1).sum())
        self.model_ = CatBoostClassifier(
            **{**PARAMS, **(self.params or {})},
            scale_pos_weight=weight,
            thread_count=self.thread_count,
            cat_features=self.cat_columns_,
        )
        self.model_.fit(self._frame(X), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._frame(X))


def main():
    p = parser("M9 CatBoost with native categoricals on the saved folds.")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    try:
        ctx = get_context(args)
        params = {**PARAMS, "thread_count": args.threads}
        factory = partial(M9Classifier, thread_count=args.threads)
        oof, row, _ = evaluate(ctx, factory, "M9", params)
        publish(ctx, oof, row, "M9")
    except (ValueError, OSError) as exc:
        p.exit(2, f"catboost: {exc}\n")


if __name__ == "__main__":
    main()
