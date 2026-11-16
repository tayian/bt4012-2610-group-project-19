# Fold-local sampling for mixed numeric/categorical LightGBM inputs.

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

STRATEGIES = ("none", "scale_pos_weight", "undersample", "smote")


def balance_training(X, y, strategy, seed):
    y = np.asarray(y)
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if strategy in ("none", "scale_pos_weight"):
        return X, y
    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        if negatives < 3 * positives:
            raise ValueError(
                "Cannot undersample to 1:3 when fewer than three negatives per positive exist."
            )
        sampler = RandomUnderSampler(sampling_strategy={0: 3 * positives}, random_state=seed)
        return sampler.fit_resample(X, y)
    if strategy != "smote":
        raise ValueError(f"Unknown imbalance strategy: {strategy}")
    if negatives <= positives:
        return X, y
    if positives < 2:
        raise ValueError("SMOTE needs at least two positive training clients.")

    from imblearn.over_sampling import SMOTE, SMOTENC

    cats = [c for c in X if isinstance(X[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in X if c not in cats]
    if not nums:
        raise ValueError("SMOTENC requires numeric features as well as categories.")
    # Imputation/scaling is used only for neighbour search and synthetic generation.
    # Preserve the original real rows, including their native missing values, for fitting.
    aux = X.copy()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    aux[nums] = scaler.fit_transform(imputer.fit_transform(X[nums]))
    for c in cats:
        marker = "__SAMPLER_MISSING__"
        while marker in X[c].cat.categories:
            marker += "_"
        aux[c] = X[c].cat.add_categories([marker]).fillna(marker)
    kwargs = dict(
        sampling_strategy={1: negatives}, random_state=seed, k_neighbors=min(5, positives - 1)
    )
    sampler = SMOTENC(categorical_features="auto", **kwargs) if cats else SMOTE(**kwargs)
    resampled, labels = sampler.fit_resample(aux, y)
    extra = resampled.iloc[len(X) :].copy()
    extra[nums] = scaler.inverse_transform(extra[nums])
    for c in nums:
        if X[c].isna().all():
            extra[c] = np.nan
        elif np.array_equal(X[c].dropna(), np.round(X[c].dropna())):
            # Integer-valued columns (counts, 0/1 flags) stay whole numbers; otherwise only
            # synthetic frauds would have values such as has_gaz = 0.4, which trees learn.
            extra[c] = np.round(extra[c])
    for c in cats:
        # The sampler's missing marker becomes missing again in the native model input.
        extra[c] = pd.Categorical(extra[c], categories=X[c].cat.categories)
    return pd.concat([X, extra], ignore_index=True), np.asarray(labels)
