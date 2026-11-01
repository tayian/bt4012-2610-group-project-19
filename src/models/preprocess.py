# Column selectors and the missing-category fix shared by the linear and neural pipelines.
# Kept free of torch so the linear baseline runs without it.

import numpy as np
import pandas as pd

CATEGORICAL = {"client_catg", "district", "region"}


def categorical_columns(frame):
    return [c for c in frame if c in CATEGORICAL or not pd.api.types.is_numeric_dtype(frame[c])]


def numeric_columns(frame):
    cats = set(categorical_columns(frame))
    return [c for c in frame if c not in cats]


def as_object(frame):
    # sklearn imputers expect np.nan rather than pandas' nullable string pd.NA.
    return frame.astype(object).where(frame.notna(), np.nan)
