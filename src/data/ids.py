# Client ID checks shared by every step that reads or joins on client_id.

import pandas as pd


def check_ids(ids, name, unique=False):
    # Return the IDs as strings; fail on missing, blank or (optionally) repeated IDs.
    ids = pd.Series(ids).astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{name}: client_id must be nonempty and nonmissing.")
    if unique and ids.duplicated().any():
        raise ValueError(f"{name}: client_id must be unique.")
    return ids
