# Out-of-fold score files shared by every model: columns, file names and a safe writer.
# Kept free of torch so the linear baseline runs without it.

import re
from pathlib import Path

import numpy as np
import pandas as pd

OOF_COLUMNS = ["client_id", "fold", "score", "model", "feature_version"]


def feature_version(path):
    # features_v1_with_inspection.parquet -> ("v1", "v1_with_inspection"): the version and the
    # OOF file suffix.
    stem = Path(path).stem
    match = re.fullmatch(r"features_(v\d+)(?:_\w+)?", stem)
    if not match:
        raise ValueError("Feature filename must look like features_v0.parquet.")
    return match.group(1), stem.removeprefix("features_")


def oof_frame(client_ids, folds, scores, model, version):
    return pd.DataFrame(
        {
            "client_id": pd.Series(client_ids).astype("string").to_numpy(),
            "fold": np.asarray(folds, dtype="int64"),
            "score": np.asarray(scores, dtype=np.float64),
            "model": model,
            "feature_version": version,
        }
    )[OOF_COLUMNS]


def replace_run_row(previous, row):
    # The results table with any earlier row of the same run_id replaced by row.
    if "run_id" in previous:
        previous = previous.loc[previous["run_id"].ne(row["run_id"])]
    if previous.empty:
        return pd.DataFrame([row])
    return pd.concat([previous, pd.DataFrame([row])], ignore_index=True)


def write_frames(frames):
    # Write each {path: DataFrame} as .csv or .parquet. Every file goes to a temporary path
    # first and is renamed only once all are written, so a failed run keeps the old files.
    temps = {Path(path): Path(f"{path}.tmp") for path in frames}
    try:
        for (path, temp), frame in zip(temps.items(), frames.values()):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".csv":
                frame.to_csv(temp, index=False)
            else:
                frame.to_parquet(temp, index=False)
        for path, temp in temps.items():
            temp.replace(path)
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)
