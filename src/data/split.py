# Create the train/test split and CV folds once, then only validate and reuse them.
# Run: python3 -m src.data.split

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.ids import check_ids

SEED = 4012
TEST_SHARE = 0.2
N_FOLDS = 5
SPLIT_FILES = ("train_ids.csv", "test_ids.csv", "folds.csv")


def read_clients(path):
    clients = pd.read_csv(
        path, usecols=["client_id", "target"], dtype={"client_id": "string"}, keep_default_na=False
    )
    check_ids(clients["client_id"], "raw clients", unique=True)
    clients["target"] = pd.to_numeric(clients["target"], errors="raise")
    if set(clients["target"].unique()) != {0, 1}:
        raise ValueError("Raw client target must contain binary classes 0 and 1.")
    return clients


def split_clients(clients):
    # Stratified holdout, then stratified folds inside the training clients.
    train, test = train_test_split(
        clients[["client_id", "target"]],
        test_size=TEST_SHARE,
        stratify=clients["target"],
        random_state=SEED,
    )
    train = train.reset_index(drop=True)
    if train["target"].value_counts().min() < N_FOLDS:
        raise ValueError(f"Each training class needs at least {N_FOLDS} clients.")
    folds = train[["client_id"]].copy()
    folds["fold"] = -1
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_idx) in enumerate(cv.split(train, train["target"])):
        folds.loc[valid_idx, "fold"] = fold
    return train[["client_id"]], test[["client_id"]].reset_index(drop=True), folds


def write_splits(split_dir, train, test, folds):
    # mode="x" refuses to overwrite existing split files.
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(SPLIT_FILES, (train, test, folds)):
        frame.to_csv(split_dir / name, index=False, mode="x")


def read_splits(split_dir, clients):
    # Load the committed split files and check they still match the client table.
    split_dir = Path(split_dir)

    def read(name, cols):
        return pd.read_csv(
            split_dir / name, usecols=cols, dtype={"client_id": "string"}, keep_default_na=False
        )

    train = read("train_ids.csv", ["client_id"])
    test = read("test_ids.csv", ["client_id"])
    folds = read("folds.csv", ["client_id", "fold"])
    for name, frame in (("train_ids", train), ("test_ids", test), ("folds", folds)):
        check_ids(frame["client_id"], name, unique=True)
    train_set, test_set = set(train.client_id), set(test.client_id)
    if not train_set or not test_set or train_set & test_set:
        raise ValueError("Train/test IDs must be nonempty and disjoint.")
    if train_set | test_set != set(clients.client_id):
        raise ValueError("Saved train/test IDs must cover the raw client table exactly.")
    if len(test) != math.ceil(TEST_SHARE * len(clients)):
        raise ValueError("Saved test IDs do not match an 80/20 split of these clients.")
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise")
    if set(folds.client_id) != train_set or set(folds["fold"]) != set(range(N_FOLDS)):
        raise ValueError(
            f"folds.csv must cover training IDs exactly, with folds 0 to {N_FOLDS - 1}."
        )
    folds["fold"] = folds["fold"].astype("int64")
    return train, test, folds


def prepare(clients_path, split_dir):
    # Create the split files if none exist; otherwise validate and reuse them.
    clients = read_clients(clients_path)
    present = [(Path(split_dir) / name).is_file() for name in SPLIT_FILES]
    if any(present) and not all(present):
        raise ValueError("Only some split files exist. Restore the committed set; do not re-split.")
    if not any(present):
        write_splits(split_dir, *split_clients(clients))
        print(f"Created new split files in {split_dir}.")
    train, test, _ = read_splits(split_dir, clients)
    print(f"Splits ready: {len(train):,} train / {len(test):,} test clients.")


def main():
    parser = argparse.ArgumentParser(
        description="Create or validate the train/test split and folds."
    )
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    try:
        prepare(args.clients, args.split_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"splits: {exc}\n")


if __name__ == "__main__":
    main()
