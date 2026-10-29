"""Validate W1 raw inputs; create splits once or reuse the committed files."""
import argparse
import math
from pathlib import Path

import pandas as pd

from src.features.aggregate import INVOICE_COLUMNS


def read_clients(path):
    clients = pd.read_csv(path, usecols=["client_id", "target"],
                          dtype={"client_id": "string"}, keep_default_na=False)
    validate_ids(clients, "raw clients")
    clients["target"] = pd.to_numeric(clients["target"], errors="raise")
    if set(clients["target"].unique()) != {0, 1}:
        raise ValueError("Raw client target must contain binary classes 0 and 1.")
    return clients


def validate_ids(frame, name):
    ids = frame["client_id"]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{name}: client IDs must be nonempty, nonmissing and unique.")


def read_splits(split_dir, clients):
    split_dir = Path(split_dir)
    train = pd.read_csv(split_dir / "train_ids.csv", usecols=["client_id"],
                        dtype={"client_id": "string"}, keep_default_na=False)
    test = pd.read_csv(split_dir / "test_ids.csv", usecols=["client_id"],
                       dtype={"client_id": "string"}, keep_default_na=False)
    folds = pd.read_csv(split_dir / "folds.csv", usecols=["client_id", "fold"],
                        dtype={"client_id": "string"}, keep_default_na=False)
    for name, frame in (("train_ids", train), ("test_ids", test), ("folds", folds)):
        validate_ids(frame, name)
    train_set, test_set = set(train.client_id), set(test.client_id)
    if not train_set or not test_set or train_set & test_set:
        raise ValueError("Train/test IDs must be nonempty and disjoint.")
    if train_set | test_set != set(clients.client_id):
        raise ValueError("Saved train/test IDs must cover the raw client table exactly.")
    if len(test) != math.ceil(0.2 * len(clients)):
        raise ValueError("Saved test IDs do not match an 80/20 split of these clients.")
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise")
    if set(folds.client_id) != train_set or set(folds["fold"]) != set(range(5)):
        raise ValueError("folds.csv must cover training IDs exactly, with folds 0 through 4.")
    folds["fold"] = folds["fold"].astype("int64")
    return train, test, folds


def prepare(clients_path, invoices_path, split_dir):
    clients = read_clients(clients_path)
    header = pd.read_csv(invoices_path, nrows=0)
    missing = set(INVOICE_COLUMNS) - set(header.columns)
    if missing:
        raise ValueError(f"Missing raw invoice columns: {sorted(missing)}")
    paths = [Path(split_dir) / f for f in ("train_ids.csv", "test_ids.csv", "folds.csv")]
    present = [path.is_file() for path in paths]
    if any(present) and not all(present):
        raise ValueError("Only some split files exist. Restore the committed set; do not re-split.")
    if not any(present):
        from src.data.split import make_splits
        make_splits(client_path=clients_path, out=str(split_dir))
    train, test, _ = read_splits(split_dir, clients)
    print(f"Data ready: {len(train):,} train / {len(test):,} test clients. Saved splits reused.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--invoices", type=Path, default=Path("data/raw/invoice_train.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    try:
        prepare(args.clients, args.invoices, args.split_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"data: {exc}\n")


if __name__ == "__main__":
    main()
