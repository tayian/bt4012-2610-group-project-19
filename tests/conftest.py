# Tests marked needs_data read files built by python run.py all, so they skip until those exist.

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILT = [
    "data/interim/client.parquet",
    "data/interim/invoice_clean.parquet",
    "data/processed/features_v1.parquet",
    "data/processed/features_v4.parquet",
]


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: needs the data built by python run.py all")


def pytest_collection_modifyitems(config, items):
    if all((ROOT / path).is_file() for path in BUILT):
        return
    skip = pytest.mark.skip(reason="needs the built data; run python run.py all first")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)
