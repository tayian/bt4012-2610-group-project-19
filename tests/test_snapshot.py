# Unit tests for src/llm/snapshot.py and get_client_snapshot(client_id).

import pandas as pd

from src.llm.faithfulness import _flatten_numbers
from src.llm.retrieval import get_client_snapshot
from src.llm.snapshot import KEY_FEATURE_NAMES, SNAPSHOT_CONFIG


def test_snapshot_config_centralized():
    required_keys = {
        "features_v0_path",
        "features_v1_path",
        "oof_path",
        "shap_path",
        "interim_client_path",
        "interim_invoice_clean_path",
        "splits_dir",
        "raw_clients_path",
    }
    assert required_keys.issubset(set(SNAPSHOT_CONFIG.keys()))


def test_signals_read_from_features_v1_and_nan_kept_missing():
    f1 = pd.read_parquet(SNAPSHOT_CONFIG["features_v1_path"]).set_index("client_id")
    row = f1.loc["train_Client_100109"]
    snap = get_client_snapshot("train_Client_100109")
    assert snap["benford_mad"] == round(float(row["benford_mad"]), 4)
    assert snap["round_reading_share"] == round(float(row["round_reading_share"]), 4)
    for k in ("n_index_backwards", "n_recon_mismatch", "n_bad_months_number"):
        assert snap[k] == int(row[k])
    for k in ("yoy_drop_max", "zero_run_max", "n_index_backwards", "benford_mad"):
        assert k in KEY_FEATURE_NAMES and k in snap["key_features"]

    # benford_mad is NaN with fewer than 20 positive readings: reported as missing, not 0.
    cid = f1.index[f1["benford_mad"].isna()][0]
    snap = get_client_snapshot(cid)
    assert snap["benford_mad"] is None
    assert snap["key_features"]["benford_mad"] == {"value": None, "train_pct": None}


def test_get_client_snapshot_structure_and_values():
    snap = get_client_snapshot("train_Client_100109")
    assert snap["client_id"] == "train_Client_100109"
    assert snap["split"] == "train"
    assert isinstance(snap["risk_score"], float)
    assert 0.0 <= snap["risk_score"] <= 1.0
    assert isinstance(snap["risk_rank_pct"], float)
    assert 0.0 <= snap["risk_rank_pct"] <= 100.0

    # Profile checks.
    prof = snap["profile"]
    for k in (
        "client_catg",
        "region",
        "district",
        "tenure_years",
        "n_counters",
        "n_invoices",
        "has_elec",
        "has_gaz",
        "counter_type",
    ):
        assert k in prof
    assert prof["n_invoices"] > 0

    # Key features + training percentiles.
    kf = snap["key_features"]
    assert "cons_pm_mean" in kf
    assert "value" in kf["cons_pm_mean"]
    assert "train_pct" in kf["cons_pm_mean"]
    assert 0.0 <= kf["cons_pm_mean"]["train_pct"] <= 100.0

    # SHAP top 5 drivers.
    shap5 = snap["shap_top5"]
    assert len(shap5) == 5
    assert [s["rank"] for s in shap5] == [1, 2, 3, 4, 5]
    for s in shap5:
        assert "feature" in s and "value" in s and "shap_value" in s

    # Benford and invalid-reading counts.
    assert 0.0 <= snap["benford_mad"] <= 1.0
    assert 0.0 <= snap["round_reading_share"] <= 1.0
    for k in (
        "n_index_backwards",
        "n_index_rollover",
        "n_recon_mismatch",
        "n_bad_months_number",
        "has_missing_months",
    ):
        assert k in snap
        assert k in snap["invalid_reading_counts"]
        assert isinstance(snap[k], int)

    # Compatibility with faithfulness.py number extraction.
    flat = _flatten_numbers(snap)
    assert snap["risk_score"] in flat
    assert snap["risk_rank_pct"] in flat
