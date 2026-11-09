# Unit tests for src/llm/snapshot.py and get_client_snapshot(client_id).

import numpy as np

from src.llm.faithfulness import _flatten_numbers
from src.llm.retrieval import get_client_snapshot
from src.llm.snapshot import SNAPSHOT_CONFIG, compute_benford_and_rounding


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


def test_benford_and_rounding_math():
    # Generate synthetic Benford-distributed numbers -> MAD should be small.
    digits = np.repeat(np.arange(1, 10), [301, 176, 125, 97, 79, 67, 58, 51, 46])
    cons_pm = digits.astype(np.float64) * 10.0
    cons = np.array([10, 15, 22, 33, 44], dtype=np.float64)  # 2 of 5 end in 0 or 5 -> 0.4
    mad, round_share = compute_benford_and_rounding(cons, cons_pm)
    assert mad < 0.002
    assert abs(round_share - 0.4) < 1e-6


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
