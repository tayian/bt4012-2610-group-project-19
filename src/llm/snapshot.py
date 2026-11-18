# Client snapshot for query rewriting and Inspector Briefs. Risk scores are the final model's
# (M10 on features_v4, out of fold on training clients) and the SHAP drivers are its tuned
# LightGBM component's. All input paths are in SNAPSHOT_CONFIG.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.split import read_clients, read_splits

SNAPSHOT_CONFIG: dict[str, Any] = {
    "features_v0_path": Path("data/processed/features_v0.parquet"),
    "features_v1_path": Path("data/processed/features_v4.parquet"),
    "oof_path": Path("results/oof/M10_v4.parquet"),
    "shap_path": Path("results/shap_top5_v4.parquet"),
    "interim_client_path": Path("data/interim/client.parquet"),
    "interim_invoice_clean_path": Path("data/interim/invoice_clean.parquet"),
    "interim_invoice_raw_path": Path("data/interim/invoice.parquet"),
    "splits_dir": Path("data/splits"),
    "raw_clients_path": Path("data/raw/client_train.csv"),
    "default_alert_date": "2025-06-01",
    "model_version": "M10_v4",
}

KEY_FEATURE_NAMES = (
    "cons_pm_mean",
    "cons_pm_median",
    "cons_pm_std",
    "cons_pm_min",
    "cons_pm_max",
    "cons_pm_mad",
    "cons_pm_skew",
    "cons_pm_kurt",
    "share_level_2_plus",
    "cons_pm_mean_elec",
    "cons_pm_mean_gaz",
    "invoices_per_year",
    "months_number_mean",
    "months_number_max",
    "days_since_last_invoice_mean",
    "days_since_last_invoice_max",
    "tenure_years",
    "yoy_drop_max",
    "slope_last3y",
    "sudden_spike_share",
    "zero_run_max",
    "zero_share",
    "robust_z_max",
    "n_index_backwards",
    "n_recon_mismatch",
    "n_bad_months_number",
    "benford_mad",
    "round_reading_share",
)

# Features also reported at the top level of the snapshot (definitions in
# docs/feature_list.md; NaN stays None).
SIGNAL_FLOATS = ("benford_mad", "round_reading_share")
SIGNAL_COUNTS = ("n_index_backwards", "n_recon_mismatch", "n_bad_months_number")


def _clean(value: Any, ndigits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), ndigits)


class SnapshotStore:
    # Loads the inputs once and builds snapshots with training-only feature percentiles.

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**SNAPSHOT_CONFIG, **(config or {})}
        self._loaded = False
        self._train_ids: set[str] = set()
        self._test_ids: set[str] = set()
        self._features_by_id: dict[str, dict[str, Any]] = {}
        self._train_sorted_cols: dict[str, np.ndarray] = {}
        self._oof_by_id: dict[str, dict[str, Any]] = {}
        self._shap_by_id: dict[str, list[dict[str, Any]]] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        raw_clients = read_clients(Path(self.config["raw_clients_path"]))
        train_df, test_df, _ = read_splits(Path(self.config["splits_dir"]), raw_clients)
        self._train_ids = set(train_df["client_id"].astype(str))
        self._test_ids = set(test_df["client_id"].astype(str))

        f1_path = Path(self.config["features_v1_path"])
        f0_path = Path(self.config["features_v0_path"])
        f1 = pd.read_parquet(f1_path)

        # Profile columns missing from the final feature table come from features_v0.
        profile_cols = [
            "client_catg",
            "district",
            "region",
            "tenure_years",
            "has_elec",
            "has_gaz",
            "n_counters",
        ]
        missing_profile = [c for c in profile_cols if c not in f1.columns]
        if missing_profile and f0_path.is_file():
            f0 = pd.read_parquet(f0_path, columns=["client_id", *missing_profile])
            features = f0.merge(f1, on="client_id", how="inner", validate="one_to_one")
        else:
            features = f1.copy()

        self._features_by_id = {str(row["client_id"]): row for row in features.to_dict("records")}

        # Sorted training-client values per feature, for percentile lookups.
        train_mask = features["client_id"].astype(str).isin(self._train_ids)
        train_feats = features.loc[train_mask]
        numeric_cols = [
            c
            for c in features.columns
            if c not in ("client_id", "client_catg", "district", "region")
            and pd.api.types.is_numeric_dtype(features[c])
        ]
        for col in numeric_cols:
            vals = train_feats[col].dropna().to_numpy(dtype=np.float64)
            if len(vals) > 0:
                self._train_sorted_cols[col] = np.sort(vals)

        # Risk scores and their percentile rank (0 to 100).
        oof_path = Path(self.config["oof_path"])
        if oof_path.is_file():
            oof = pd.read_parquet(oof_path)
            oof["client_id"] = oof["client_id"].astype(str)
            oof["rank_pct"] = oof["score"].rank(pct=True, method="average") * 100.0
            for row in oof.to_dict("records"):
                self._oof_by_id[str(row["client_id"])] = {
                    "score": round(float(row["score"]), 4),
                    "rank_pct": round(float(row["rank_pct"]), 2),
                    "fold": int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else None,
                    "model": str(row.get("model", self.config["model_version"])),
                    "feature_version": str(row.get("feature_version", "v0")),
                }

        shap_path = Path(self.config["shap_path"])
        if shap_path.is_file():
            shap_df = pd.read_parquet(shap_path)
            shap_df["client_id"] = shap_df["client_id"].astype(str)
            shap_df = shap_df.sort_values(["client_id", "rank"], kind="stable")
            for row in shap_df.to_dict("records"):
                cid = str(row["client_id"])
                val = None if pd.isna(row["value"]) else round(float(row["value"]), 4)
                self._shap_by_id.setdefault(cid, []).append(
                    {
                        "rank": int(row["rank"]),
                        "feature": str(row["feature"]),
                        "value": val,
                        "shap_value": round(float(row["shap_value"]), 4),
                    }
                )

        self._loaded = True

    def _train_percentile(self, col: str, value: float | None) -> float | None:
        if value is None or not np.isfinite(value):
            return None
        arr = self._train_sorted_cols.get(col)
        if arr is None or len(arr) == 0:
            return None
        left = int(np.searchsorted(arr, value, side="left"))
        right = int(np.searchsorted(arr, value, side="right"))
        pct = 100.0 * (left + right) / (2.0 * len(arr))
        return round(float(pct), 1)

    def _invoice_metrics(self, client_id: str) -> dict[str, Any]:
        # Facts that are not model features: exact meter rollovers, the largest
        # reconciliation gap and the codes seen on the client's invoices.
        clean_path = Path(self.config["interim_invoice_clean_path"])
        raw_path = Path(self.config["interim_invoice_raw_path"])

        inv_c = pd.read_parquet(
            clean_path,
            columns=["client_id", "recon_gap", "flag_index_rollover"],
            filters=[("client_id", "==", client_id)],
        )
        n_rollover = int(inv_c["flag_index_rollover"].sum())
        max_gap = inv_c["recon_gap"].max() if len(inv_c) else 0.0
        max_recon_gap = round(float(0.0 if pd.isna(max_gap) else max_gap), 2)

        codes_seen = {"counter_statue": [], "reading_remarque": [], "tarif_type": []}
        if raw_path.is_file():
            inv_r = pd.read_parquet(
                raw_path,
                columns=["client_id", "counter_statue", "reading_remarque", "tarif_type"],
                filters=[("client_id", "==", client_id)],
            )
            if len(inv_r) > 0:
                for field in ("counter_statue", "reading_remarque", "tarif_type"):
                    codes_seen[field] = sorted(
                        inv_r[field].dropna().astype(str).str.strip().unique().tolist()
                    )

        return {
            "n_index_rollover": n_rollover,
            "max_recon_gap": max_recon_gap,
            "codes_seen": codes_seen,
        }

    def get_client_snapshot(
        self,
        client_id: str,
        alert_date: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_loaded()
        cid = str(client_id)
        if cid not in self._features_by_id:
            raise KeyError(f"Client ID {cid!r} not found in features table.")

        row = self._features_by_id[cid]
        split = (
            "train" if cid in self._train_ids else ("test" if cid in self._test_ids else "unknown")
        )

        has_elec = int(row.get("has_elec", 0) or 0)
        has_gaz = int(row.get("has_gaz", 0) or 0)
        if has_elec and has_gaz:
            counter_type = "both"
        elif has_elec:
            counter_type = "ELEC"
        elif has_gaz:
            counter_type = "GAZ"
        else:
            counter_type = "both"

        tenure_val = row.get("tenure_years")
        tenure_clean = (
            None if tenure_val is None or pd.isna(tenure_val) else round(float(tenure_val), 2)
        )

        profile = {
            "client_catg": int(row["client_catg"]),
            "region": int(row["region"]),
            "district": int(row["district"]),
            "tenure_years": tenure_clean,
            "n_counters": int(row.get("n_counters", 0) or 0),
            "n_invoices": int(row.get("n_invoices", 0) or 0),
            "has_elec": has_elec,
            "has_gaz": has_gaz,
            "counter_type": counter_type,
        }

        shap_list = list(self._shap_by_id.get(cid, []))

        # The standard key features plus any feature in the client's SHAP top 5.
        feature_names = list(KEY_FEATURE_NAMES)
        for s_item in shap_list:
            fname = s_item["feature"]
            if fname not in feature_names and fname in row:
                feature_names.append(fname)

        # A missing feature (e.g. benford_mad with < 20 positive readings) is kept as None.
        key_features: dict[str, dict[str, float | None]] = {}
        for fname in feature_names:
            if fname not in row:
                continue
            val = _clean(row[fname])
            pct = None if val is None else self._train_percentile(fname, float(row[fname]))
            key_features[fname] = {"value": val, "train_pct": pct}

        oof_info = self._oof_by_id.get(cid)
        risk_score = oof_info["score"] if oof_info else None
        risk_rank_pct = oof_info["rank_pct"] if oof_info else None
        model_version = (
            f"{oof_info['model']}_{oof_info['feature_version']}"
            if oof_info
            else str(self.config["model_version"])
        )

        inv_stats = self._invoice_metrics(cid)
        signals: dict[str, Any] = {f: _clean(row.get(f)) for f in SIGNAL_FLOATS}
        for f in SIGNAL_COUNTS:
            v = row.get(f)
            signals[f] = None if v is None or pd.isna(v) else int(v)
        counts = {
            "n_index_backwards": signals["n_index_backwards"],
            "n_index_rollover": inv_stats["n_index_rollover"],
            "n_recon_mismatch": signals["n_recon_mismatch"],
            "n_bad_months_number": signals["n_bad_months_number"],
            "has_missing_months": int(row.get("has_missing_months", 0) or 0),
            "max_recon_gap": inv_stats["max_recon_gap"],
        }

        return {
            "client_id": cid,
            "split": split,
            "alert_date": str(alert_date or self.config["default_alert_date"]),
            "risk_score": risk_score,
            "risk_rank_pct": risk_rank_pct,
            "model_version": model_version,
            "profile": profile,
            "key_features": key_features,
            "shap_top5": shap_list,
            "benford_mad": signals["benford_mad"],
            "round_reading_share": signals["round_reading_share"],
            **counts,
            "invalid_reading_counts": counts,
            "codes_seen": inv_stats["codes_seen"],
        }


_DEFAULT_STORE: SnapshotStore | None = None


def get_snapshot_store(config: dict[str, Any] | None = None) -> SnapshotStore:
    global _DEFAULT_STORE
    if config is not None:
        return SnapshotStore(config=config)
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = SnapshotStore()
    return _DEFAULT_STORE


def get_client_snapshot(
    client_id: str,
    alert_date: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = get_snapshot_store(config=config)
    return store.get_client_snapshot(client_id=client_id, alert_date=alert_date)
