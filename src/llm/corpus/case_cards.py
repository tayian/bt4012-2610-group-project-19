# Build precedent case cards (data/corpus/case_cards.jsonl) for 300 training fraud clients.
#
# Non-leakage contract:
# - Sampled strictly from training split clients (data/splits/train_ids.csv) with target == 1.
# - Uses observed fields and computed features only (no invented on-site stories, tamper findings,
#   or recovered amounts). Typology links are rule-based pattern matches on observed features.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import read_clients, read_splits
from src.features import aggregate, behavioural

SEED = 4012
N_CASES = 300


def _load_or_build_features(
    interim_dir: Path,
    processed_dir: Path,
    clients: pd.DataFrame,
    invoices_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    f0_path = processed_dir / "features_v0.parquet"
    f1_path = processed_dir / "features_v1.parquet"
    if f0_path.is_file():
        f0 = pd.read_parquet(f0_path)
    else:
        f0 = aggregate.build_features(invoices_clean, clients)
    if f1_path.is_file():
        f1 = pd.read_parquet(f1_path)
    else:
        f1 = behavioural.build_features(invoices_clean, clients["client_id"])
    return f0, f1


def _compute_invoice_summaries(
    inv_clean: pd.DataFrame,
    inv_raw: pd.DataFrame,
    fraud_ids: set[str],
) -> dict[str, dict]:
    # Filter both clean and raw invoices to candidate training fraud clients only for speed.
    ic = inv_clean[inv_clean["client_id"].isin(fraud_ids)].copy()
    ir = inv_raw[inv_raw["client_id"].isin(fraud_ids)].copy()

    ic = ic.sort_values(["client_id", "invoice_date"], kind="stable")
    ic["non_rollover_backwards"] = ic["flag_index_backwards"] & (~ic["flag_index_rollover"])
    ic["is_zero"] = ic["cons"] == 0

    # Vectorised per-client data-quality and date aggregations.
    dq = ic.groupby("client_id", sort=False).agg(
        first_invoice=("invoice_date", "min"),
        last_invoice=("invoice_date", "max"),
        n_index_backwards=("non_rollover_backwards", "sum"),
        n_index_rollover=("flag_index_rollover", "sum"),
        n_recon_mismatch=("flag_recon_mismatch", "sum"),
        n_bad_months_number=("flag_bad_months_number", "sum"),
        zero_count=("is_zero", "sum"),
        total_rows=("cons", "size"),
        max_coeff=("counter_coefficient", "max"),
    )

    # Vectorised first-half vs second-half median cons_pm.
    valid = ic[~ic["flag_partial_year"] & ic["cons_pm"].notna()][["client_id", "cons_pm"]].copy()
    v_counts = valid.groupby("client_id", sort=False)["cons_pm"].transform("size")
    # Fallback to all non-null cons_pm rows for clients with <2 non-partial rows.
    short_ids = set(dq.index) - set(valid[v_counts >= 2]["client_id"].unique())
    if short_ids:
        fallback = ic[ic["client_id"].isin(short_ids) & ic["cons_pm"].notna()][
            ["client_id", "cons_pm"]
        ]
        valid = pd.concat([valid[~valid["client_id"].isin(short_ids)], fallback], ignore_index=True)

    valid_grp = valid.groupby("client_id", sort=False)["cons_pm"]
    pos = valid_grp.cumcount()
    cnt = valid_grp.transform("size")
    mid = cnt // 2
    h1 = valid[pos < mid].groupby("client_id", sort=False)["cons_pm"].median()
    h2 = valid[pos >= mid].groupby("client_id", sort=False)["cons_pm"].median()
    cnt_per_client = valid.groupby("client_id", sort=False)["cons_pm"].size()

    # Vectorised raw codes seen and specific code pattern counts.
    ir["cons_total"] = (
        ir["consommation_level_1"]
        + ir["consommation_level_2"]
        + ir["consommation_level_3"]
        + ir["consommation_level_4"]
    )
    ir["statue_clean"] = ir["counter_statue"].dropna().astype(str).str.strip()
    ir["remarque_clean"] = ir["reading_remarque"].dropna().astype(str).str.strip()
    ir["tarif_clean"] = ir["tarif_type"].dropna().astype(str).str.strip()
    ir["is_stat1_pos"] = (ir["statue_clean"] == "1") & (ir["cons_total"] > 0)
    ir["is_rem8"] = ir["remarque_clean"] == "8"

    statues_s = (
        ir[["client_id", "statue_clean"]]
        .drop_duplicates()
        .sort_values(["client_id", "statue_clean"])
        .groupby("client_id", sort=False)["statue_clean"]
        .agg(list)
    )
    remarques_s = (
        ir[["client_id", "remarque_clean"]]
        .drop_duplicates()
        .sort_values(["client_id", "remarque_clean"])
        .groupby("client_id", sort=False)["remarque_clean"]
        .agg(list)
    )
    tarifs_s = (
        ir[["client_id", "tarif_clean"]]
        .drop_duplicates()
        .sort_values(["client_id", "tarif_clean"])
        .groupby("client_id", sort=False)["tarif_clean"]
        .agg(list)
    )
    code_counts = ir.groupby("client_id", sort=False).agg(
        stat1_pos=("is_stat1_pos", "sum"),
        rem8_count=("is_rem8", "sum"),
    )

    out: dict[str, dict] = {}
    for cid, row in dq.iterrows():
        cid_str = str(cid)
        n_val = int(cnt_per_client.get(cid, 0))
        if n_val >= 2:
            m1 = float(h1.get(cid))
            m2 = float(h2.get(cid))
            t_sum = (
                f"median cons_pm {m1:.1f} in the first half of the history, "
                f"{m2:.1f} in the second half"
            )
        elif n_val == 1:
            m1 = float(h2.get(cid))
            m2 = m1
            t_sum = f"median cons_pm {m1:.1f} across 1 valid invoice"
        else:
            m1, m2, t_sum = None, None, "no valid monthly consumption periods"

        first_dt = pd.Timestamp(row["first_invoice"]).strftime("%Y-%m-%d")
        last_dt = pd.Timestamp(row["last_invoice"]).strftime("%Y-%m-%d")
        total_rows = max(1, int(row["total_rows"]))
        out[cid_str] = {
            "first_invoice": first_dt,
            "last_invoice": last_dt,
            "n_index_backwards": int(row["n_index_backwards"]),
            "n_index_rollover": int(row["n_index_rollover"]),
            "n_recon_mismatch": int(row["n_recon_mismatch"]),
            "n_bad_months_number": int(row["n_bad_months_number"]),
            "zero_share": float(row["zero_count"]) / total_rows,
            "max_coeff": float(row["max_coeff"]),
            "h1_median": m1,
            "h2_median": m2,
            "trajectory_summary": t_sum,
            "codes_seen": {
                "counter_statue": statues_s.get(cid, []),
                "reading_remarque": remarques_s.get(cid, []),
                "tarif_type": tarifs_s.get(cid, []),
            },
            "stat1_pos": int(code_counts.at[cid, "stat1_pos"]) if cid in code_counts.index else 0,
            "rem8_count": int(code_counts.at[cid, "rem8_count"]) if cid in code_counts.index else 0,
        }
    return out


def _match_typologies(rec: dict) -> list[str]:
    # Rule-based pattern matching against playbook signatures (observed data only).
    matched: list[str] = []
    m1, m2 = rec["h1_median"], rec["h2_median"]
    z_share = rec["zero_share"]
    mean_c = rec["cons_pm_mean"] or 0.0
    med_c = rec["cons_pm_median"] or 0.0
    std_c = rec["cons_pm_std"] or 0.0
    max_c = rec["cons_pm_max"] or 0.0
    min_c = rec["cons_pm_min"] or 0.0
    lvl2 = rec["share_level_2_plus"] or 0.0
    statues = set(rec["codes_seen"]["counter_statue"])
    remarques = set(rec["codes_seen"]["reading_remarque"])

    # TYP-01: Direct Meter Bypass / Phase Shunt (sharp drop + zeros)
    if m1 is not None and m2 is not None and m1 > 40.0 and m2 < 0.40 * m1 and z_share > 0.08:
        matched.append("TYP-01")
    # TYP-02: External Magnet Slowdown (sustained drop without complete zeros)
    if m1 is not None and m2 is not None and m1 > 40.0 and 0.0 < m2 < 0.60 * m1 and z_share <= 0.08:
        matched.append("TYP-02")
    # TYP-03: Internal Register / Index Rollback (backwards reading or reconciliation mismatches)
    if rec["n_index_backwards"] > 0 or rec["n_recon_mismatch"] >= 2:
        matched.append("TYP-03")
    # TYP-04: Meter-Reader Collusion & Under-Reporting (low variance with remark 6/9)
    if "9" in remarques and mean_c > 20.0 and 0.0 < std_c < 0.35 * mean_c:
        matched.append("TYP-04")
    # TYP-05: Unauthorised Self-Reconnection (consumption under counter_statue 1)
    if rec["stat1_pos"] > 0:
        matched.append("TYP-05")
    # TYP-06: Meter Jamming & Mechanical Breakdown (statue 3/4 or high zeros + missing months)
    if ("3" in statues or "4" in statues) or (z_share >= 0.25 and rec["has_missing_months"] == 1):
        matched.append("TYP-06")
    # TYP-07: Tariff Misclassification (high consumption in category 11)
    if rec["has_elec"] == 1 and rec["client_catg"] == "11" and (mean_c > 550.0 or lvl2 > 0.55):
        matched.append("TYP-07")
    # TYP-08: Estimated Reading Abuse (repeated remark 8)
    if rec["rem8_count"] >= 5:
        matched.append("TYP-08")
    # TYP-09: Catch-Up Billing After Intermittent Tampering (extreme max-to-min spike ratio)
    if max_c / (min_c + 1.0) > 30.0 and max_c > 3.0 * (med_c + 1.0):
        matched.append("TYP-09")
    # TYP-10: Neutral Disconnection & Terminal Inversion (dual fuel with very low ELEC vs GAZ)
    if (
        rec["has_elec"] == 1
        and rec["has_gaz"] == 1
        and (rec.get("cons_pm_mean_elec") or 0.0) < 25.0
        and (rec.get("cons_pm_mean_gaz") or 0.0) > 60.0
    ):
        matched.append("TYP-10")
    # TYP-11: Instrument Transformer (CT/PT) Tampering (counter_coefficient > 1)
    if rec["has_elec"] == 1 and rec["max_coeff"] > 1.0:
        matched.append("TYP-11")
    # TYP-12: Direct Service Drop Tapping (long tenure with near-zero consumption)
    if rec["has_elec"] == 1 and (rec["tenure_years"] or 0.0) > 5.0 and mean_c < 15.0:
        matched.append("TYP-12")

    if not matched:
        if m1 is not None and m2 is not None and m2 < m1:
            matched.append("TYP-02")
        elif max_c > 2.5 * (med_c + 1.0):
            matched.append("TYP-09")
        else:
            matched.append("TYP-01")

    return matched[:3]


def _select_stratified_cases(records: list[dict], n_cases: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_id = {r["client_id"]: r for r in records}
    selected: list[str] = []
    selected_set: set[str] = set()

    def add_Pool(pool_ids: list[str], quota: int):
        avail = sorted(cid for cid in pool_ids if cid not in selected_set)
        if not avail or quota <= 0:
            return
        take = min(len(avail), quota)
        chosen = rng.choice(avail, size=take, replace=False).tolist()
        for cid in sorted(chosen):
            selected.append(cid)
            selected_set.add(cid)

    # 1. Ensure rare non-rollover backwards readings in training fraud are included (TYP-03).
    backwards_ids = [r["client_id"] for r in records if r["n_index_backwards"] > 0]
    add_Pool(backwards_ids, min(15, len(backwards_ids)))

    # 2. Ensure coverage of the 4 spec pattern buckets + rare categories (client_catg 12, 51).
    catg_51 = [r["client_id"] for r in records if r["client_catg"] == "51"]
    catg_12 = [r["client_id"] for r in records if r["client_catg"] == "12"]
    add_Pool(catg_51, 20)
    add_Pool(catg_12, 10)

    drop_ids = [
        r["client_id"]
        for r in records
        if "TYP-01" in r["matched_typologies"] or "TYP-02" in r["matched_typologies"]
    ]
    recon_ids = [r["client_id"] for r in records if "TYP-03" in r["matched_typologies"]]
    stat1_ids = [r["client_id"] for r in records if "TYP-05" in r["matched_typologies"]]
    rem8_ids = [r["client_id"] for r in records if "TYP-08" in r["matched_typologies"]]

    add_Pool(drop_ids, 60)
    add_Pool(recon_ids, 35)
    add_Pool(stat1_ids, 55)
    add_Pool(rem8_ids, 55)

    # 3. Fill the remainder stratified across region so all regions are represented.
    remaining_needed = n_cases - len(selected)
    if remaining_needed > 0:
        by_region: dict[str, list[str]] = {}
        for r in records:
            cid = r["client_id"]
            if cid not in selected_set:
                by_region.setdefault(r["region"], []).append(cid)
        regions = sorted(by_region)
        total_avail = sum(len(v) for v in by_region.values())
        for reg in regions:
            share = max(1, int(round(remaining_needed * len(by_region[reg]) / max(1, total_avail))))
            add_Pool(by_region[reg], min(share, n_cases - len(selected)))

    if len(selected) < n_cases:
        rest = [r["client_id"] for r in records if r["client_id"] not in selected_set]
        add_Pool(rest, n_cases - len(selected))

    final_ids = sorted(selected[:n_cases])
    return [by_id[cid] for cid in final_ids]


def format_case_card_text(card: dict) -> str:
    meters = []
    if card["has_elec"]:
        meters.append("ELEC")
    if card["has_gaz"]:
        meters.append("GAZ")
    meter_str = ", ".join(meters) if meters else "NONE"

    y_start = card["first_invoice"][:4]
    y_end = card["last_invoice"][:4]
    c = card["consumption"]
    dq = card["data_quality"]
    cs = card["codes_seen"]

    missing_txt = "missing months present" if dq["has_missing_months"] else "no missing-month gaps"
    statues = ", ".join(cs["counter_statue"]) or "none"
    remarques = ", ".join(cs["reading_remarque"]) or "none"
    tarifs = ", ".join(cs["tarif_type"]) or "none"
    typs = ", ".join(card["matched_typologies"])

    lines = [
        f"[{card['id']}] Region {card['region']}, District {card['district']} | "
        f"client_catg {card['client_catg']} | Meters: {meter_str}",
        f"Tenure: {card['tenure_years']:.1f} years | "
        f"Invoices: {card['n_invoices']} ({y_start}-{y_end}) | "
        f"Split: {card['split']} | target = {card['target']} (fraud confirmed)",
        f"Consumption: mean cons_pm {c['cons_pm_mean']:.1f}, "
        f"median {c['cons_pm_median']:.1f}, std {c['cons_pm_std']:.1f}, "
        f"upper-tariff share {c['share_level_2_plus']:.2f}; {c['trajectory_summary']}.",
        f"Data quality: {dq['n_index_backwards']} backwards readings, "
        f"{dq['n_index_rollover']} meter rollovers, "
        f"{dq['n_recon_mismatch']} reconciliation mismatches, "
        f"{dq['n_bad_months_number']} bad-month rows, {missing_txt}.",
        f"Codes seen: counter_statue {statues}; reading_remarque {remarques}; tarif_type {tarifs}.",
        f"Pattern match (not a finding): {typs}.",
    ]
    return "\n".join(lines)


def case_card_to_chunk(card: dict) -> dict:
    if card["has_elec"] and card["has_gaz"]:
        ctype = "both"
    elif card["has_elec"]:
        ctype = "ELEC"
    elif card["has_gaz"]:
        ctype = "GAZ"
    else:
        ctype = "both"

    return {
        "id": card["id"],
        "text": format_case_card_text(card),
        "doc": "case",
        "section": "precedent_case",
        "version": "v1.0",
        "valid_from": "2005-01-01",
        "valid_to": None,
        "counter_type": ctype,
        "jurisdiction": "TN",
        "synthetic": False,
        "client_id": card["client_id"],
        "client_catg": card["client_catg"],
        "district": card["district"],
        "region": card["region"],
        "split": card["split"],
        "matched_typologies": card["matched_typologies"],
    }


def build_case_cards(
    raw_clients_path: Path = Path("data/raw/client_train.csv"),
    split_dir: Path = Path("data/splits"),
    interim_dir: Path = Path("data/interim"),
    processed_dir: Path = Path("data/processed"),
    out_path: Path = Path("data/corpus/case_cards.jsonl"),
    n_cases: int = N_CASES,
    seed: int = SEED,
) -> list[dict]:
    raw_clients = read_clients(raw_clients_path)
    train_df, test_df, _ = read_splits(split_dir, raw_clients)
    train_ids = set(train_df["client_id"].astype(str))
    test_ids = set(test_df["client_id"].astype(str))

    clients = pd.read_parquet(interim_dir / "client.parquet")
    clean_cols = [
        "client_id",
        "invoice_date",
        "counter_coefficient",
        "cons",
        "cons_pm",
        "flag_index_backwards",
        "flag_index_rollover",
        "flag_recon_mismatch",
        "flag_bad_months_number",
        "flag_partial_year",
    ]
    if (
        not (processed_dir / "features_v0.parquet").is_file()
        or not (processed_dir / "features_v1.parquet").is_file()
    ):
        inv_clean = pd.read_parquet(interim_dir / "invoice_clean.parquet")
    else:
        inv_clean = pd.read_parquet(interim_dir / "invoice_clean.parquet", columns=clean_cols)
    inv_raw = pd.read_parquet(
        interim_dir / "invoice.parquet",
        columns=[
            "client_id",
            "counter_statue",
            "reading_remarque",
            "tarif_type",
            "consommation_level_1",
            "consommation_level_2",
            "consommation_level_3",
            "consommation_level_4",
        ],
    )
    f0, f1 = _load_or_build_features(interim_dir, processed_dir, clients, inv_clean)

    # Merge profile + v0 + v1 for training fraud clients only.
    fraud_clients = clients[
        clients["client_id"].astype(str).isin(train_ids) & (clients["target"] == 1)
    ][["client_id", "target"]].copy()
    fraud_ids = set(fraud_clients["client_id"].astype(str))
    if fraud_ids & test_ids:
        raise ValueError("Leakage check failed: test client IDs found in fraud candidate pool.")

    merged = fraud_clients.merge(
        f0[
            [
                "client_id",
                "client_catg",
                "district",
                "region",
                "tenure_years",
                "has_elec",
                "has_gaz",
            ]
        ],
        on="client_id",
        how="left",
    ).merge(
        f1[
            [
                "client_id",
                "n_invoices",
                "cons_pm_mean",
                "cons_pm_median",
                "cons_pm_std",
                "cons_pm_min",
                "cons_pm_max",
                "share_level_2_plus",
                "cons_pm_mean_elec",
                "cons_pm_mean_gaz",
                "has_missing_months",
            ]
        ],
        on="client_id",
        how="left",
    )
    merged = merged[merged["n_invoices"] > 0].sort_values("client_id", kind="stable")

    inv_summaries = _compute_invoice_summaries(
        inv_clean, inv_raw, set(merged["client_id"].astype(str))
    )

    records: list[dict] = []
    for row in merged.to_dict("records"):
        cid = str(row["client_id"])
        if cid not in inv_summaries:
            continue
        s = inv_summaries[cid]
        rec = {
            "client_id": cid,
            "client_catg": str(row["client_catg"]),
            "district": int(row["district"]),
            "region": str(row["region"]),
            "has_elec": int(row["has_elec"]),
            "has_gaz": int(row["has_gaz"]),
            "tenure_years": round(float(row["tenure_years"] or 0.0), 2),
            "n_invoices": int(row["n_invoices"]),
            "cons_pm_mean": round(float(row["cons_pm_mean"] or 0.0), 2),
            "cons_pm_median": round(float(row["cons_pm_median"] or 0.0), 2),
            "cons_pm_std": round(
                float(0.0 if pd.isna(row["cons_pm_std"]) else row["cons_pm_std"]), 2
            ),
            "cons_pm_min": float(0.0 if pd.isna(row["cons_pm_min"]) else row["cons_pm_min"]),
            "cons_pm_max": float(0.0 if pd.isna(row["cons_pm_max"]) else row["cons_pm_max"]),
            "share_level_2_plus": round(
                float(0.0 if pd.isna(row["share_level_2_plus"]) else row["share_level_2_plus"]),
                4,
            ),
            "cons_pm_mean_elec": (
                None if pd.isna(row["cons_pm_mean_elec"]) else float(row["cons_pm_mean_elec"])
            ),
            "cons_pm_mean_gaz": (
                None if pd.isna(row["cons_pm_mean_gaz"]) else float(row["cons_pm_mean_gaz"])
            ),
            "has_missing_months": int(row["has_missing_months"]),
            **s,
        }
        rec["matched_typologies"] = _match_typologies(rec)
        records.append(rec)

    chosen = _select_stratified_cases(records, n_cases=n_cases, seed=seed)
    cards: list[dict] = []
    for r in chosen:
        if r["client_id"] in test_ids or r["client_id"] not in train_ids:
            raise ValueError(f"Non-training client selected for case card: {r['client_id']}")
        cards.append(
            {
                "id": f"CASE-{r['client_id']}",
                "client_id": r["client_id"],
                "split": "train",
                "target": 1,
                "client_catg": r["client_catg"],
                "district": r["district"],
                "region": r["region"],
                "has_elec": r["has_elec"],
                "has_gaz": r["has_gaz"],
                "tenure_years": r["tenure_years"],
                "n_invoices": r["n_invoices"],
                "first_invoice": r["first_invoice"],
                "last_invoice": r["last_invoice"],
                "consumption": {
                    "cons_pm_mean": r["cons_pm_mean"],
                    "cons_pm_median": r["cons_pm_median"],
                    "cons_pm_std": r["cons_pm_std"],
                    "share_level_2_plus": r["share_level_2_plus"],
                    "trajectory_summary": r["trajectory_summary"],
                },
                "data_quality": {
                    "n_index_backwards": r["n_index_backwards"],
                    "n_index_rollover": r["n_index_rollover"],
                    "n_recon_mismatch": r["n_recon_mismatch"],
                    "n_bad_months_number": r["n_bad_months_number"],
                    "has_missing_months": r["has_missing_months"],
                },
                "codes_seen": r["codes_seen"],
                "matched_typologies": r["matched_typologies"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cards),
        encoding="utf-8",
    )
    return cards


def main():
    parser = argparse.ArgumentParser(
        description="Build 300 precedent case cards from training fraud clients (observed data only)."
    )
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/corpus/case_cards.jsonl"))
    parser.add_argument("--n-cases", type=int, default=N_CASES)
    args = parser.parse_args()
    try:
        cards = build_case_cards(
            raw_clients_path=args.clients,
            split_dir=args.splits,
            interim_dir=args.interim,
            processed_dir=args.processed,
            out_path=args.out,
            n_cases=args.n_cases,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"case_cards: {exc}\n")
    print(f"Wrote {len(cards):,} case cards to {args.out}")


if __name__ == "__main__":
    main()
