# Regenerate the fake fixtures in data/stubs/ for interface tests.
#
# Only the 1,000 client IDs are real. Labels and raw invoices are random; everything after
# them goes through the real pipeline code: src/data/load.py reads the fake raw CSVs,
# clean_invoices cleans them, and features_v0 and features_v1 are built by
# src/features/{aggregate,behavioural}.py. The sequence tensor, splits, scores, SHAP values,
# corpus chunks and briefs are random with the agreed shapes and column names. Nothing here is
# a project result.

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import clean_invoices
from src.data.load import LEVELS, load_clients, load_invoices
from src.data.split import split_clients
from src.features import aggregate, behavioural

SEED = 4012
N_CLIENTS = 1000
N_POSITIVE = 56  # about the real 5.6% fraud rate
INVOICES_PER_CLIENT = 12
N_DUPLICATES = 10  # exact copies, so clean.py drops duplicates as it does on the real data
MODELS = [f"M{i}" for i in range(1, 12)]
VERSIONS = ["v0", "v1", "v2"]
FINAL_MODEL = "STUB-M10-v2"
CATEGORICAL = ["client_catg", "district", "region"]

# Planned v2 features; only the names are fixed so far.
PLANNED_V2_COLUMNS = {
    "yoy_drop_max": "float64",
    "round_reading_share": "float64",
    "benford_mad": "float64",
    "n_index_backwards": "int64",
}
# Fields that may be recorded during an inspection. features_v2_with_inspection includes them
# and features_v2 leaves them out, the layout the real files will use once they exist.
INSPECTION_COLUMNS = ["statue_share_0", "statue_share_1", "remarque_share_6", "n_statue_changes"]

FIXED_CHUNKS = [
    {
        "id": "TYP-01",
        "text": "STUB / SYNTHETIC: An example consumption pattern may prompt a meter check.",
        "doc": "typology",
        "section": "example",
        "version": "stub",
        "valid_from": "2000-01-01",
        "valid_to": None,
        "counter_type": "both",
        "jurisdiction": "TN",
    },
    {
        "id": "CODE-counter_statue-1",
        "text": "STUB / SYNTHETIC: Example code entry; meaning is not authoritative.",
        "doc": "codebook",
        "section": "example",
        "version": "stub",
        "valid_from": "2000-01-01",
        "valid_to": None,
        "counter_type": "both",
        "jurisdiction": "TN",
        "field": "counter_statue",
    },
    {
        "id": "SOP-v2023-2.1",
        "text": "STUB / SYNTHETIC: Example historical inspection clause.",
        "doc": "sop",
        "section": "2.1",
        "version": "v2023",
        "valid_from": "2023-01-01",
        "valid_to": "2024-12-31",
        "counter_type": "both",
        "jurisdiction": "TN",
    },
    {
        "id": "SOP-v2025-2.1",
        "text": "STUB / SYNTHETIC: Example current inspection clause.",
        "doc": "sop",
        "section": "2.1",
        "version": "v2025",
        "valid_from": "2025-01-01",
        "valid_to": None,
        "counter_type": "both",
        "jurisdiction": "TN",
    },
]
N_CASES = 10

README = """# Fake STEG fixtures (stubs)

Everything in this folder is fake except the 1,000 client IDs, which are taken from
data/raw/client_train.csv. Labels and raw invoices are random. The interim files,
features_v0 and features_v1 are built from those fake raw rows by the real load, clean
and feature code, so they have the real schemas. seq.npy is random with the agreed shape
(clients x 120 months x 3 channels). features_v2 (planned columns) and
features_v2_with_inspection (planned and inspection-time columns), splits, scores, SHAP
values, corpus chunks and briefs are random values with the real column names. Use them
only for interface tests, never as results. The split here is synthetic; the official split
is in data/splits/.

Regenerate with `python run.py stubs` (`python3 -m src.data.make_stubs`). It reads only the client
IDs of data/raw/client_train.csv. manifest.json lists every file.
"""

FAITHFULNESS = """# Interface double only: same call as the real scorer, with random scores
# that do not validate a brief. No LLM calls are made.
from random import Random


def score_brief(brief, snapshot, evidence) -> dict:
    rng = Random("STUB:" + str(brief.get("client_id", "")))
    return {
        "is_stub": True,
        "citation_validity": round(rng.random(), 4),
        "number_accuracy": round(rng.random(), 4),
        "rule_applicability": round(rng.random(), 4),
        "faithfulness": round(rng.uniform(1, 5), 4),
        "retrieval": None,
        "needs_human_review": True,
    }
"""


def random_column(rng, name, dtype, n):
    if name.startswith("has_"):
        return rng.integers(0, 2, n)
    if str(dtype) == "int64":
        return rng.integers(0, 20, n)
    return rng.gamma(2.0, 50.0, n)


def raw_clients(rng, ids):
    # Rows shaped like data/raw/client_train.csv.
    n = len(ids)
    target = np.zeros(n)
    target[rng.choice(n, N_POSITIVE, replace=False)] = 1
    created = pd.Timestamp("1980-01-01") + pd.to_timedelta(rng.integers(0, 38 * 365, n), unit="D")
    return pd.DataFrame(
        {
            "disrict": rng.choice([60, 62, 63, 69], n),
            "client_id": ids,
            "client_catg": rng.choice(["11", "12", "51"], n, p=[0.9, 0.05, 0.05]),
            "region": rng.choice(["101", "104", "301", "311"], n),
            "creation_date": created.strftime("%d/%m/%Y"),
            "target": target,
        }
    )


def raw_invoices(rng, ids):
    # Rows shaped like data/raw/invoice_train.csv, with the kinds of mess the real file has.
    n = len(ids) * INVOICES_PER_CLIENT
    owner = np.repeat(np.arange(len(ids)), INVOICES_PER_CLIENT)
    gaz = rng.random(n) < 0.3
    meter = rng.integers(1, 10**7, (len(ids), 2))
    dates = pd.Timestamp("2005-01-01") + pd.to_timedelta(rng.integers(0, 15 * 365, n), unit="D")
    levels = np.column_stack(
        [
            rng.integers(0, 600, n),
            rng.integers(0, 300, n) * (rng.random(n) < 0.4),
            rng.integers(0, 200, n) * (rng.random(n) < 0.1),
            rng.integers(0, 200, n) * (rng.random(n) < 0.05),
        ]
    )
    cons = levels.sum(axis=1)
    old_index = rng.integers(0, 100000, n)
    # Some readings do not reconcile with cons; a 5-digit meter passing 99999 restarts at 0.
    idx_diff = np.where(rng.random(n) < 0.05, cons + rng.integers(-50, 50, n), cons)
    invoices = pd.DataFrame(
        {
            "client_id": ids[owner],
            "invoice_date": dates.strftime("%Y-%m-%d"),
            "tarif_type": np.where(gaz, rng.choice([40, 45], n), rng.choice([10, 11], n)),
            "counter_number": meter[owner, gaz.astype(int)],
            # Padded and junk codes, as in the raw file.
            "counter_statue": rng.choice(
                ["0", "1", "5", " 0", "A"], n, p=[0.88, 0.05, 0.03, 0.02, 0.02]
            ),
            "counter_code": np.where(gaz, 5, rng.choice([203, 207, 413], n)),
            "reading_remarque": rng.choice(["6", "8", "9", "413"], n, p=[0.4, 0.3, 0.29, 0.01]),
            "counter_coefficient": np.where(rng.random(n) < 0.01, 0, 1),
            **{level: levels[:, i] for i, level in enumerate(LEVELS)},
            "old_index": old_index,
            "new_index": (old_index + idx_diff) % 100000,
            # A few impossible billing periods, so the cleaning flags are exercised.
            "months_number": rng.choice(
                [2, 4, 6, 12, 0, 300], n, p=[0.3, 0.5, 0.1, 0.08, 0.01, 0.01]
            ),
            "counter_type": np.where(gaz, "GAZ", rng.choice(["ELEC", "elec "], n, p=[0.98, 0.02])),
        }
    )
    invoices = invoices.sort_values(["client_id", "invoice_date"], kind="stable")
    copies = invoices.iloc[rng.choice(n, N_DUPLICATES, replace=False)]
    return pd.concat([invoices, copies], ignore_index=True)


def load_raw(clients, invoices):
    # Run the real CSV loader on the fake rows.
    with tempfile.TemporaryDirectory() as directory:
        paths = Path(directory) / "client_train.csv", Path(directory) / "invoice_train.csv"
        clients.to_csv(paths[0], index=False)
        invoices.to_csv(paths[1], index=False)
        return load_clients(paths[0]), load_invoices(paths[1])


def make_features(rng, invoices, clients):
    # v0 and v1 come from the real feature code; v2 adds random planned columns, and
    # v2_with_inspection also the inspection-time columns.
    v0 = aggregate.build_features(invoices, clients)
    v1 = behavioural.build_features(invoices, clients["client_id"])
    n = len(v1)
    v2 = v1.copy()
    for column, dtype in PLANNED_V2_COLUMNS.items():
        v2[column] = pd.Series(random_column(rng, column, dtype, n)).astype(dtype)
    inspection = v2.copy()
    share = rng.random(n)
    inspection["statue_share_0"] = share
    inspection["statue_share_1"] = 1 - share
    inspection["remarque_share_6"] = rng.random(n)
    inspection["n_statue_changes"] = rng.integers(0, 12, n).astype("int64")
    return {
        "features_v0": v0,
        "features_v1": v1,
        "features_v2": v2,
        "features_v2_with_inspection": inspection,
    }


def make_shap(rng, features):
    numeric = [c for c in features.columns if c not in ["client_id", *CATEGORICAL]]
    rows = []
    for record in features.to_dict("records"):
        chosen = rng.choice(numeric, 5, replace=False)
        shap = rng.normal(0, 0.3, 5)
        for rank, i in enumerate(np.argsort(-np.abs(shap)), start=1):
            rows.append(
                {
                    "client_id": str(record["client_id"]),
                    "rank": rank,
                    "feature": str(chosen[i]),
                    "value": float(record[chosen[i]]),
                    "shap_value": float(shap[i]),
                }
            )
    return pd.DataFrame(rows).astype({"client_id": object, "rank": "int64", "feature": object})


def make_brief(row, cases):
    tier = "low" if row.score < 0.4 else "medium" if row.score < 0.8 else "high"
    similar = next(c for c in cases if c != f"CASE-{row.client_id}")
    return {
        "client_id": row.client_id,
        "risk_score": row.score,
        "risk_rank_pct": row.rank_pct,
        "risk_tier": tier,
        "evidence": [
            {"point": f"Synthetic model score: {row.score}.", "citations": ["SNAPSHOT"]},
            {"point": "Synthetic example of a consumption pattern.", "citations": ["TYP-01"]},
            {
                "point": "Synthetic inspection clause for interface testing.",
                "citations": ["SOP-v2025-2.1"],
            },
        ],
        "onsite_checks": ["Synthetic example: verify the meter serial number."],
        "applicable_rules": [
            {"rule_id": "SOP-v2025-2.1", "action": "Synthetic example: check the meter."}
        ],
        "similar_cases": [similar],
        "uncertainty": "STUB: all values and evidence are synthetic; no fraud conclusion is supported.",
    }


def build_stubs(out, clients_csv):
    rng = np.random.default_rng(SEED)
    all_ids = pd.read_csv(clients_csv, usecols=["client_id"], dtype="string")["client_id"]
    ids = np.sort(all_ids.to_numpy(dtype=str)[rng.choice(len(all_ids), N_CLIENTS, replace=False)])
    written = []

    def save(name, write):
        # write(path) creates the file; the name is recorded for the manifest.
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path)
        written.append(name)

    def save_parquet(frame, name):
        save(name, lambda path: frame.to_parquet(path, index=False, engine="pyarrow"))

    def save_text(text, name):
        save(name, lambda path: path.write_text(text))

    clients, raw = load_raw(raw_clients(rng, ids), raw_invoices(rng, ids))
    invoices = clean_invoices(raw)
    save_parquet(clients, "data/interim/client.parquet")
    save_parquet(invoices, "data/interim/invoice_clean.parquet")

    features = make_features(rng, invoices, clients)
    for name, table in features.items():
        save_parquet(table, f"data/processed/{name}.parquet")

    # Its own generator, so the other stub files keep their random draws.
    seq = np.random.default_rng([SEED, 1]).random((len(clients), 120, 3), dtype=np.float32)
    save("data/processed/seq.npy", lambda path: np.save(path, seq))
    save(
        "data/processed/seq_client_ids.npy",
        lambda path: np.save(path, clients["client_id"].to_numpy(dtype=str)),
    )

    train, test, folds = split_clients(clients)
    for frame, name in ((train, "train_ids"), (test, "test_ids"), (folds, "folds")):
        save(f"data/splits/{name}.csv", lambda path: frame.to_csv(path, index=False))

    for model in MODELS:
        for version in VERSIONS:
            oof = pd.DataFrame(
                {
                    "client_id": folds["client_id"].astype("string"),
                    "fold": folds["fold"].astype("int64"),
                    "score": rng.random(len(folds)),
                    "model": model,
                    "feature_version": version,
                }
            )
            save_parquet(oof, f"results/oof/{model}_{version}.parquet")

    split = np.where(pd.Series(ids).isin(test["client_id"]), "test", "train")
    final = pd.DataFrame(
        {"client_id": pd.array(ids, dtype="string"), "split": split, "score": rng.random(len(ids))}
    )
    final["rank_pct"] = final.groupby("split")["score"].rank(pct=True)
    final["model_version"] = FINAL_MODEL
    final = final.astype({"split": object, "model_version": object})
    save_parquet(final, "results/final_scores.parquet")

    # The v2 SHAP rows draw feature names from the table with the inspection-time columns, so
    # those names appear in the fixtures too.
    shap_tables = {**features, "features_v2": features["features_v2_with_inspection"]}
    shap = {version: make_shap(rng, shap_tables[f"features_{version}"]) for version in VERSIONS}
    for version, table in shap.items():
        save_parquet(table, f"results/shap_top5_{version}.parquet")
    # The final model uses v2, so its SHAP files repeat the v2 table.
    save_parquet(shap["v2"], "results/shap_top5.parquet")
    save_parquet(shap["v2"], "results/shap_top5_final.parquet")

    profile = clients.set_index("client_id")
    case_ids = np.sort(rng.choice(train["client_id"].to_numpy(dtype=str), N_CASES, replace=False))
    chunks = list(FIXED_CHUNKS)
    for cid in case_ids:
        chunks.append(
            {
                "id": f"CASE-{cid}",
                "text": f"STUB / SYNTHETIC: Example case for {cid}; its label and facts are synthetic.",
                "doc": "case",
                "section": "example",
                "version": "stub",
                "valid_from": "2000-01-01",
                "valid_to": None,
                "counter_type": "both",
                "jurisdiction": "TN",
                "client_id": cid,
                "client_catg": profile.at[cid, "client_catg"],
                "region": profile.at[cid, "region"],
                "split": "train",
            }
        )
    save_text("".join(json.dumps(c) + "\n" for c in chunks), "data/corpus/chunks.jsonl")

    cases = [f"CASE-{cid}" for cid in case_ids]
    for row in final.itertuples(index=False):
        brief = make_brief(row, cases)
        save_text(json.dumps(brief, indent=2) + "\n", f"results/briefs/{row.client_id}.json")

    save_text("", "src/__init__.py")
    save_text("", "src/llm/__init__.py")
    save_text(FAITHFULNESS, "src/llm/faithfulness.py")
    save_text(README, "README.md")

    files = sorted([*written, "manifest.json"])
    manifest = {
        "generator": "src.data.make_stubs",
        "is_stub": True,
        "seed": SEED,
        "source": "data/raw/client_train.csv (client IDs only)",
        "built_by": [
            "src.data.load",
            "src.data.clean.clean_invoices",
            "src.features.aggregate",
            "src.features.behavioural",
        ],
        "n_clients": N_CLIENTS,
        "n_train": len(train),
        "n_test": len(test),
        "versions": [int(v[1:]) for v in VERSIONS],
        "models": MODELS,
        "split_source": "synthetic; never use as the official evaluation split",
        "planned_v2_columns": list(PLANNED_V2_COLUMNS),
        "inspection_columns": list(INSPECTION_COLUMNS),
        "files": files,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the fake fixtures in data/stubs/ from 1,000 real client IDs "
        "and the real pipeline code."
    )
    parser.add_argument("--out", type=Path, default=Path("data/stubs"))
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    args = parser.parse_args()
    try:
        old_manifest = args.out / "manifest.json"
        old_files = json.loads(old_manifest.read_text())["files"] if old_manifest.exists() else []
        manifest = build_stubs(args.out, args.clients)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"make_stubs: {exc}\n")

    # Remove stubs from an older layout that are no longer generated.
    stale = sorted(set(old_files) - set(manifest["files"]))
    for name in stale:
        (args.out / name).unlink(missing_ok=True)
    total = sum((args.out / name).stat().st_size for name in manifest["files"])
    print(f"Wrote {len(manifest['files']):,} stub files ({total / 1e6:.1f} MB) to {args.out}")
    if stale:
        print(f"Removed {len(stale)} stale stub files: {', '.join(stale)}")


if __name__ == "__main__":
    main()
