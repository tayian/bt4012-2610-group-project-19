# Unit tests for data/corpus/case_cards.jsonl and data/corpus/chunks.jsonl.

import json
from pathlib import Path

from src.data.split import read_clients, read_splits
from src.llm.corpus.build_chunks import REQUIRED_FIELDS

CHUNKS_PATH = Path("data/corpus/chunks.jsonl")
CASES_PATH = Path("data/corpus/case_cards.jsonl")


def test_chunks_file_exists_and_counts():
    assert CHUNKS_PATH.is_file(), "data/corpus/chunks.jsonl is missing"
    assert CASES_PATH.is_file(), "data/corpus/case_cards.jsonl is missing"

    chunks = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(chunks) == 420

    by_doc = {}
    for c in chunks:
        by_doc[c["doc"]] = by_doc.get(c["doc"], 0) + 1

    assert by_doc == {
        "playbook": 12,
        "sop": 26,
        "codebook": 82,
        "case": 300,
    }


def test_required_fields_and_unique_ids():
    chunks = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = set()
    for c in chunks:
        for field in REQUIRED_FIELDS:
            assert field in c, f"Chunk {c.get('id')} missing {field}"
        assert c["id"] not in ids, f"Duplicate id {c['id']}"
        ids.add(c["id"])
        assert c["text"].startswith(f"[{c['id']}]")
        assert c["jurisdiction"] == "TN"
        assert c["counter_type"] in ("ELEC", "GAZ", "both")
        assert isinstance(c["synthetic"], bool)


def test_sop_versions_and_validity():
    chunks = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sop_chunks = [c for c in chunks if c["doc"] == "sop"]
    v2023 = [c for c in sop_chunks if c["version"] == "v2023"]
    v2025 = [c for c in sop_chunks if c["version"] == "v2025"]
    assert len(v2023) == 13
    assert len(v2025) == 13
    for c in v2023:
        assert c["valid_from"] == "2023-01-01"
        assert c["valid_to"] == "2025-01-01"
        assert c["synthetic"] is True
    for c in v2025:
        assert c["valid_from"] == "2025-01-01"
        assert c["valid_to"] is None
        assert c["synthetic"] is True


def test_case_cards_non_leakage_and_observed_only():
    clients = read_clients(Path("data/raw/client_train.csv"))
    train_df, test_df, _ = read_splits(Path("data/splits"), clients)
    train_ids = set(train_df["client_id"].astype(str))
    test_ids = set(test_df["client_id"].astype(str))
    fraud_map = dict(zip(clients["client_id"].astype(str), clients["target"].astype(int)))

    cards = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cards) == 300

    for card in cards:
        cid = card["client_id"]
        assert cid in train_ids, f"{cid} is not in train_ids.csv"
        assert cid not in test_ids, f"Leakage: {cid} is in test_ids.csv"
        assert fraud_map[cid] == 1, f"{cid} is not a confirmed fraud client (target != 1)"
        assert card["split"] == "train"
        assert card["target"] == 1
        assert card["n_invoices"] > 0
        # A card may match no rule; it then carries no typology instead of a guessed one.
        typs = card["matched_typologies"]
        assert len(typs) <= 3
        assert all(t in {f"TYP-{i:02d}" for i in range(1, 13)} for t in typs), typs
