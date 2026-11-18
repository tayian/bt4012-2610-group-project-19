# Tests for src/llm/query_rewriter.py, which turns a snapshot into three search queries.

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.llm.query_rewriter import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PATH,
    QUERY_REWRITER_SYSTEM_PROMPT,
    fallback_queries_from_snapshot,
    format_snapshot_for_rewriter,
    rewrite_snapshot_to_queries,
    select_sample_clients,
)
from src.llm.snapshot import get_client_snapshot


def test_select_sample_clients_and_formatting():
    clients = select_sample_clients(n_clients=5)
    assert len(clients) == 5
    assert len(set(clients)) == 5
    assert clients == select_sample_clients(n_clients=5)

    # 3 fraud + 2 non-fraud, labels joined from the client table.
    labels = pd.read_parquet("data/interim/client.parquet", columns=["client_id", "target"])
    targets = labels.set_index("client_id").loc[clients, "target"].tolist()
    assert sorted(targets) == [0, 0, 1, 1, 1]

    snap = get_client_snapshot(clients[0])
    formatted = format_snapshot_for_rewriter(snap)
    parsed = json.loads(formatted)
    assert parsed["client_id"] == clients[0]
    assert "shap_top5" in parsed
    assert "codes_seen" in parsed

    fb = fallback_queries_from_snapshot(snap)
    assert isinstance(fb, list)
    assert len(fb) == 3
    assert all(isinstance(q, str) and len(q.strip()) > 0 for q in fb)


def test_rewrite_snapshot_with_mock_client():
    called = {}

    class DummyModels:
        def generate_content(self, *, model, contents, config):
            called["model"] = model
            called["contents"] = contents
            called["config"] = config
            return SimpleNamespace(
                text=json.dumps(
                    [
                        "counter_statue 1 illegal reconnection consumption while disconnected",
                        "reading_remarque 9 anomalous meter reading remark code",
                        "Benford MAD non-conformity round reading share meter reader collusion",
                    ]
                )
            )

    dummy_client = SimpleNamespace(models=DummyModels())
    snap = get_client_snapshot("train_Client_0")
    queries = rewrite_snapshot_to_queries(snap, model=DEFAULT_MODEL, client=dummy_client)

    assert len(queries) == 3
    assert called["model"] == DEFAULT_MODEL
    assert called["config"]["system_instruction"] == QUERY_REWRITER_SYSTEM_PROMPT
    assert called["config"]["response_mime_type"] == "application/json"
    assert called["config"]["temperature"] == 0.0


def test_five_client_rewriter_eval_artifact():
    assert Path(DEFAULT_OUTPUT_PATH).is_file(), (
        f"Expected 5-client query rewriter evaluation output at {DEFAULT_OUTPUT_PATH}"
    )
    records = json.loads(Path(DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))
    assert len(records) == 5

    for item in records:
        assert "client_id" in item
        # The saved run records whichever model id produced it.
        assert isinstance(item.get("model"), str) and item["model"]
        queries = item.get("queries", [])
        assert isinstance(queries, list) and len(queries) == 3
        assert all(isinstance(q, str) and len(q.strip()) > 0 for q in queries)
        preview = item.get("retrieval_preview", [])
        assert len(preview) == 3
        for p in preview:
            assert len(p.get("top3_chunk_ids", [])) == 3
