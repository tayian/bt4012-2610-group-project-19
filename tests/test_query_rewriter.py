# Unit tests for src/llm/query_rewriter.py (Task 8: snapshot -> 3 search queries).

import json
from pathlib import Path
from types import SimpleNamespace

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
        assert item.get("model") == "gemini-3.8-flash"
        queries = item.get("queries", [])
        assert isinstance(queries, list) and len(queries) == 3
        assert all(isinstance(q, str) and len(q.strip()) > 0 for q in queries)
        preview = item.get("retrieval_preview", [])
        assert len(preview) == 3
        for p in preview:
            assert len(p.get("top3_chunk_ids", [])) == 3
