# Unit tests for src/llm/retrieval.py (metadata filtering, BM25, MiniLM dense, RRF, rerank).

from src.llm.retrieval import (
    HybridRetriever,
    client_filters,
    filter_chunks,
    load_chunks,
    retrieve,
    rrf,
)


def test_load_and_filter_chunks_temporal_and_counter_type():
    chunks = load_chunks()
    assert len(chunks) == 420

    # Temporal filtering: 2024-06-01 should only include v2023 SOPs.
    sop_2024 = filter_chunks(chunks, {"doc": "sop", "valid_on": "2024-06-01"})
    assert len(sop_2024) == 13
    assert all(c["version"] == "v2023" for c in sop_2024)

    # Boundary date 2025-01-01: half-open window [2023-01-01, 2025-01-01) excludes v2023.
    sop_2025 = filter_chunks(chunks, {"doc": "sop", "valid_on": "2025-01-01"})
    assert len(sop_2025) == 13
    assert all(c["version"] == "v2025" for c in sop_2025)

    # Counter-type filtering: GAZ excludes ELEC-only chunks like TYP-07 and tarif_type 11.
    gaz_chunks = filter_chunks(chunks, {"counter_type": "GAZ"})
    gaz_ids = {c["id"] for c in gaz_chunks}
    assert "TYP-07" not in gaz_ids
    assert "CODE-tarif_type-11" not in gaz_ids
    assert "CODE-tarif_type-40" in gaz_ids
    assert "TYP-01" in gaz_ids

    # Exclude target client from precedent case cards.
    any_case = next(c for c in chunks if c["doc"] == "case")
    cid = any_case["client_id"]
    filtered_cases = filter_chunks(chunks, {"doc": "case", "exclude_client_id": cid})
    assert all(c["client_id"] != cid for c in filtered_cases)

    # A case chunk without a split is dropped, not treated as train.
    no_split = {k: v for k, v in any_case.items() if k != "split"}
    assert filter_chunks([no_split], {}) == []


def test_client_never_retrieves_own_case_card():
    chunks = load_chunks()
    case = next(c for c in chunks if c["doc"] == "case")
    cid = case["client_id"]
    # The client's own card text is the best possible match for a query about it.
    query = case["text"]
    snapshot = {"client_id": cid, "alert_date": "2025-06-01", "profile": {}}
    for method in ("bm25", "hybrid"):
        ids = [c["id"] for c in retrieve(query, client_id=cid, top_k=20, method=method)]
        assert f"CASE-{cid}" not in ids
        ids = [
            c["id"]
            for c in retrieve(query, filters=client_filters(snapshot), top_k=20, method=method)
        ]
        assert f"CASE-{cid}" not in ids
    # Without the exclusion the card is found, so the test above is not vacuous.
    ids = [c["id"] for c in retrieve(query, top_k=5, method="bm25")]
    assert f"CASE-{cid}" in ids


def test_bm25_and_rrf_offline():
    retriever = HybridRetriever()
    res = retriever.retrieve(
        "counter_statue = 5 suspected anomaly",
        filters={"valid_on": "2025-06-01"},
        method="bm25",
        top_k=5,
    )
    assert len(res) == 5
    top_ids = [c["id"] for c in res]
    assert "CODE-counter_statue-5" in top_ids[:2]
    assert "SOP-v2025-3.2" in top_ids[:2]

    fused = rrf([["TYP-01", "TYP-02"], ["TYP-02", "TYP-03"]], k=60)
    assert fused[0] in ("TYP-01", "TYP-02")


def test_all_four_retrieval_methods():
    q = "strong external magnet attached to slow down the meter disc without zeros"
    for method in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        res = retrieve(
            q,
            filters={"doc": ["playbook", "sop"], "valid_on": "2025-06-01"},
            top_k=5,
            method=method,
        )
        assert len(res) == 5
        ids = [c["id"] for c in res]
        assert "TYP-02" in ids[:3], f"TYP-02 missing from top 3 for method={method}: {ids}"
        assert all(not i.startswith("SOP-v2023") for i in ids)


def test_benchmark_queries_and_eval_table():
    from pathlib import Path

    import pandas as pd

    from src.llm.retrieval_eval import load_benchmark_queries

    queries = load_benchmark_queries()
    assert len(queries) == 25
    chunks = {c["id"] for c in load_chunks()}
    for q in queries:
        for gid in q["gold_ids"]:
            assert gid in chunks, f"Gold ID {gid} in {q['id']} not found in chunks.jsonl"

    eval_csv = Path("results/tables/retrieval_eval.csv")
    if eval_csv.is_file():
        df = pd.read_csv(eval_csv)
        assert list(df["method"]) == ["dense", "bm25", "hybrid", "hybrid + rerank"]
        for col in ("recall@3", "recall@5", "mrr"):
            assert col in df.columns
            assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all()
