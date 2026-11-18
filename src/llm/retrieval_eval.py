# Evaluates dense, bm25, hybrid and hybrid + rerank retrieval on the 25-query benchmark and
# writes results/tables/retrieval_eval.csv plus the gold chunk's rank per query and method.
# With 25 queries, one query moves overall recall by 0.04.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.llm.faithfulness import retrieval_metrics
from src.llm.retrieval import HybridRetriever

QUERIES_PATH = Path("data/corpus/retrieval_queries.json")
OUT_PATH = Path("results/tables/retrieval_eval.csv")
BY_QUERY_PATH = Path("results/tables/retrieval_eval_by_query.csv")

EVAL_METHODS = (
    ("dense", "dense"),
    ("bm25", "bm25"),
    ("hybrid", "hybrid"),
    ("hybrid + rerank", "hybrid_rerank"),
)


def load_benchmark_queries(path: Path = QUERIES_PATH) -> list[dict]:
    queries = json.loads(path.read_text(encoding="utf-8"))
    if len(queries) != 25:
        raise ValueError(f"Expected 25 benchmark queries, got {len(queries)}")
    by_cat: dict[str, int] = {}
    for q in queries:
        cat = q.get("category", "")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    expected = {"concept": 10, "exact_code": 10, "mixed": 5}
    if by_cat != expected:
        raise ValueError(f"Query category counts {by_cat} != expected {expected}")
    return queries


def evaluate_retrieval(
    queries_path: Path = QUERIES_PATH,
    chunks_path: Path | None = None,
    out_path: Path = OUT_PATH,
    by_query_path: Path | None = BY_QUERY_PATH,
    top_k: int = 5,
    candidate_k: int = 20,
) -> pd.DataFrame:
    queries = load_benchmark_queries(queries_path)
    retriever = HybridRetriever(chunks_path=chunks_path)

    gold_all = {q["id"]: set(q["gold_ids"]) for q in queries}
    gold_by_cat = {
        cat: {q["id"]: set(q["gold_ids"]) for q in queries if q["category"] == cat}
        for cat in ("concept", "exact_code", "mixed")
    }

    rows: list[dict] = []
    hits: list[dict] = []
    for display_name, method_key in EVAL_METHODS:
        ranked: dict[str, list[str]] = {}
        for q in queries:
            res = retriever.retrieve(
                query=q["query"],
                filters=q.get("filters"),
                top_k=top_k,
                candidate_k=candidate_k,
                method=method_key,
            )
            ranked[q["id"]] = [c["id"] for c in res]
            gold_ranks = [r for r, cid in enumerate(ranked[q["id"]], 1) if cid in gold_all[q["id"]]]
            hits.append(
                {
                    "query_id": q["id"],
                    "category": q["category"],
                    "method": display_name,
                    "gold_rank": min(gold_ranks) if gold_ranks else None,
                }
            )

        overall = retrieval_metrics(ranked, gold_all)
        row: dict[str, float | int | str] = {
            "method": display_name,
            "n_queries": int(overall["n_queries"]),
            "recall@3": round(float(overall["recall@3"]), 4),
            "recall@5": round(float(overall["recall@5"]), 4),
            "mrr": round(float(overall["mrr"]), 4),
        }
        for cat in ("concept", "exact_code", "mixed"):
            sub_m = retrieval_metrics(ranked, gold_by_cat[cat])
            row[f"{cat}_recall@3"] = round(float(sub_m["recall@3"]), 4)
            row[f"{cat}_recall@5"] = round(float(sub_m["recall@5"]), 4)
            row[f"{cat}_mrr"] = round(float(sub_m["mrr"]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if by_query_path is not None:
        by_query = pd.DataFrame(hits)
        by_query["gold_rank"] = by_query["gold_rank"].astype("Int64")
        by_query.to_csv(by_query_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dense, bm25, hybrid, and hybrid + rerank on the 25-query benchmark."
    )
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--by-query-out", type=Path, default=BY_QUERY_PATH)
    args = parser.parse_args()
    try:
        df = evaluate_retrieval(
            queries_path=args.queries,
            chunks_path=args.chunks,
            out_path=args.out,
            by_query_path=args.by_query_out,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"retrieval_eval: {exc}\n")
    print(df.to_string(index=False))
    print(f"Wrote {args.out} and {args.by_query_out}")


if __name__ == "__main__":
    main()
