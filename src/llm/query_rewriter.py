# Query rewriter: turns a client risk snapshot into 3 short search queries for RAG retrieval.
#
# Uses the Section 14.4 system prompt and Gemini structured JSON output (model: GEMINI_MODEL,
# default gemini-2.5-flash).

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.llm.retrieval import client_filters
from src.llm.snapshot import SNAPSHOT_CONFIG, get_client_snapshot

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Project standard: GEMINI_MODEL from .env (see .env.example), else gemini-2.5-flash.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SAMPLE_SEED = 4012

QUERY_REWRITER_SYSTEM_PROMPT = (
    "You turn a utility-fraud risk snapshot into exactly 3 short search queries. "
    "One query per distinct suspicious pattern. Use plain words a policy manual "
    "or playbook would use. If a meter status or remark code appears, include "
    "the exact code in one query. Return a JSON list of 3 strings, nothing else."
)

DEFAULT_OUTPUT_PATH = Path("results/tables/query_rewriter_eval.json")


def format_snapshot_for_rewriter(snapshot: dict[str, Any]) -> str:
    # Serialize the relevant fields of a client snapshot for the query rewriter LLM.
    compact = {
        "client_id": snapshot.get("client_id"),
        "alert_date": snapshot.get("alert_date"),
        "risk_score": snapshot.get("risk_score"),
        "risk_rank_pct": snapshot.get("risk_rank_pct"),
        "profile": snapshot.get("profile", {}),
        "shap_top5": snapshot.get("shap_top5", []),
        "key_features": snapshot.get("key_features", {}),
        "benford_mad": snapshot.get("benford_mad"),
        "round_reading_share": snapshot.get("round_reading_share"),
        "invalid_reading_counts": snapshot.get("invalid_reading_counts", {}),
        "codes_seen": snapshot.get("codes_seen", {}),
    }
    return json.dumps(compact, indent=2, ensure_ascii=False)


def fallback_queries_from_snapshot(snapshot: dict[str, Any]) -> list[str]:
    # Deterministic rule-based fallback returning 3 search queries from a client snapshot.
    codes_seen = snapshot.get("codes_seen", {}) or {}
    statues = [str(c) for c in codes_seen.get("counter_statue", []) if str(c) != "0"]
    remarques = [str(c) for c in codes_seen.get("reading_remarque", []) if str(c) != "6"]
    profile = snapshot.get("profile", {}) or {}

    # Query 1: exact meter status or reading remark code query
    if statues and remarques:
        q1 = f"meter status counter_statue {statues[0]} and reading_remarque {remarques[0]} inspection protocol"
    elif statues:
        q1 = f"meter status code counter_statue {statues[0]} irregular consumption billing"
    elif remarques:
        q1 = f"meter reading remark code reading_remarque {remarques[0]} estimated or anomalous reading"
    else:
        catg = profile.get("client_catg", 11)
        q1 = f"client_catg {catg} counter_statue 0 reading_remarque 6 tariff classification check"

    # Query 2: consumption / SHAP pattern query
    shap_top5 = snapshot.get("shap_top5", []) or []
    top_features = [str(item.get("feature", "")) for item in shap_top5]
    if any("gaz" in f for f in top_features) or (
        profile.get("has_elec") and profile.get("has_gaz")
    ):
        q2 = "electricity and gas multi-counter consumption discrepancy meter bypass"
    elif any("max" in f or "std" in f or "skew" in f for f in top_features):
        q2 = "sudden consumption spike or high consumption variability catch-up billing"
    else:
        q2 = "abrupt consumption drop meter bypass shunt or magnet slowdown"

    # Query 3: forensic / data-quality / billing interval pattern
    benford = float(snapshot.get("benford_mad") or 0.0)
    round_share = float(snapshot.get("round_reading_share") or 0.0)
    # n_index_backwards already excludes exact meter rollovers.
    n_backwards = int(snapshot.get("n_index_backwards") or 0)
    n_recon = int(snapshot.get("n_recon_mismatch") or 0)
    has_missing = int(snapshot.get("has_missing_months") or 0)

    if n_backwards > 0 or n_recon > 0:
        q3 = "backwards meter index reading reconciliation gap index rollback tampering"
    elif benford > 0.015 or round_share > 0.40:
        q3 = "Benford MAD non-conformity round reading share meter reader collusion estimated reading"
    elif has_missing > 0:
        q3 = "missing billing months long gap between invoices defective or jammed meter"
    else:
        q3 = "mandatory on-site physical meter inspection seal verification calibration test"

    return [q1, q2, q3]


def _parse_three_queries(raw_text: str) -> list[str]:
    parsed = json.loads(raw_text)
    if isinstance(parsed, dict):
        for key in ("queries", "search_queries", "items"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected JSON list of strings from query rewriter, got: {type(parsed).__name__}"
        )
    cleaned = [str(q).strip() for q in parsed if str(q).strip()]
    if len(cleaned) != 3:
        raise ValueError(
            f"Expected exactly 3 queries from query rewriter, got {len(cleaned)}: {cleaned}"
        )
    return cleaned


def rewrite_snapshot_to_queries(
    snapshot: dict[str, Any],
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    allow_fallback: bool = False,
    max_retries: int = 4,
) -> list[str]:
    # Turn a client risk snapshot into 3 short search queries using Gemini.
    import time

    if client is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            if allow_fallback:
                return fallback_queries_from_snapshot(snapshot)
            raise RuntimeError("GEMINI_API_KEY is not set in environment or .env file.")

        from google import genai

        client = genai.Client(api_key=api_key)

    prompt = format_snapshot_for_rewriter(snapshot)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": QUERY_REWRITER_SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": list[str],
                    "temperature": 0.0,
                },
            )
            return _parse_three_queries(response.text)
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if attempt + 1 < max_retries and ("503" in msg or "UNAVAILABLE" in msg or "429" in msg):
                time.sleep(2.0 * (attempt + 1))
                continue
            break

    if allow_fallback:
        return fallback_queries_from_snapshot(snapshot)
    assert last_err is not None
    raise last_err


def select_sample_clients(n_clients: int = 5, seed: int = SAMPLE_SEED) -> list[str]:
    # Deterministic sample of training clients with OOF scores: about 3/5 fraud and 2/5
    # non-fraud, each class split into equal score bands with one client drawn per band, so
    # the sample covers high and low ranks rather than only the top of the list.
    import numpy as np
    import pandas as pd

    oof = pd.read_parquet(Path(SNAPSHOT_CONFIG["oof_path"]), columns=["client_id", "score"])
    labels = pd.read_parquet(
        Path(SNAPSHOT_CONFIG["interim_client_path"]), columns=["client_id", "target"]
    )
    df = oof.merge(labels, on="client_id", how="inner", validate="one_to_one")
    df = df.dropna(subset=["target"]).sort_values(["score", "client_id"], kind="stable")
    rng = np.random.default_rng(seed)
    n_fraud = max(1, round(0.6 * n_clients))
    picks: list[str] = []
    for target, n in ((1, n_fraud), (0, n_clients - n_fraud)):
        ids = df.loc[df["target"] == target, "client_id"].astype(str).to_numpy()
        for band in np.array_split(ids, n) if n > 0 else []:
            picks.append(str(rng.choice(band)))
    return picks


def evaluate_rewriter_on_clients(
    client_ids: list[str] | None = None,
    n_clients: int = 5,
    model: str = DEFAULT_MODEL,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
    client: Any | None = None,
    allow_fallback: bool = False,
) -> list[dict[str, Any]]:
    # Run the query rewriter on 5 clients and verify downstream retrieval on each query.
    from src.llm.retrieval import get_retriever

    selected_ids = (
        client_ids if client_ids is not None else select_sample_clients(n_clients=n_clients)
    )
    retriever = get_retriever()

    results: list[dict[str, Any]] = []
    for cid in selected_ids:
        snapshot = get_client_snapshot(cid)
        queries = rewrite_snapshot_to_queries(
            snapshot,
            model=model,
            client=client,
            allow_fallback=allow_fallback,
        )
        filters = client_filters(snapshot)
        retrieved_per_query = []
        for q in queries:
            top_chunks = retriever.retrieve(
                query=q,
                filters=filters,
                top_k=3,
                method="hybrid_rerank",
            )
            retrieved_per_query.append(
                {
                    "query": q,
                    "top3_chunk_ids": [c["id"] for c in top_chunks],
                }
            )

        results.append(
            {
                "client_id": cid,
                "model": model,
                "risk_score": snapshot.get("risk_score"),
                "risk_rank_pct": snapshot.get("risk_rank_pct"),
                "codes_seen": snapshot.get("codes_seen"),
                "shap_top5_features": [s["feature"] for s in snapshot.get("shap_top5", [])],
                "queries": queries,
                "retrieval_preview": retrieved_per_query,
            }
        )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return results


def main() -> None:
    results = evaluate_rewriter_on_clients(n_clients=5, model=DEFAULT_MODEL)
    print(f"Tested query rewriter ({DEFAULT_MODEL}) on {len(results)} clients:")
    for item in results:
        print(
            f"\nClient: {item['client_id']} "
            f"(risk_score={item['risk_score']}, rank_pct={item['risk_rank_pct']}, "
            f"codes={item['codes_seen']})"
        )
        for idx, q_info in enumerate(item["retrieval_preview"], start=1):
            print(f"  Q{idx}: {q_info['query']}")
            print(f"      -> Top-3 chunks: {q_info['top3_chunk_ids']}")


if __name__ == "__main__":
    main()
