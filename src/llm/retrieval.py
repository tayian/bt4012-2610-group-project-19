# Hybrid RAG retrieval engine for STEG Fraud Analytics.
#
# Pipeline:
#   1. Metadata hard-filtering (jurisdiction, counter_type, temporal [valid_from, valid_to),
#      doc/section/field/value/region/client_catg, and train-only split for precedent cases).
#   2. Sparse retrieval: BM25Okapi top-20 over tokenized filtered chunks.
#   3. Dense retrieval: sentence-transformers/all-MiniLM-L6-v2 cosine similarity top-20.
#   4. Reciprocal Rank Fusion (RRF, k=60) combining dense top-20 and BM25 top-20 -> top-5.
#   5. Optional 4th retrieval mode (hybrid_rerank): Cross-Encoder
#      (cross-encoder/ms-marco-MiniLM-L-6-v2) reranking on the top-20 RRF candidates -> top-5.

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/corpus/chunks.jsonl")
STUB_CHUNKS_PATH = Path("data/stubs/data/corpus/chunks.jsonl")

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
DEFAULT_CANDIDATE_K = 20
DEFAULT_TOP_K = 5

WORD_RE = re.compile(r"[a-z0-9_]+")
COMPOUND_RE = re.compile(r"[a-z0-9_]+(?:[-.][a-z0-9_]+)+")
CODE_PAIR_RE = re.compile(
    r"\b(counter_statue|reading_remarque|tarif_type|counter_code|client_catg)"
    r"(?:\s*(?:==|=|-)\s*|\s+)([a-z0-9]+)\b"
)
SOP_CLAUSE_RE = re.compile(r"\bsop-v\d{4}-(\d+\.\d+)\b")


def tokenize(text: str) -> list[str]:
    # Extract alphanumeric words, compound IDs, normalized field=value pairs, and clause numbers.
    lowered = text.lower()
    words = WORD_RE.findall(lowered)
    compounds = COMPOUND_RE.findall(lowered)
    code_pairs = [f"{f}={v}" for f, v in CODE_PAIR_RE.findall(lowered)]
    clauses = SOP_CLAUSE_RE.findall(lowered)
    return words + compounds + code_pairs + clauses


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    # Reciprocal Rank Fusion (RRF) over multiple ranked chunk ID lists.
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in first_seen:
                first_seen[cid] = len(first_seen)
    return sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid]))


def rrf_scores(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def load_chunks(path: Path | str | None = None) -> list[dict]:
    # Load corpus chunks from data/corpus/chunks.jsonl (falling back to stub chunks if missing).
    target = Path(path) if path is not None else CHUNKS_PATH
    if not target.is_file() and path is None and STUB_CHUNKS_PATH.is_file():
        target = STUB_CHUNKS_PATH
    if not target.is_file():
        raise FileNotFoundError(f"Corpus chunks file not found: {target}")
    return [
        json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _parse_date(val: str | dt.date | None) -> dt.date | None:
    if val is None or val == "":
        return None
    if isinstance(val, dt.date):
        return val
    return dt.date.fromisoformat(str(val)[:10])


def _is_valid_on(chunk: dict, alert_date: dt.date) -> bool:
    vf = _parse_date(chunk.get("valid_from")) or dt.date(1900, 1, 1)
    if alert_date < vf:
        return False
    vt = _parse_date(chunk.get("valid_to"))
    if vt is None:
        return True
    # Half-open window [valid_from, valid_to) when valid_to is Jan 1 (e.g. 2025-01-01 in sop_v2023),
    # or inclusive when valid_to is Dec 31 (e.g. 2024-12-31 in fixtures/stubs).
    if vt.month == 1 and vt.day == 1:
        return alert_date < vt
    return alert_date <= vt


def filter_chunks(chunks: list[dict], filters: dict[str, Any] | None = None) -> list[dict]:
    # Hard pre-retrieval metadata filter.
    f = filters or {}
    jurisdiction = f.get("jurisdiction", "TN")
    counter_type = f.get("counter_type")
    alert_raw = f.get("valid_on") or f.get("alert_date")
    alert_date = _parse_date(alert_raw) if alert_raw else None

    doc_filter = f.get("doc")
    if isinstance(doc_filter, str):
        doc_set = {doc_filter}
    elif doc_filter is not None:
        doc_set = set(doc_filter)
    else:
        doc_set = None

    section_filter = f.get("section")
    if isinstance(section_filter, str):
        section_set = {section_filter}
    elif section_filter is not None:
        section_set = set(section_filter)
    else:
        section_set = None

    version_filter = f.get("version")
    field_filter = f.get("field")
    value_filter = f.get("value")
    catg_filter = f.get("client_catg")
    region_filter = f.get("region")
    district_filter = f.get("district")
    exclude_client = f.get("exclude_client_id")

    out: list[dict] = []
    for c in chunks:
        # 1. Jurisdiction check (default TN).
        if jurisdiction and c.get("jurisdiction", "TN") != jurisdiction:
            continue

        # 2. Non-leakage check on precedent case cards: strictly train split only.
        if c.get("doc") == "case" and c.get("split", "train") != "train":
            continue

        # 3. Exclude the target client itself from precedent cases.
        if exclude_client and (
            c.get("client_id") == exclude_client or c.get("id") == f"CASE-{exclude_client}"
        ):
            continue

        # 4. Counter type compatibility: 'ELEC' matches ('ELEC', 'both'); 'GAZ' matches ('GAZ', 'both').
        if counter_type in ("ELEC", "GAZ"):
            c_type = c.get("counter_type", "both")
            if c_type not in (counter_type, "both"):
                continue

        # 5. Temporal validity window [valid_from, valid_to).
        if alert_date is not None and not _is_valid_on(c, alert_date):
            continue

        # 6. Document / section / version filters.
        if doc_set is not None and c.get("doc") not in doc_set:
            continue
        if section_set is not None and c.get("section") not in section_set:
            continue
        if version_filter is not None and c.get("version") != version_filter:
            continue

        # 7. Codebook field / value filters.
        if field_filter is not None and c.get("field") != field_filter:
            continue
        if value_filter is not None and str(c.get("value")) != str(value_filter):
            continue

        # 8. Case card demographic filters (applied when present on the chunk).
        if catg_filter is not None and c.get("client_catg") is not None:
            if str(c.get("client_catg")) != str(catg_filter):
                continue
        if region_filter is not None and c.get("region") is not None:
            if str(c.get("region")) != str(region_filter):
                continue
        if district_filter is not None and c.get("district") is not None:
            if str(c.get("district")) != str(district_filter):
                continue

        out.append(c)

    return out


class HybridRetriever:
    """In-memory hybrid retriever over chunks.jsonl with cached dense embeddings and lazy reranker."""

    def __init__(
        self,
        chunks: list[dict] | None = None,
        chunks_path: Path | str | None = None,
        dense_model_name: str = DENSE_MODEL_NAME,
        rerank_model_name: str = RERANK_MODEL_NAME,
    ):
        self.chunks = list(chunks) if chunks is not None else load_chunks(chunks_path)
        self.id_to_idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.id_to_chunk = {c["id"]: c for c in self.chunks}
        # Weight the chunk heading line (3x) so defining chunks rank above passing cross-references.
        self.tokenized = []
        for c in self.chunks:
            txt = c.get("text", "")
            first_line = txt.splitlines()[0] if txt else ""
            self.tokenized.append(tokenize(first_line) * 2 + tokenize(txt))

        self.dense_model_name = dense_model_name
        self.rerank_model_name = rerank_model_name
        self._dense_model = None
        self._embeddings: np.ndarray | None = None
        self._rerank_model = None

    def _ensure_dense(self) -> tuple[Any, np.ndarray]:
        if self._dense_model is None or self._embeddings is None:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer(self.dense_model_name)
            texts = [c.get("text", "") for c in self.chunks]
            self._embeddings = np.asarray(
                self._dense_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        return self._dense_model, self._embeddings

    def _ensure_reranker(self) -> Any:
        if self._rerank_model is None:
            from sentence_transformers import CrossEncoder

            self._rerank_model = CrossEncoder(self.rerank_model_name)
        return self._rerank_model

    def _bm25_rank(
        self, query: str, filtered: list[dict], top_n: int
    ) -> tuple[list[str], dict[str, float]]:
        if not filtered:
            return [], {}
        q_tokens = tokenize(query)
        corpus_tokens = [self.tokenized[self.id_to_idx[c["id"]]] for c in filtered]
        bm25 = BM25Okapi(corpus_tokens)
        # Use Lucene-style non-negative IDF: log(1 + (N - df + 0.5)/(df + 0.5)) = log1p(exp(idf))
        # so terms appearing in >50% of chunks (e.g. across the 300 case cards) stay positive.
        bm25.idf = {word: math.log1p(math.exp(idf)) for word, idf in bm25.idf.items()}
        raw_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
        order = np.argsort(-raw_scores, kind="stable")[:top_n]
        ranked_ids = [filtered[i]["id"] for i in order]
        score_map = {filtered[i]["id"]: float(raw_scores[i]) for i in order}
        return ranked_ids, score_map

    def _dense_rank(
        self, query: str, filtered: list[dict], top_n: int
    ) -> tuple[list[str], dict[str, float]]:
        if not filtered:
            return [], {}
        model, emb = self._ensure_dense()
        q_vec = np.asarray(
            model.encode(query, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        idxs = [self.id_to_idx[c["id"]] for c in filtered]
        sub_emb = emb[idxs]
        sims = sub_emb @ q_vec
        order = np.argsort(-sims, kind="stable")[:top_n]
        ranked_ids = [filtered[i]["id"] for i in order]
        score_map = {filtered[i]["id"]: float(sims[i]) for i in order}
        return ranked_ids, score_map

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        method: str = "hybrid",
        rrf_k: int = RRF_K,
    ) -> list[dict]:
        norm_method = method.strip().lower().replace(" + ", "_").replace("+", "_").replace(" ", "_")
        if norm_method == "rerank":
            norm_method = "hybrid_rerank"
        if norm_method not in ("bm25", "dense", "hybrid", "hybrid_rerank"):
            raise ValueError(
                f"Unknown retrieval method {method!r}; expected one of "
                "('bm25', 'dense', 'hybrid', 'hybrid_rerank')."
            )

        filtered = filter_chunks(self.chunks, filters)
        if not filtered or top_k <= 0:
            return []

        cand_n = max(top_k, candidate_k)

        if norm_method == "bm25":
            ranked_ids, score_map = self._bm25_rank(query, filtered, top_k)
            return self._format_results(ranked_ids[:top_k], score_map, "bm25")

        if norm_method == "dense":
            ranked_ids, score_map = self._dense_rank(query, filtered, top_k)
            return self._format_results(ranked_ids[:top_k], score_map, "dense")

        # Hybrid: dense top-20 + BM25 top-20 -> RRF (k=60).
        dense_ids, _ = self._dense_rank(query, filtered, cand_n)
        bm25_ids, _ = self._bm25_rank(query, filtered, cand_n)
        fused_ids = rrf([dense_ids, bm25_ids], k=rrf_k)
        fused_scores = rrf_scores([dense_ids, bm25_ids], k=rrf_k)

        if norm_method == "hybrid":
            return self._format_results(fused_ids[:top_k], fused_scores, "hybrid")

        # hybrid_rerank: Cross-Encoder rerank on top-20 RRF candidates before keeping top-5.
        top_candidates = fused_ids[:cand_n]
        if not top_candidates:
            return []
        reranker = self._ensure_reranker()
        pairs = [(query, self.id_to_chunk[cid].get("text", "")) for cid in top_candidates]
        ce_scores = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=np.float64)
        order = np.argsort(-ce_scores, kind="stable")[:top_k]
        reranked_ids = [top_candidates[i] for i in order]
        ce_map = {top_candidates[i]: float(ce_scores[i]) for i in order}
        return self._format_results(reranked_ids, ce_map, "hybrid_rerank")

    def _format_results(
        self,
        chunk_ids: list[str],
        score_map: dict[str, float],
        method_label: str,
    ) -> list[dict]:
        results: list[dict] = []
        for rank, cid in enumerate(chunk_ids, start=1):
            base = dict(self.id_to_chunk[cid])
            base["score"] = round(float(score_map.get(cid, 0.0)), 6)
            base["rank"] = rank
            base["retrieval_method"] = method_label
            results.append(base)
        return results


_DEFAULT_RETRIEVER: HybridRetriever | None = None
_DEFAULT_RETRIEVER_PATH: Path | None = None


def get_retriever(chunks_path: Path | str | None = None) -> HybridRetriever:
    global _DEFAULT_RETRIEVER, _DEFAULT_RETRIEVER_PATH
    resolved = Path(chunks_path).resolve() if chunks_path is not None else CHUNKS_PATH.resolve()
    if _DEFAULT_RETRIEVER is None or _DEFAULT_RETRIEVER_PATH != resolved:
        _DEFAULT_RETRIEVER = HybridRetriever(chunks_path=chunks_path)
        _DEFAULT_RETRIEVER_PATH = resolved
    return _DEFAULT_RETRIEVER


def retrieve(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    method: str = "hybrid",
    chunks_path: Path | str | None = None,
) -> list[dict]:
    retriever = get_retriever(chunks_path=chunks_path)
    return retriever.retrieve(
        query=query,
        filters=filters,
        top_k=top_k,
        candidate_k=candidate_k,
        method=method,
    )


def get_client_snapshot(
    client_id: str,
    alert_date: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.llm.snapshot import get_client_snapshot as _get_client_snapshot

    return _get_client_snapshot(client_id=client_id, alert_date=alert_date, config=config)
