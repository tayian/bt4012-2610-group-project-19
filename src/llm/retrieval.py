# Hybrid retrieval over data/corpus/chunks.jsonl: metadata filters first, then BM25 and dense
# (all-MiniLM-L6-v2) top 20 fused with reciprocal rank fusion, optionally reranked by a
# cross-encoder. client_filters builds the filters for one client.

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
    # Words, compound IDs, normalised field=value pairs and SOP clause numbers.
    lowered = text.lower()
    words = WORD_RE.findall(lowered)
    compounds = COMPOUND_RE.findall(lowered)
    code_pairs = [f"{f}={v}" for f, v in CODE_PAIR_RE.findall(lowered)]
    clauses = SOP_CLAUSE_RE.findall(lowered)
    return words + compounds + code_pairs + clauses


def rrf_scores(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    # Chunk IDs by fused score; the stable sort breaks ties by first appearance.
    scores = rrf_scores(rankings, k)
    return sorted(scores, key=lambda cid: -scores[cid])


def load_chunks(path: Path | str | None = None) -> list[dict]:
    # Falls back to the stub corpus when the real one has not been built.
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
    # valid_to is exclusive when it is 1 January (sop_v2023 ends 2025-01-01) and inclusive
    # otherwise (the fixtures end on 2024-12-31).
    if vt.month == 1 and vt.day == 1:
        return alert_date < vt
    return alert_date <= vt


def _as_set(value: Any) -> set | None:
    if value is None:
        return None
    return {value} if isinstance(value, str) else set(value)


def filter_chunks(chunks: list[dict], filters: dict[str, Any] | None = None) -> list[dict]:
    # Metadata filter applied before retrieval.
    f = filters or {}
    jurisdiction = f.get("jurisdiction", "TN")
    counter_type = f.get("counter_type")
    alert_raw = f.get("valid_on") or f.get("alert_date")
    alert_date = _parse_date(alert_raw) if alert_raw else None

    doc_set = _as_set(f.get("doc"))
    section_set = _as_set(f.get("section"))

    version_filter = f.get("version")
    field_filter = f.get("field")
    value_filter = f.get("value")
    catg_filter = f.get("client_catg")
    region_filter = f.get("region")
    district_filter = f.get("district")
    exclude_client = f.get("exclude_client_id")

    out: list[dict] = []
    for c in chunks:
        if jurisdiction and c.get("jurisdiction", "TN") != jurisdiction:
            continue

        # Precedent case cards must come from the training split (a card without a split is
        # dropped, not assumed to be train).
        if c.get("doc") == "case" and c.get("split") != "train":
            continue

        # Never the client's own case card.
        if exclude_client and (
            c.get("client_id") == exclude_client or c.get("id") == f"CASE-{exclude_client}"
        ):
            continue

        if counter_type in ("ELEC", "GAZ"):
            c_type = c.get("counter_type", "both")
            if c_type not in (counter_type, "both"):
                continue

        if alert_date is not None and not _is_valid_on(c, alert_date):
            continue

        if doc_set is not None and c.get("doc") not in doc_set:
            continue
        if section_set is not None and c.get("section") not in section_set:
            continue
        if version_filter is not None and c.get("version") != version_filter:
            continue

        if field_filter is not None and c.get("field") != field_filter:
            continue
        if value_filter is not None and str(c.get("value")) != str(value_filter):
            continue

        # Case card demographic filters apply only to chunks that carry the field.
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
    # In-memory retriever; the dense model, embeddings and reranker load on first use.

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
        # The heading line counts three times so defining chunks rank above cross-references.
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
        # Lucene-style IDF, log1p(exp(idf)), keeps terms found in over half the chunks (such as
        # words shared by the 300 case cards) positive.
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

        dense_ids, _ = self._dense_rank(query, filtered, cand_n)
        bm25_ids, _ = self._bm25_rank(query, filtered, cand_n)
        fused_ids = rrf([dense_ids, bm25_ids], k=rrf_k)
        fused_scores = rrf_scores([dense_ids, bm25_ids], k=rrf_k)

        if norm_method == "hybrid":
            return self._format_results(fused_ids[:top_k], fused_scores, "hybrid")

        # hybrid_rerank: the cross-encoder reorders the fused candidates.
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


def client_filters(snapshot: dict[str, Any], **extra: Any) -> dict[str, Any]:
    # Filters for retrieval about one client: its alert date and meter type, and never its
    # own precedent case card.
    filters = {
        "alert_date": snapshot.get("alert_date"),
        "counter_type": (snapshot.get("profile") or {}).get("counter_type"),
        "exclude_client_id": snapshot.get("client_id"),
    }
    filters.update(extra)
    return filters


def retrieve(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    method: str = "hybrid",
    chunks_path: Path | str | None = None,
    client_id: str | None = None,
) -> list[dict]:
    # With client_id set, that client's own case card is always excluded.
    if client_id is not None:
        filters = {**(filters or {}), "exclude_client_id": str(client_id)}
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
