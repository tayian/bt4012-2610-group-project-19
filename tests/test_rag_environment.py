# Checks that the RAG libraries work here on the project's playbook.
# The embedding test downloads MiniLM on first run and is skipped if it cannot load.

import re
import unittest
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

PLAYBOOK = Path(__file__).resolve().parents[1] / "data/corpus/source_docs/playbook.md"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def playbook_chunks():
    # One chunk per typology section: "### [TYP-01] Title" up to the next heading.
    text = PLAYBOOK.read_text(encoding="utf-8")
    parts = re.split(r"^###\s+\[(TYP-\d{2})\]([^\n]*)\n", text, flags=re.M)
    chunks = {}
    for i in range(1, len(parts) - 2, 3):
        body = re.split(r"^#{1,3}\s", parts[i + 2], flags=re.M)[0]
        chunks[parts[i]] = parts[i + 1] + "\n" + body
    return chunks


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def rrf(rankings, k=60):
    # Reciprocal rank fusion over several ranked lists of chunk IDs.
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class RagEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = playbook_chunks()
        cls.ids = list(cls.chunks)
        cls.bm25 = BM25Okapi([tokenize(t) for t in cls.chunks.values()])

    def bm25_ranking(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        return [self.ids[i] for i in np.argsort(-scores)]

    def test_playbook_has_twelve_typologies(self):
        self.assertEqual(self.ids, [f"TYP-{i:02d}" for i in range(1, 13)])

    def test_bm25_finds_exact_terms(self):
        self.assertEqual(self.bm25_ranking("meter bypass shunt")[0], "TYP-01")
        self.assertEqual(self.bm25_ranking("magnet slowdown")[0], "TYP-02")

    def test_rrf_combines_rankings(self):
        fused = rrf([["TYP-01", "TYP-02", "TYP-03"], ["TYP-02", "TYP-01", "TYP-04"]])
        self.assertEqual(set(fused[:2]), {"TYP-01", "TYP-02"})
        self.assertEqual(fused[-1], "TYP-04")

    def test_dense_and_hybrid_retrieval(self):
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL)
        except Exception as exc:  # offline or model unavailable
            self.skipTest(f"embedding model unavailable: {exc}")
        vectors = model.encode(list(self.chunks.values()), normalize_embeddings=True)
        query = "someone attached a strong magnet to slow down the electricity meter"
        q = model.encode(query, normalize_embeddings=True)
        dense = [self.ids[i] for i in np.argsort(-(vectors @ q))]
        self.assertIn("TYP-02", dense[:3])
        hybrid = rrf([dense[:20], self.bm25_ranking(query)[:20]])
        self.assertIn("TYP-02", hybrid[:3])


if __name__ == "__main__":
    unittest.main()
