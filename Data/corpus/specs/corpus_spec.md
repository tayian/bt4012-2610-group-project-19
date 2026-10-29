# STEG Fraud Analytics: Grounding Corpus Specification (`data/corpus/corpus_spec.md`)

**Project:** BT4012 Fraud Analytics — The Tunisian Company of Electricity and Gas (STEG)  
**Author / Workstream:** Role D (LLM, RAG & Agent Lead)  
**Status:** Approved Outline (Week 1 Milestone)  
**Last Updated:** October 2026  

---

## 1. Overview & Architectural Role

The non-parametric retrieval-augmented generation (RAG) system generates audit-ready, grounded **Inspector Briefs** for field teams inspecting customers flagged by the upstream ML fusion model. To ensure zero hallucinations, auditability, and legal defensibility, the LLM is restricted to citing verified documentary evidence.

This directory (`data/corpus/`) houses the complete knowledge corpus of **4 structured documents**, designed to test both dense semantic matching and exact keyword retrieval while enforcing strict metadata filtering:

```
+-----------------------------------------------------------------------------------------+
|                                    KNOWLEDGE CORPUS                                     |
+-----------------------------+-----------------------------+-----------------------------+
| Doc 1: Typology Playbook    | Doc 2: STEG Code Book       | Doc 3: Inspection SOPs      |
| (12 Physical Attack Vectors)| (Categorical Values & Mean) | (Synthetic v2023 / v2025)   |
| IDs: [TYP-01] to [TYP-12]   | IDs: [CODE-<field>-<value>] | IDs: [SOP-vYYYY-x.x]        |
| Chunking: 1 per typology    | Chunking: 1 per code state  | Chunking: 1 per clause      |
+-----------------------------+-----------------------------+-----------------------------+
                                              |
                              +---------------+---------------+
                              | Doc 4: Precedent Case Cards   |
                              | (300 Historical Fraud Dossiers|
                              | IDs: [CASE-<client_id>]       |
                              | Chunking: 1 per case card     |
                              +-------------------------------+
```

---

## 2. Master Table of Contents & Corpus Sizing

| Document ID & Title | Primary File Path | Target Chunks | Chunk ID Prefix | Target Word Count | Core Lecture Concept Demonstrated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Doc 1: NTL Typology Playbook** | `data/corpus/source_docs/playbook.md` | 12 chunks | `[TYP-01]` to `[TYP-12]` | 200–400 words / chunk | Structural chunking by domain concept; physical mechanism & data signature mapping. |
| **Doc 2: STEG Utility Code Book** | `data/corpus/source_docs/codebook.md` | ~25–30 chunks | `[CODE-<field>-<val>]` | 50–100 words / chunk | Exact alphanumeric code retrieval; shows where BM25 keyword matching outperforms dense semantic embeddings. |
| **Doc 3: Inspection SOPs (v2023 & v2025)** | `data/corpus/source_docs/sop_v2023.md`<br>`data/corpus/source_docs/sop_v2025.md` | 24–30 chunks (12–15 per ver) | `[SOP-v2023-x.x]`<br>`[SOP-v2025-x.x]` | 80–150 words / chunk | Temporal metadata filtering (`valid_from`, `valid_to`) preventing superseded policies from misdirecting current actions. |
| **Doc 4: Precedent Case Cards** | `data/corpus/case_cards.jsonl` | 300 cards | `[CASE-<client_id>]` | 100–180 words / card | Analogous precedent retrieval from training data only (strictly never from held-out test split). |

---

## 3. Metadata Standards & Filtering Schema

Every chunk in the consolidated index (`data/corpus/chunks.jsonl`) adheres to the following metadata contract:

```json
{
  "id": "TYP-01",
  "text": "Full text of the chunk...",
  "doc": "playbook",
  "section": "meter_bypass",
  "version": "v1.0",
  "valid_from": "2005-01-01",
  "valid_to": null,
  "counter_type": "both",
  "jurisdiction": "TN"
}
```

### Pre-Retrieval Filtering Rules
Before dense and sparse similarity scores are computed, candidates are hard-filtered against the client inspection context:
1. `jurisdiction == 'TN'` (Tunisia — drops cross-border synthetic test policies).
2. `counter_type in [client.counter_type, 'both']` (Prevents gas-only fraud mechanisms from being cited for an electricity-only meter).
3. `valid_from <= inspection_date < (valid_to or '9999-12-31')` (Enforces temporal validity; excludes retired SOP clauses).
4. `split == 'train'` (Strictly enforced on precedent cases to prevent test-set data leakage).

---

## 4. Chunking Strategy & Retrieval Optimization

1. **Chunk Boundaries:** Chunks are split by semantic structure (one complete section, clause, or case), never arbitrary character or token counts. This preserves complete physical definitions and prevents split rule criteria.
2. **Citation Identifiers:** Every chunk begins with its canonical identifier enclosed in square brackets (e.g. `[TYP-01]`, `[SOP-v2025-2.1]`, `[CODE-counter_statue-5]`). The LLM prompt sandwich instructs the model to cite these exact bracketed IDs.
3. **Hybrid Search Synergy:**
   * **Dense Embeddings (`all-MiniLM-L6-v2`):** Matches semantic descriptions (e.g. *"customer consumption dropped dramatically after 2014"* matches `TYP-01` Meter Bypass).
   * **Sparse BM25 (`BM25Okapi`):** Matches exact code strings and clause numbers (e.g. `reading_remarque 9`, `counter_statue 5`, `§3.2`) where dense vector distance often drifts.
   * **RRF ($k=60$):** Combines dense and sparse ranks without scale-mismatch distortion.
   * **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`):** Jointly scores top 20 fused candidates to extract the final top 5 high-precision evidence chunks.

