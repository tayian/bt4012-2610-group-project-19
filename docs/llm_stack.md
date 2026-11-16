# LLM & RAG Technical Stack Specification (`docs/llm_stack.md`)

**Project:** BT4012 Fraud Analytics — The Tunisian Company of Electricity and Gas (STEG)  
**Workstream:** LLM, RAG & agents  
**Status:** Draft  
**Last Updated:** October 2026  

---

## 1. Overview & Architectural Objectives

The LLM/RAG layer in this project does **not** predict fraud or assign labels on its own; risk ranking is strictly owned by the ML model suite (M1–M10: LightGBM, CatBoost, Autoencoder, 1D-CNN). The LLM's operational role is to generate an **audit-ready, grounded Inspector Brief** for human field inspectors explaining *why* a customer was flagged under a finite monthly inspection budget.

The stack is chosen to support these goals:
1. **Grounded numbers and precedents**: the model is asked to cite exact chunk IDs (`[TYP-xx]`, `[CODE-<field>-<value>]`, `[SOP-v2025-x.x]`, `[CASE-<client_id>]`, `[SNAPSHOT]`) from metadata-filtered hybrid retrieval. This reduces hallucination but does not rule it out; the faithfulness checks in §4 measure how often briefs stay grounded.
2. **Repeatability & auditability**: `temperature = 0.0` (module guideline: *"Temperature 0 for anything auditable"*). This makes outputs much more repeatable but not guaranteed identical across runs or model versions.
3. **Decoupled Execution**: Fast, local embedding and sparse search requiring no remote vector database infrastructure.
4. **Prompt Security**: Structural isolation of untrusted meter-reader remarks and customer metadata against prompt injection.

```
+-------------------------------------------------------------+
|               ML Risk Scorer & SHAP Top-5                   |
|                 (LightGBM / Fusion Model)                   |
+-------------------------------------------------------------+
                              |
                              v
                  [Client Fraud Risk Snapshot]
                              |
                              v
+-------------------------------------------------------------+
|              Query Rewriter (3 Targeted Queries)            |
+-------------------------------------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +-------------------+           +-------------------+
    |  Dense Retrieval  |           | Sparse Retrieval  |
    | (all-MiniLM-L6-v2)|           |     (BM25Okapi)   |
    +-------------------+           +-------------------+
              |                               |
              +---------------+---------------+
                              |
                              v
                [Reciprocal Rank Fusion (RRF)]
                              |
                              v
              +-------------------------------+
              |   Reranker (Cross-Encoder)    |
              |   Top 5 Evidence Chunks       |
              +-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|        Brief Writer LLM (Prompt Sandwich + Pydantic)        |
|   Tested: gemini-2.5-flash | Alt: current Claude / GPT     |
|            (Temperature 0, cites retrieved chunks)          |
+-------------------------------------------------------------+
                              |
                              v
                  [Inspector Brief + faithfulness checks]
```

---

## 2. Core Stack Components

| Layer | Component / Library | Implementation Specification | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Primary LLM** | Google Gemini API (`google-genai`) | `gemini-2.5-flash`, set once by `GEMINI_MODEL` in `.env` and checked by `tests/test_llm_api.py` (`python run.py test-llm`). Experiments use this one model ID throughout and record it with every brief. | Native `response_schema` support for Pydantic output; context window comfortably holds five evidence chunks plus the client snapshot. |
| **Alternative LLMs** | Anthropic Claude API / OpenAI API | A current Claude or GPT model; pin and record the exact model ID when the comparison is run. | Kept behind a provider interface for comparison runs. |
| **Decoding Config** | Temperature: `0.0`<br>Max Tokens: `1200` | `top_p = 1.0`, `seed = 4012` where the provider supports it | Makes runs as repeatable as the provider allows, for reproducible evaluation. |
| **Dense Retrieval** | `sentence-transformers` | `sentence-transformers/all-MiniLM-L6-v2` (384 dims, normalized cosine/dot product) | Small (about 90 MB) and fast enough on CPU for a corpus of ~420 chunks (not benchmarked); good general-purpose semantic matching. |
| **Sparse Retrieval** | `rank-bm25` | `BM25Okapi` with regex word/code tokenization | Captures exact alphanumeric codes (`counter_statue`, `reading_remarque`, `tarif_type`) where dense embeddings drift. |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Custom fusion function ($k = 60$ constant), Top 20 dense + Top 20 sparse $\rightarrow$ Top 5 fused | Solves score calibration mismatch between BM25 unbounded scores and cosine similarities. |
| **Cross Re-ranker** | `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scores top 20 query-chunk pairs jointly via full cross-attention transformer, selecting top 5 highest-relevance chunks. |
| **Structured Output** | `pydantic>=2.0` | `InspectorBrief` Pydantic `BaseModel` schema with regex validators | Enforces machine-actionable JSON structure for downstream queueing, preventing silent schema drift. |
| **Evaluation / Judge** | Rule-based regex + LLM-as-a-judge | `src/llm/faithfulness.py` | Checks citation existence, number accuracy against the snapshot, and citation faithfulness. |

---

## 3. Component Details & Design Choices

### 3.1 LLM Generation Parameters & Prompt Architecture
* **System Prompting ("Prompt Sandwich" Pattern)**:
  - *Top Slice (Role & Constraints)*: Role definition ("You are a utility fraud case analyst for STEG...") and strict negative boundaries ("Do NOT declare guilt; cite only verified evidence; do NOT convert risk scores into percentage probabilities of fraud.").
  - *Filling (Evidence & Grounding)*: Structured client snapshot (features, SHAP top-5), top-5 retrieved evidence chunks (typologies, SOP clauses, precedent case cards).
  - *Bottom Slice (Instructions & Schema)*: Grounding constraint ("Cite only what is in the evidence. Never invent numbers.") placed nearest the final tokens to prevent "lost-in-the-middle" degradation, followed by the JSON schema.
* **Prompt Injection Defense**:
  - All untrusted client fields (e.g. meter-reader remarks, client notes, external descriptions) are strictly encapsulated within `<untrusted_data>...</untrusted_data>` tags.
  - The system prompt explicitly instructs the model to treat all content within `<untrusted_data>` strictly as passive text evidence, ignoring any embedded instructions or prompt manipulation.

### 3.2 Hybrid Retrieval Architecture (BM25 + all-MiniLM-L6-v2)
- **Dense Embedding**: Encodes query and chunks into normalized 384-dimensional vectors. Cosine similarity retrieves the top-20 semantic candidates.
- **Sparse BM25**: Indexing tokenizes exact utility codes (e.g., `CODE-counter_statue-5`, `SOP-v2025-2.1`). BM25 retrieves the top-20 keyword candidates.
- **RRF Merge Formula**:
  $$\text{RRF Score}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{sparse}}(d)}$$
  Top 5 unique chunks by RRF score (or refined by Cross-Encoder) are forwarded to prompt context.
- **Metadata Filters Applied Prior to Fusion**:
  * `jurisdiction == 'TN'`
  * `split == 'train'` (Strict anti-leakage rule: precedent case cards come only from the training set; a case chunk without a `split` is dropped)
  * The client's own case card (`CASE-<client_id>`) is always excluded when retrieving for that client (`client_filters(snapshot)` or `retrieve(..., client_id=...)` in `src/llm/retrieval.py`)
  * `valid_from <= alert_date < (valid_to or '9999-12-31')` (SOP version matching; half-open window `[valid_from, valid_to)`, so v2023 does not apply on 2025-01-01)
  * `alert_date` is the date the risk list is scored and the brief is written, not an invoice date (invoices end in 2019). It is set explicitly for each experiment.

### Retrieval benchmark and current status
- `python3 -m src.llm.retrieval_eval` scores dense, BM25, hybrid and hybrid + rerank on the 25 queries in `data/corpus/retrieval_queries.json` (10 concept, 10 exact-code, 5 mixed) and writes `results/tables/retrieval_eval.csv` plus the gold chunk's rank per query and method in `results/tables/retrieval_eval_by_query.csv`. With 25 queries one query moves overall recall by 0.04 (0.10 within the 10-query categories, 0.20 within the 5 mixed queries), so small differences between methods are not meaningful on their own.
- The exact-code and mixed queries are written as an inspector would ask them ("what does meter status 46 mean"); they do not contain the chunk ID or copy the chunk's sentences. An earlier version did, which made those categories trivially easy (recall 1.0 for BM25).
- Snapshot signal features (`yoy_drop_max`, `zero_run_max`, `n_index_backwards`, `benford_mad`, `round_reading_share`, ...) are read from `features_v1` with training percentiles; a missing value (e.g. `benford_mad` with fewer than 20 positive readings) is passed as missing, not 0.
- Test clients have no risk score or SHAP drivers yet (the OOF scores cover training clients only), so briefs for held-out clients need the final model's test scores, which come in a later step.
- `results/tables/query_rewriter_eval.json` was produced with the old default model id (`gemini-3.8-flash`) and the old client sample (the five top-scored clients, two of them non-fraud). The rewriter now uses `GEMINI_MODEL` (default `gemini-2.5-flash`) and a seeded sample of 3 fraud and 2 non-fraud training clients spread across score ranks; the file must be rerun with an API key (`python run.py query-rewriter`).

### 3.3 Output Schema Specification (`results/briefs/*.json`)
Every brief generated by the model must validate against the following Pydantic schema structure:

```json
{
  "client_id": "string",
  "risk_score": 0.0,
  "risk_rank_pct": 0.0,
  "risk_tier": "high | medium | low",
  "evidence": [
    {
      "point": "string",
      "citations": ["TYP-01", "SNAPSHOT"]
    }
  ],
  "onsite_checks": ["string"],
  "applicable_rules": [
    {
      "rule_id": "SOP-v2025-2.1",
      "action": "string"
    }
  ],
  "similar_cases": ["CASE-train_Client_10023"],
  "uncertainty": "string"
}
```

* **Risk Score Calibration & Score Interpretation**:
  - **Raw Scores vs. Real Probabilities:** Models using class weighting (e.g. the Logistic Regression baseline with balanced class weights has an average score of ~0.39 even though only 5.6% of clients are fraud, and LightGBM with class weights behaves similarly) produce raw scores that are uncalibrated and **not** real probabilities.
  - **Operational Requirement:** Either:
    - `risk_score` must be a calibrated probability (calibrating the final model, e.g. via isotonic regression, before it feeds the briefs), or
    - The brief only shows `risk_rank_pct` and `risk_tier`, and `risk_score` is explicitly labelled as a relative priority score, not a "chance of fraud".
  - **Prompt Restriction on Probability Phrases:** The brief generation prompt must forbid the LLM from translating the score into a percentage or likelihood claim (e.g. "62% likely to be fraud"). Percentile rank (`risk_rank_pct`) is always safe since it depends purely on ordering across the client population; `risk_tier` is likewise safe as long as tiers are partitioned strictly by rank percentiles (e.g. top 1% = `high`, 1–5% = `medium`, rest = `low`) rather than uncalibrated raw score thresholds.

---

## 4. Grounding & Evaluation Harness (Faithfulness)

Built with the evaluation workstream (`src/llm/faithfulness.py`):
1. **Citation Existence Check:** Every bracketed citation in the generated brief must match an ID present in the retrieved evidence set or `[SNAPSHOT]`. (Pass criteria: 100%).
2. **Numerical Veracity Check:** Regex extraction of every number in the brief; every number must appear verbatim in the client's `SNAPSHOT` or a cited clause. (Pass criteria: 100%).
3. **Temporal Validity Check:** Every SOP clause cited must satisfy `valid_from <= alert_date < valid_to` (half-open window).
4. **Three-Run Benchmark:** The system will be evaluated across 3 experimental regimes (RAG lecture concept):
   * Run 1: No evidence (raw LLM baseline).
   * Run 2: Unfiltered hybrid RAG.
   * Run 3: Metadata-filtered hybrid RAG.

---

## 5. Environment & Dependencies

Add to root `requirements.txt`:
```txt
# LLM Providers
google-genai>=0.1.0
google-generativeai>=0.8.0
anthropic>=0.25.0
openai>=1.20.0

# Retrieval & Embeddings
sentence-transformers>=2.6.0
rank-bm25>=0.2.2
torch>=2.2.0

# Data & Schema Validation
pydantic>=2.5.0
pyyaml>=6.0.1
```

### Environment Variables
Configure in local `.env` (gitignored):
```bash
# Primary
GEMINI_API_KEY="your-gemini-api-key"

# Alternatives / Benchmarking
ANTHROPIC_API_KEY="your-anthropic-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

---

## 6. Mock / Offline Mode for Testing
To ensure the pipeline can be run without incurring API costs and during automated testing:
* A `MockLLMClient` will be implemented in `src/llm/briefs.py` that parses the snapshot and template-fills a valid JSON Inspector Brief deterministically if no API key is configured.
