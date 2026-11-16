# RAG pipeline

How the corpus documents become retrieval chunks and how a query flows through hybrid retrieval to an Inspector Brief.

Human-Authored Markdown Docs:
  ├── playbook.md         ──┐
  ├── codebook.md         ──┼──► [Parsing Script] ──┐
  ├── sop_v2023.md        ──┤                        │
  └── sop_v2025.md        ──┘                        │
                                                     ├──►  data/corpus/chunks.jsonl
Python-Generated Cases:                             │     (~420 total chunks)
  └── case_cards.py       ──────► [case_cards]   ────┘
      (From 300 training clients)
                                                                │
                                                                ▼
                                                   [Hybrid Retrieval Engine]
                                                   ├── 1. Metadata Pre-Filter
                                                   ├── 2. BM25 (Exact Code Matching)
                                                   ├── 3. MiniLM (Dense Semantic Search)
                                                   ├── 4. Reciprocal Rank Fusion (RRF)
                                                   └── 5. Cross-Encoder Reranker
                                                                │
                                                                ▼
                                                    [Top 5 Evidence Chunks]
                                                                │
                                                                ▼
                                                    [Prompt Sandwich Context]
                                                                │
                                                                ▼
                                                   [Grounded Inspector Brief]

The metadata pre-filter keeps precedent case cards from the training split only and, when retrieving for a client, always drops that client's own card (`CASE-<client_id>`). Retrieval quality is measured with `python3 -m src.llm.retrieval_eval` (see `docs/llm_stack.md`); with 25 benchmark queries one query is 0.04 of recall.

**Briefs and agents.** `python run.py briefs` writes Inspector Briefs for the 30 clients the final model (M10) ranks highest, under three evidence settings: no evidence, unfiltered hybrid retrieval and filtered hybrid retrieval. `python run.py agents` runs the single agent and the multi-agent workflow (Planner, Evidence Gatherer, Brief Writer, Auditor) on the same clients. In the unfiltered run, expired SOP-v2023 clauses are retrieved for 57% of clients and rule applicability falls to 66%. With the metadata filter, it is 100%. Tables, caching and model notes are in `docs/llm_stack.md` section 7.
