Human-Authored Markdown Docs:
  ├── playbook.md         ──┐
  ├── codebook.md         ──┼──► [Parsing Script] ──┐
  ├── sop_v2023.md        ──┤                        │
  └── sop_v2025.md        ──┘                        │
                                                     ├──►  data/corpus/chunks.jsonl
Python-Generated Cases:                             │     (~350 total chunks)
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
