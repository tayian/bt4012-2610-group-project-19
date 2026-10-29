# BT4012 Fraud Analytics — Group Project 19
**The Tunisian Company of Electricity and Gas (STEG) Non-Technical Loss (NTL) Detection & Grounded Investigation**  
*National University of Singapore (NUS) — AY2026/2027 Semester 1*

---

## 1. Project Overview

The **Tunisian Company of Electricity and Gas (STEG)** incurs estimated annual losses of ~200 million Tunisian Dinar (TND) due to energy diversion, meter tampering, billing anomalies, and unmetered consumption. 

Because on-site inspection capacity is strictly finite, the operational objective is **not simply binary classification**, but **client-level risk ranking under a constrained monthly inspection budget** (top 1%, 5%, 10%), paired with an **auditable, grounded LLM "Inspector Brief"** to explain to field investigators why a customer was flagged, what physical signs to inspect on premise, and what remains uncertain.

* **Primary Evaluation Metric:** PR-AUC (Average Precision) & Precision/Recall at top $k$ inspection budget ($k \in \{1, 5, 10\}$%).
* **Business Evaluation Metric:** Expected Net Recovery Cost Curve: $\text{Net}(k) = V \times \text{TP}(k) - C \times k$ (evaluated at $V/C \in \{5, 10, 20\}$).
* **Investigation Layer:** Deterministic RAG (BM25 + MiniLM + RRF + Cross-Encoder) and Multi-Agent workflow generating schema-enforced, citation-backed Inspector Briefs.

There is a pipeline_RAG.md in the repo that explains the RAG pipeline so that it would be easier to understand and visualize the RAG pipeline.

---

## 2. Repository Structure

This repository is organized into modular directories following the team's weekly hand-off contracts:

```text
BT4012Project/
├── .env.example               # Template for environment variables and LLM API keys
├── .gitignore                  # Git ignore rules (protects .env, large parquets, caches)
├── requirements.txt            # Project dependencies across ML, DL, and NLP stacks
├── README.md                   # Project documentation and reproduction guide
│
├── data/                       # Data storage (split by pipeline stage)
│   ├── raw/                    # Untouched source CSVs (client_train, invoice_train)
│   ├── interim/                # Cleaned, standardized parquets (invoice_clean, client)
│   ├── processed/              # Feature stores (features_v1, features_v2, seq.npy)
│   ├── splits/                 # Stratified train_ids.csv, test_ids.csv, folds.csv
│   └── corpus/                 # RAG knowledge corpus (playbook, codebook, SOPs, case cards)
│       ├── specs/              # Architectural blueprints & data schemas
│       │   ├── corpus_spec.md  # Master retrieval & metadata filtering specification
│       │   ├── sop_spec.md     # SOP clause taxonomy & threshold change matrix
│       │   └── case_cards_spec.md # Specification for 300 precedent fraud case dossiers
│       ├── source_docs/        # Human-authored knowledge base in Markdown
│       │   ├── playbook.md     # 12 Non-Technical Loss (NTL) Typologies ([TYP-01] to [TYP-12])
│       │   ├── codebook.md     # Categorical variable decodings ([CODE-<field>-<value>])
│       │   ├── sop_v2023.md    # Synthetic inspection SOP (historical 2023-2024 rules)
│       │   └── sop_v2025.md    # Synthetic inspection SOP (active 2025+ enforcement rules)
│       ├── case_cards.jsonl    # (Week 3) 300 precedent fraud case dossiers
│       └── chunks.jsonl        # (Week 2) Master compiled production RAG index (~350 chunks)
│
├── docs/                       # Architectural specifications & technical contracts
│   ├── sources.md              # 8 authoritative academic and industry references
│   ├── llm_stack.md            # Technical specifications for RAG and agent architecture
│   └── eval_protocol.md        # Frozen evaluation protocol, metrics, and cost model
│
├── src/                        # Production pipeline modules
│   ├── data/                   # Data ingestion, cleaning, and stratified splitting
│   ├── features/               # Feature engineering (aggregate, behavioural, graph, Benford)
│   ├── models/                 # Model implementations (linear, trees, MLP, autoencoder, CNN, fusion)
│   ├── eval/                   # Metrics harness, cost curves, calibration, and fairness
│   ├── llm/                    # Hybrid retrieval, prompt generation, Pydantic briefs, agent tools
│   └── viz/                    # Standardized plotting styles and figure generators
│
├── notebooks/                  # Interactive analysis and experimentation notebooks
│   ├── 01_eda.ipynb            # Descriptive statistics, distributions, and hypothesis tests
│   ├── 02_benford.ipynb        # Digital analysis and Benford's Law anomaly detection
│   └── 03_models.ipynb         # Model benchmark comparisons and tuning
│
└── results/                    # Persistent experimental artifacts and logs
    ├── tables/                 # Benchmark tables (main_results.csv, stat_tests.csv)
    ├── figures/                # Publication-ready charts (cost curves, PR curves, SHAP)
    ├── oof/                    # Out-of-fold prediction parquets per model family
    ├── briefs/                 # Generated Inspector Brief JSON files
    └── agent_logs/             # Audit logs for multi-agent tool execution traces
```

---

## 3. Environment Setup & `requirements.txt`

The repository relies on **Python 3.11+** (the course tutorial notebook `L6_RAG_Hands_On.ipynb` was originally executed on Python 3.10.x, but explicitly recommends **Python 3.11+** for fresh environments; this project baseline is actively configured and verified on **Python 3.12.1**), along with a curated set of dependencies spanning tabular ML, deep learning, graph analytics, and hybrid RAG.

> [!NOTE]
> **Python Version Compatibility:** While the historical course lab ran on Python 3.10, all core ML/DL/NLP dependencies (`torch`, `lightgbm`, `sentence-transformers`, `scikit-learn`, `pydantic`) have modern pre-compiled wheels supporting Python 3.12 without deprecation or build errors.

### 3.1 Creating / Activating Your Conda Environment
It is strongly recommended to run this project inside a dedicated Conda virtual environment (e.g. `BT4012`):

```bash
# Create the environment with Python 3.12 (if not already created)
conda create -n BT4012 python=3.12 -y

# Activate the environment
conda activate BT4012
```

### 3.2 Installing Dependencies
Install all required libraries using `pip`:

```bash
pip install -r requirements.txt
```

### 3.3 Dependencies Breakdown in `requirements.txt`
* **Core Tabular & Classical ML:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `catboost`, `imbalanced-learn`, `optuna`, `shap`, `pyarrow`.  
  *(Used for feature engineering, Optuna hyperparameter tuning, tree boosting, and SHAP feature attributions).*
* **Deep Learning & Graph Analytics:** `torch`, `networkx`.  
  *(Used for the PyTorch Autoencoder anomaly detector, 1D-CNN temporal sequence models, and kNN customer behavioral graphs).*
* **Visualization & Utilities:** `matplotlib`, `seaborn`, `pyyaml`, `python-dotenv`, `tqdm`.  
  *(Used for publication-style figures, YAML configurations, environment loading, and progress bars).*
* **LLM, Hybrid RAG & NLP (Role D):** `rank-bm25`, `sentence-transformers`, `pydantic`, `google-genai`, `google-generativeai`.  
  *(Used for sparse BM25 indexing, dense semantic embeddings with all-MiniLM-L6-v2, Pydantic JSON schema validation, and Gemini API calls).*
* **Optional Alternative LLM Providers:** `openai`, `anthropic`.  
  *(Used for multi-provider benchmark comparisons).*

---

## 4. Configuration & Secrets Management (`.env.example`)

To protect team API keys and maintain strict cybersecurity hygiene, **private credentials are never committed to the Git repository**.

### 4.1 How to Set Up Your Local `.env`
1. The repository includes a template file named [`.env.example`](file:///.env.example).
2. Create your own local `.env` file by copying the template:
   ```bash
   # On Windows (PowerShell):
   Copy-Item .env.example .env

   # On macOS / Linux:
   cp .env.example .env
   ```
3. Open your newly created `.env` file and insert your personal API keys:
   ```ini
   # Primary LLM Provider (Google Gemini API)
   GEMINI_API_KEY=your_actual_gemini_api_key_here

   # Optional Alternative Providers
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENAI_API_KEY=your_openai_key_here

   # Global Random Seed (Frozen across all splits and models)
   RANDOM_SEED=4012
   ```

### 4.2 Security Protection via `.gitignore`
* The project's [`.gitignore`](file:///.gitignore) is explicitly configured to ignore `.env` and `*.env`:
  ```gitignore
  # Environment Variables & Secrets
  .env
  *.env
  ```
* Each developer maintains their own local `.env` file. Git will automatically prevent this file from being staged or pushed to GitHub, ensuring API keys and secrets remain secure.

---

## 5. Team Roles & Workstream Ownership

To eliminate workflow dependencies, work is decoupled through standardized interface contracts (`docs/` and `data/splits/`) and tagged releases (`release-wN`):

* **Role A (Data, Features & Graph Lead):** Cleaning, EDA, hypothesis tests, Benford's Law analysis, feature store (`features_v1`, `features_v2`), kNN graph features, and fairness audits.
* **Role B (Modelling, Decision & Integrator Lead):** Stratified splitting harness (`SEED=4012`), linear baselines (M1), tree ensembles (M2, M3, M9), model tuning, stacking fusion (M10), cost curves, and test-set evaluation.
* **Role C (Deep Learning, Robustness & Evaluation Lead):** Resampled monthly sequence tensor (`seq.npy`), tabular MLP (M4), Autoencoder anomaly scorer (M6), 1D-CNN (M7), adversarial attacks (PGD/mimicry) & defence, and brief faithfulness scoring.
* **Role D (LLM, RAG & Agent Lead):** Knowledge corpus construction, hybrid retrieval (BM25 + MiniLM + RRF), grounded Inspector Brief generation, single/multi-agent workflows, and prompt injection defense.

---

## 6. Reproducibility & Global Standards
* **Fixed Random Seed:** `SEED = 4012` is enforced across all splits, cross-validation folds, numpy/torch generators, and model training routines.
* **Unit of Analysis:** Every model evaluates and scores at the **client level** (one row = one client), as the client is the physical unit of inspection.
* **Strict Non-Leakage Contract:** Resampling (e.g. SMOTE), scalers, and target encoders are fitted strictly inside training folds. RAG precedent case cards are drawn **exclusively from confirmed training-set fraud cases** and never from the held-out test split.