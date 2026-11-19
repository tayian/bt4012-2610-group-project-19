# STEG Fraud Detection (BT4012 Group 19)

We rank customers of STEG, the Tunisian electricity and gas utility, by how likely they are to be committing fraud (meter tampering, bypassing, illegal reconnection and similar "non-technical losses"). STEG can only inspect a few hundred homes a month, so the goal is to put the right customers at the top of the inspection list. An LLM then writes a short, cited "Inspector Brief" for each flagged customer explaining why they were flagged and what to check on site.

## Quick start

You need Python 3.10 or newer. Every command below works on Windows, macOS and Linux.

**1. Get the code**

```bash
git clone https://github.com/tayian/bt4012-2610-group-project-19.git
cd bt4012-2610-group-project-19
```

**2. Install the packages** (a virtual environment keeps them separate from your other projects)

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Once the environment is active, `python` refers to it on every system, so the rest of the steps are the same everywhere.

**3. The data.** The two raw files are in the repo, gzipped: `data/raw/client_train.csv.gz` and `data/raw/invoice_train.csv.gz`. The first step of `python run.py all` (`check-raw`) unpacks them into `data/raw/` and checks them against `data/checksums.txt`, so there is nothing to copy. Everything else is built from these two files.

Optional shortcut: the `steg_project_data` folder provided with the submission also holds the files the pipeline builds. Copy its `interim/` folder into `data/interim/` and its `processed/` folder into `data/processed/` to skip rebuilding them. Its `README.txt` lists every file.

The data are STEG's training files from the Zindi *Fraud Detection in Electricity and Gas Consumption* challenge: 135,493 clients and 4,476,749 invoices up to December 2019.

**4. Run everything**

```bash
python run.py all
```

This takes about 15 minutes on a laptop and needs an internet connection the first time, because the retrieval evaluation downloads two small models (`all-MiniLM-L6-v2` and the `ms-marco-MiniLM-L-6-v2` cross-encoder) from Hugging Face. It checks the two data files, rebuilds the data, the feature tables (including `features_v4`), the baseline models and their SHAP drivers, the RAG corpus and the retrieval evaluation, and finishes by running the tests. The files it writes match the committed ones. The final model and most numbers in the report come from `python run.py final` and the other steps listed under [Reproducing the final results](#reproducing-the-final-results). If your installed package versions differ from ours, `git status` may still list some `.parquet`, `.png` or notebook files as modified: libraries such as pyarrow and matplotlib write their version number into these files, so the bytes change while the contents stay identical.

**macOS note.** If LightGBM fails to import with a `libomp.dylib` error, run `brew install libomp`.

**Tested with** Python 3.10.6 on macOS. `requirements.txt` gives minimum versions; `requirements-lock.txt` lists the exact versions that produced the committed results.

**5. (Optional) LLM features.** Only needed for the Inspector Briefs. Make a copy of `.env.example` named `.env`, put your Gemini API key in it, then check that the key works:

```bash
python run.py test-llm
```

## What `python run.py all` does

| Step | Command | Produces |
|---|---|---|
| Check the raw data | `python run.py check-raw` | Unpacks the gzipped raw files and confirms they are the exact course files (`data/checksums.txt`) |
| Check the split | `python run.py splits` | Validates the committed train/test split in `data/splits/` |
| Clean | `python run.py interim` | `data/interim/*.parquet`, `results/tables/data_quality.csv`, `code_values.csv` |
| Baseline features | `python run.py features` | `data/processed/features_v0.parquet` |
| Behaviour features | `python run.py features-v1` | `data/processed/features_v1.parquet` and `features_v1_with_inspection.parquet` |
| Feature statistics | `python run.py feature-stats` | `results/tables/stat_tests.csv`, `corr_pairs.csv`, `results/figures/corr_heatmap.png` |
| Peer-residual and graph features | `python run.py peer`, `python run.py graph` | `data/processed/peer_residual.parquet`, `graph_features.parquet`, `knn_edges.parquet` |
| Peer and graph feature table | `python run.py features-v2` | `data/processed/features_v2.parquet` and `features_v2_with_inspection.parquet` |
| Meter, tariff and recency features | `python run.py features-v3` | `data/processed/meter_tariff.parquet`, `features_v3.parquet` and `features_v4.parquet` (the final model's table) |
| Graph model | `python run.py label-spreading` | `results/oof/M8_v1.parquet` and its row in `main_results.csv` |
| RAG code book | `python run.py codebook` | `data/corpus/source_docs/codebook.md` |
| RAG corpus | `python run.py corpus` | `data/corpus/case_cards.jsonl` (300 precedent case cards) and `data/corpus/chunks.jsonl` |
| Retrieval evaluation | `python run.py retrieval-eval` | `results/tables/retrieval_eval.csv` and `retrieval_eval_by_query.csv` (downloads the retrieval models once) |
| Test fixtures | `python run.py stubs` | `data/stubs/` |
| Exploratory analysis | `python run.py eda` | `notebooks/01_eda.ipynb`, `results/tables/basics.csv`, `descriptive.csv`, EDA figures |
| Baseline model | `python run.py baseline` | `results/tables/main_results.csv`, `results/oof/M1_v0.parquet` |
| LightGBM model | `python run.py lightgbm` | `results/tables/main_results.csv`, `results/oof/M3_v0.parquet` |
| Baseline explanations | `python run.py shap` | `results/shap_top5_v0.parquet`, five OOF drivers per training client |
| Tests | `python run.py test` | Runs every test; the 22 that read built data skip until `python run.py all` has run |

Each command also runs the steps it depends on, so you can run any one on its own. `python run.py --list` shows every step.

The heavier model experiments run explicitly with `python run.py model-experiments`:
the LR regularisation grid, random-forest grid, isotonic calibration and imbalance
comparison, plus SHAP. Completed candidates are checkpointed locally and reused on a
matching rerun. The routine `all` command runs the core pipeline, SHAP, the RAG corpus and tests.
See [docs/model_experiments.md](docs/model_experiments.md) for the commands, output names
and evaluation rules.

## Reproducing the final results

`python run.py all` rebuilds the data, features, baseline models and the retrieval evaluation.
The final model and its results come from:

```bash
python run.py final
```

It tunes LightGBM on `features_v4` (`tune`), trains CatBoost (`catboost`) and the wide-and-deep
CNN (`wide-deep-final`), runs the fusion ablation that selects the final model (`fusion`), scores
the held-out test set (`test-set`), and reruns the fairness, calibration, stability, budget and
robustness analyses (`responsible`, `budget-figures`, `robustness-final`). It takes about 6 hours
on a laptop CPU (most of it the wide-and-deep CNN and the Optuna search).

`fusion` also reads the out-of-fold scores of the other models in the ablation. These are
committed, so it runs on a fresh clone; to rebuild them too, run `python run.py mlp --mlp-loss focal`
(M4), `cnn` (M7), `graphsage` (M8sage) and `anomaly` (M5, M6) first. `label-spreading` (M8) is
already part of `all`. The other numbers in the
report come from separate steps: `benford`, `eda-figures`, `time-robustness`,
`features-v3-audit`, `rescue`, `pu`, `briefs` and `agents` (see the tables below).

The out-of-fold and test scores behind the final model and the reported tables are committed
in `results/`, so the tables and figures can be checked without rerunning; some exploratory
runs are kept only locally. The Inspector Brief steps (`briefs`, `agents`) need a Gemini key;
their LLM responses are committed too, so `python run.py briefs` rescores them without API
calls.

## Final model, evaluation and Inspector Briefs

These steps build the final model and everything reported about it, in order. Each runs the
steps it depends on. The test set is scored once, by `test-set`; the LLM steps need a Gemini
key in `.env`, and their responses are cached in `results/`, so a rerun on the same model makes
no API calls.

| Command | What it does |
|---|---|
| `python run.py features-v3` | Builds the F12 meter, tariff and recency features, `features_v3` (v1 + F12, the leakage comparison) and `features_v4` (the final table, without the features a meter replacement moves) |
| `python run.py features-v3-audit` | F12 ablation, meter-change timing, the pre-2013 leakage check and `meter_change_year.png` (about 1.5 h) |
| `python run.py time-robustness` | Tuned M3 on invoices up to 2016 against the full history |
| `python run.py tune` | Tunes LightGBM (M3) with Optuna on `features_v4`, 30 trials (about an hour) |
| `python run.py catboost` | CatBoost (M9) on `features_v4` |
| `python run.py wide-deep-final` | Wide-and-deep CNN (M7wd) on `features_v4`, the second model in the final stack (about 3 h on CPU) |
| `python run.py fusion` | Fusion ablation and the M10 stacker's out-of-fold scores (needs the model OOF files) |
| `python run.py robustness-final` | Consumption-hiding attacks on the final feature table, with defences |
| `python run.py test-set` | Refits on all training clients and scores the held-out test set once (`test_results.csv`) |
| `python run.py responsible` | Fairness of inspection allocation, calibration by district, SHAP stability |
| `python run.py budget-figures` | Precision against inspection budget, net recovery and the cost-model table |
| `python run.py benford`, `python run.py eda-figures` | Benford tests and the remaining EDA figures |
| `python run.py corpus` | The 300 precedent case cards and the RAG `chunks.jsonl` |
| `python run.py retrieval-eval` | Dense, BM25, hybrid and reranked retrieval on the 25-query benchmark |
| `python run.py query-rewriter` | Query rewriter (snapshot to 3 search queries) on 5 clients |
| `python run.py briefs` | Inspector Briefs for the top 30 clients: no evidence, hybrid and filtered hybrid (`three_run_eval.csv`) |
| `python run.py agents` | Single agent against multi-agent on the same 30 clients (`agent_comparison.csv`) |
| `python run.py test-briefs`, `python run.py test-llm` | Brief and agent tests with the LLM mocked; a live check of the Gemini key |

## Other commands

| Command | What it does |
|---|---|
| `python run.py seq` | Monthly consumption tensor `data/processed/seq.npy` (clients x 120 months x 3 channels) |
| `python run.py seq-plot` | Sanity plot of the sequences (`results/figures/seq_sanity.png`) |
| `python run.py mlp` | MLP (M4) on `features_v1`; `--mlp-loss both` also trains the focal-loss version. `--dl-features v0` runs this and the other neural steps on `features_v0` |
| `python run.py cnn` | Dual-input 1D-CNN (M7) on the sequences and `features_v1`; saves fold models to `results/models/` |
| `python run.py graphsage` | GraphSAGE (M8sage) on the client graph with `features_v2` node features |
| `python run.py anomaly` | Isolation forest (M5) and autoencoder (M6) |
| `python run.py wide-deep` | Wide-and-deep CNN (M7wd): the M7 branches plus a Conv2D branch on a 10 x 12 year-by-month grid |
| `python run.py rescue` | Share of the frauds LightGBM misses that M5/M6 rank in their top 10% (`results/tables/rescue.csv`) |
| `python run.py pu` | PU learning (M11) on the LightGBM OOF scores: estimated true fraud rate in `results/tables/pu_estimate.csv` |
| `python run.py robustness` | Attacks on LightGBM and the 1D-CNN, with defences (`results/tables/robustness.csv`, `results/figures/robustness_<version>.png`) |
| `python run.py test-faithfulness` | Checks the Inspector Brief evaluation harness on sample briefs |
| `python run.py test-rag` | Checks the retrieval libraries (downloads a ~90 MB model once) |
| `python run.py lint` | Checks code style |
| `python run.py format` | Fixes code style |

## Project layout

```text
data/
  raw/          the two course CSVs, gzipped (check-raw unpacks them)
  interim/      cleaned parquet files (built by python run.py interim)
  processed/    feature tables (built by the feature steps)
  splits/       the fixed train/test split and folds (in git, never re-split)
  corpus/       documents, case cards and chunks for the RAG system (in git)
  stubs/        small fake versions of every shared file, for testing (in git)
  checksums.txt fingerprints of the two raw files
configs/        final.json: the final model's components, stack weights and seed
docs/           decisions, evaluation protocol, features, cleaning rules, RAG and LLM design, notes
notebooks/      01_eda.ipynb (exploratory analysis), 02_benford.ipynb (Benford tests)
results/
  tables/       every reported table (main_results.csv, test_results.csv, ...)
  figures/      every reported figure
  oof/          out-of-fold scores per model and feature version
  test/         test-set scores of the final-table models
  briefs/       cached Inspector Briefs, rewritten queries and judge scores
  agent_logs/   cached single-agent and multi-agent runs and history plots
  coefficients/ logistic regression coefficients
  final_scores.parquet         final model scores for every client (train OOF and test)
  shap_top5_<version>.parquet  five SHAP drivers per client (shap_top5_final for the final model)
  models/, runs/               fold models and experiment checkpoints (local only, not in git)
src/
  data/         loading, cleaning, raw-data check, split, stubs
  features/     baseline, behaviour, peer-residual, graph, meter/tariff (F12), Benford and sequence features
  models/       logistic regression, random forest, LightGBM and its Optuna tuning, CatBoost, MLP,
                1D-CNN and wide-and-deep CNN, GraphSAGE, label spreading, anomaly models,
                PU learning, imbalance handling, the fusion stack and the one-off test-set run
  eval/         metrics, out-of-fold files, SHAP, calibration, robustness, rescue analysis,
                time robustness, fairness and other responsible-AI checks
  llm/          RAG corpus tools, retrieval, client snapshots, query rewriter, Inspector Briefs,
                single and multi-agent systems, and the faithfulness evaluation
  viz/          shared figure style and plots
tests/          unit and integration tests (python run.py test)
```

More detail on the data folder is in [data/README.md](data/README.md).

## How we evaluate

- **One row = one client.** The client is who gets inspected, so every model scores clients.
- **Fixed split.** 80% of clients for training (with 5 cross-validation folds), 20% held out as the test set, which is scored once at the end. The split is committed and never redrawn; the random seed is 4012 everywhere.
- **No leakage.** Anything learned from data (scalers, encoders, resampling) is fitted inside each training fold only.
- **Metrics.** PR-AUC is the headline metric (only 5.6% of clients are fraud). We also report precision and recall when inspecting the top 1%, 5% and 10% of clients, and the expected net recovery of an inspection budget.

The full rules are in [docs/eval_protocol.md](docs/eval_protocol.md).

## Results

The final model (M10) stacks tuned LightGBM and a wide-and-deep network on `features_v4`. That
table is the leak-free feature set: meter and tariff features a post-detection meter
replacement would move are excluded (docs/decisions.md, section 8).

| Model | CV PR-AUC (5 folds) | Test PR-AUC [95% CI] | Test recall in top 5% |
|---|---|---|---|
| M1 logistic regression | not run on v4 | 0.193 [0.179, 0.210] | 0.209 |
| M3 tuned LightGBM | 0.300 +/- 0.011 | 0.294 [0.272, 0.318] | 0.316 |
| **M10 final stack** | **0.306 +/- 0.011** | **0.294 [0.272, 0.318]** | **0.316** |

The fraud base rate is 5.6%, so the final model is about 5x better than random. The test set was
scored once. Every decision behind these numbers is in [docs/decisions.md](docs/decisions.md).

## Configuration

API keys live in a local `.env` file, which is never committed. `.env.example` is the template:

```text
# Copy this file to .env and fill in your own values. Never commit .env (it is gitignored).
# Only needed for the LLM steps and `python run.py test-llm`; the rest of the pipeline runs without it.

# Gemini API key for the Inspector Briefs
GEMINI_API_KEY=your_gemini_api_key_here
# Model used for every Inspector Brief
GEMINI_MODEL=gemini-3.1-flash-lite
```

## Documentation

| File | What it covers |
|---|---|
| [docs/decisions.md](docs/decisions.md) | Every modelling decision, the evidence for it, and the test-set results |
| [docs/eval_protocol.md](docs/eval_protocol.md) | Split, leakage rules, metrics and cost model |
| [docs/data_cleaning.md](docs/data_cleaning.md) | Every loading and cleaning rule |
| [docs/feature_list.md](docs/feature_list.md) | Every feature table and feature (F1 to F12) |
| [docs/lightgbm.md](docs/lightgbm.md) | The LightGBM baseline (M3) and the shared tree interface |
| [docs/model_experiments.md](docs/model_experiments.md) | LR and random-forest grids, calibration, imbalance comparison and SHAP |
| [docs/analysis_notes.md](docs/analysis_notes.md) | Benford tests, the second EDA pass and the time-robustness check |
| [docs/rag_pipeline.md](docs/rag_pipeline.md) | How the RAG corpus becomes an Inspector Brief |
| [docs/llm_stack.md](docs/llm_stack.md) | LLM, embedding and retrieval choices |
| [docs/sources.md](docs/sources.md) | Literature behind the fraud typologies |
| [docs/threat_model.md](docs/threat_model.md) | Attacker model and attacks for the robustness study |
| [docs/reading_notes.md](docs/reading_notes.md) | Notes on the wide-and-deep CNN paper and what transfers to our data |
