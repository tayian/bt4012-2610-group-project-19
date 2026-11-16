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

**3. Add the data.** Copy the two data files provided with this project, `client_train.csv` and `invoice_train.csv`, into `data/raw/`. These are the only data files you need; everything else is built from them. If a file is missing or different, `python run.py all` stops at the first step and says so.

**4. Run everything**

```bash
python run.py all
```

This takes about 10 minutes. It checks the two data files, rebuilds every table, figure and model result in the repo, and finishes by running the tests. When it is done, every table, figure and result matches ours exactly. If your installed package versions differ from ours, `git status` may still list some `.parquet`, `.png` or notebook files as modified: libraries such as pyarrow and matplotlib write their version number into these files, so the bytes change while the contents stay identical.

**5. (Optional) LLM features.** Only needed for the Inspector Briefs. Make a copy of `.env.example` named `.env`, put your Gemini API key in it, then check that the key works:

```bash
python run.py test-llm
```

## What `python run.py all` does

| Step | Command | Produces |
|---|---|---|
| Check the raw data | `python run.py check-raw` | Confirms `data/raw/` holds the exact course files (`data/checksums.txt`) |
| Check the split | `python run.py splits` | Validates the committed train/test split in `data/splits/` |
| Clean | `python run.py interim` | `data/interim/*.parquet`, `results/tables/data_quality.csv`, `code_values.csv` |
| Baseline features | `python run.py features` | `data/processed/features_v0.parquet` |
| Behaviour features | `python run.py features-v1` | `data/processed/features_v1.parquet` and `features_v1_with_inspection.parquet` |
| Feature statistics | `python run.py feature-stats` | `results/tables/stat_tests.csv`, `corr_pairs.csv`, `results/figures/corr_heatmap.png` |
| Peer-residual and graph features | `python run.py peer`, `python run.py graph` | `data/processed/peer_residual.parquet`, `graph_features.parquet`, `knn_edges.parquet` |
| Final features | `python run.py features-v2` | `data/processed/features_v2.parquet` and `features_v2_with_inspection.parquet` |
| Graph model | `python run.py label-spreading` | `results/oof/M8_v1.parquet` and its row in `main_results.csv` |
| RAG code book | `python run.py codebook` | `data/corpus/source_docs/codebook.md` |
| Test fixtures | `python run.py stubs` | `data/stubs/` |
| Exploratory analysis | `python run.py eda` | `notebooks/01_eda.ipynb`, `results/tables/basics.csv`, `descriptive.csv`, EDA figures |
| Baseline model | `python run.py baseline` | `results/tables/main_results.csv`, `results/oof/M1_v0.parquet` |
| LightGBM model | `python run.py lightgbm` | `results/tables/main_results.csv`, `results/oof/M3_v0.parquet` |
| Baseline explanations | `python run.py shap` | `results/shap_top5_v0.parquet`, five OOF drivers per training client |
| Tests | `python run.py test` | Runs every test |

Each command also runs the steps it depends on, so you can run any one on its own. `python run.py --list` shows every step.

The heavier model experiments run explicitly with `python run.py model-experiments`:
the LR regularisation grid, random-forest grid, isotonic calibration and imbalance
comparison, plus SHAP. Completed candidates are checkpointed locally and reused on a
matching rerun. The routine `all` command runs the core pipeline, SHAP and tests.
See [docs/model_experiments.md](docs/model_experiments.md) for the commands, output names
and evaluation rules.

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
  raw/          the two course CSVs (not in git; see Quick start)
  interim/      cleaned parquet files (built by python run.py interim)
  processed/    feature tables (built by the feature steps)
  splits/       the fixed train/test split and folds (in git, never re-split)
  corpus/       documents for the RAG system (in git)
  stubs/        small fake versions of every shared file, for testing (in git)
  checksums.txt fingerprints of the two raw files
docs/           evaluation protocol, cleaning rules, RAG design, sources, threat model, notes
notebooks/      01_eda.ipynb (exploratory analysis)
results/        tables/, figures/, oof/ (out-of-fold model scores), models/ (fold models, not in git)
src/
  data/         loading, cleaning, raw-data check, split, stubs
  features/     baseline, consumption/billing and monthly sequence features
  models/       logistic regression, LightGBM, MLP, 1D-CNN, anomaly models, PU learning, shared preprocessing
  eval/         metrics, out-of-fold files, robustness study, rescue analysis
  llm/          RAG corpus tools and the Inspector Brief evaluation harness
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
GEMINI_MODEL=gemini-2.5-flash
```

## Documentation

| File | What it covers |
|---|---|
| [docs/eval_protocol.md](docs/eval_protocol.md) | Split, leakage rules, metrics and cost model |
| [docs/data_cleaning.md](docs/data_cleaning.md) | Every loading and cleaning rule |
| [docs/rag_pipeline.md](docs/rag_pipeline.md) | How the RAG corpus becomes an Inspector Brief |
| [docs/llm_stack.md](docs/llm_stack.md) | LLM, embedding and retrieval choices |
| [docs/sources.md](docs/sources.md) | Literature behind the fraud typologies |
| [docs/threat_model.md](docs/threat_model.md) | Attacker model and attacks for the robustness study |
| [docs/reading_notes.md](docs/reading_notes.md) | Notes on the wide-and-deep CNN paper and what transfers to our data |
