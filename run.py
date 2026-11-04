# Run the pipeline. From the repo root:
#
#   python run.py all          rebuild everything committed from the two raw data files
#   python run.py baseline     run one step (and the steps it depends on)
#   python run.py --list       show every step
#
# Works the same on Windows, macOS and Linux. Each step runs the steps it needs first,
# once per call. Settings such as file locations can be changed with options (see --help).

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def commands(s):
    # Each step: (steps it needs first, commands to run, one-line description).
    raw_check = [] if s.skip_raw_check else ["check-raw"]
    dl_step = "features" if s.dl_features == "v0" else "features-v1"
    return {
        "all": (
            [
                "check-raw",
                "interim",
                "features",
                "features-v1",
                "codebook",
                "stubs",
                "eda",
                "baseline",
                "lightgbm",
                "test",
            ],
            [],
            "Rebuild every committed output from the two raw files, then run the tests",
        ),
        "check-raw": (
            [],
            [[PY, "-m", "src.data.check_raw", "--raw-dir", s.raw_dir]],
            "Check data/raw/ holds the exact course files (data/checksums.txt)",
        ),
        "splits": (
            [],
            [[PY, "-m", "src.data.split", "--clients", s.clients, "--split-dir", s.split_dir]],
            "Validate the committed train/test split and folds (created once if missing)",
        ),
        "interim": (
            raw_check,
            [
                [
                    PY,
                    "-m",
                    "src.data.load",
                    "--clients",
                    s.clients,
                    "--invoices",
                    s.invoices,
                    "--out-dir",
                    s.interim,
                ],
                [PY, "-m", "src.data.clean", "--interim", s.interim],
            ],
            "Load and clean the raw files into data/interim/ and write the data quality tables",
        ),
        "stubs": (
            raw_check,
            [[PY, "-m", "src.data.make_stubs", "--clients", s.clients]],
            "Rebuild the fake test fixtures in data/stubs/",
        ),
        "eda": (
            ["interim", "splits"],
            [
                [
                    PY,
                    "-m",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.record_timing=False",
                    "notebooks/01_eda.ipynb",
                ]
            ],
            "Re-run the exploratory analysis notebook (tables and figures)",
        ),
        "features": (
            ["interim", "splits"],
            [[PY, "-m", "src.features.aggregate", "--interim", s.interim, "--out", s.features_v0]],
            "Build the 15 baseline features (features_v0)",
        ),
        "features-v1": (
            ["interim"],
            [
                [
                    PY,
                    "-m",
                    "src.features.behavioural",
                    "--interim",
                    s.interim,
                    "--out",
                    s.features_v1,
                ]
            ],
            "Build the consumption and billing features (features_v1)",
        ),
        "seq": (
            ["interim"],
            [
                [
                    PY,
                    "-m",
                    "src.features.sequence",
                    "--invoices",
                    str(Path(s.interim) / "invoice_clean.parquet"),
                    "--clients",
                    str(Path(s.interim) / "client.parquet"),
                    "--out",
                    "data/processed/",
                ]
            ],
            "Build the monthly consumption tensor data/processed/seq.npy",
        ),
        "seq-plot": (
            ["seq", "splits"],
            [[PY, "-m", "src.viz.seq_sanity"]],
            "Plot the monthly sequences (results/figures/seq_sanity.png)",
        ),
        "codebook": (
            ["interim", "splits"],
            [
                [
                    PY,
                    "-m",
                    "src.llm.corpus.build_codebook",
                    "--interim",
                    s.interim,
                    "--splits",
                    s.split_dir,
                ]
            ],
            "Rebuild the RAG code book from the data",
        ),
        "baseline": (
            ["features"],
            [
                [
                    PY,
                    "-m",
                    "src.models.linear",
                    "--features",
                    s.features_v0,
                    "--clients",
                    s.clients,
                    "--split-dir",
                    s.split_dir,
                    "--results-dir",
                    s.results_dir,
                    "--max-iter",
                    str(s.lr_max_iter),
                ]
            ],
            "Train the logistic regression baseline (M1); writes main_results.csv",
        ),
        "lightgbm": (
            ["features"],
            [
                [
                    PY,
                    "-m",
                    "src.models.trees",
                    "--features",
                    s.features_v0,
                    "--clients",
                    s.clients,
                    "--split-dir",
                    s.split_dir,
                    "--results-dir",
                    s.results_dir,
                    "--n-jobs",
                    str(s.tree_jobs),
                ]
            ],
            "Train weighted LightGBM (M3) on features_v0; writes OOF scores and metrics",
        ),
        "mlp": (
            [dl_step, "splits"],
            [[PY, "-m", "src.models.mlp", "--features", s.dl_features, "--loss", s.mlp_loss]],
            "Train the MLP (M4)",
        ),
        "cnn": (
            [dl_step, "splits", "seq"],
            [[PY, "-m", "src.models.cnn", "--features", s.dl_features, "--save"]],
            "Train the 1D-CNN (M7) and save its fold models to results/models/",
        ),
        "anomaly": (
            [dl_step, "splits", "seq"],
            [[PY, "-m", "src.models.anomaly", "--features", s.dl_features]],
            "Score the isolation forest (M5) and autoencoder (M6)",
        ),
        "wide-deep": (
            [dl_step, "splits", "seq"],
            [[PY, "-m", "src.models.cnn", "--features", s.dl_features, "--arch", "widedeep"]],
            "Train the wide-and-deep CNN (M7wd: 1D + 10x12 Conv2D + static branches)",
        ),
        "rescue": (
            ["lightgbm", "anomaly"],
            [[PY, "-m", "src.eval.rescue", "--m3", "v0", "--anomaly", s.dl_features]],
            "Share of M3's missed frauds that M5/M6 rank highly (results/tables/rescue.csv)",
        ),
        "pu": (
            ["lightgbm"] if s.pu_m3 == "v0" else [],
            [[PY, "-m", "src.models.pu", "--m3", s.pu_m3, "--weight", s.pu_weight]],
            "PU learning on the M3 OOF scores (M11, results/tables/pu_estimate.csv)",
        ),
        "robustness": (
            ["cnn"],
            [
                [PY, "-m", "src.eval.adversarial", "all", "--features", s.dl_features],
                [PY, "-m", "src.viz.robustness_fig", "--features", s.dl_features],
            ],
            "Attack M3 and M7 and test the defences (results/tables/robustness.csv)",
        ),
        "test": ([], [[PY, "-m", "pytest", "-q", "tests"]], "Run every test"),
        "test-rag": (
            [],
            [[PY, "-m", "pytest", "-q", "tests/test_rag_environment.py"]],
            "Check the retrieval libraries (downloads a ~90 MB model once)",
        ),
        "test-faithfulness": (
            [],
            [[PY, "-m", "pytest", "-q", "tests/test_faithfulness.py"]],
            "Check the Inspector Brief evaluation harness on sample briefs",
        ),
        "test-llm": (
            [],
            [[PY, "-m", "pytest", "-q", "-rs", "tests/test_llm_api.py"]],
            "Check the Gemini API key in .env (skipped without one)",
        ),
        "lint": (
            [],
            [[PY, "-m", "ruff", "check", "."], [PY, "-m", "ruff", "format", "--check", "."]],
            "Check code style",
        ),
        "format": (
            [],
            [[PY, "-m", "ruff", "check", "--fix", "."], [PY, "-m", "ruff", "format", "."]],
            "Fix code style",
        ),
    }


def run(step, steps, done):
    # Run a step's dependencies first, then the step; each step runs at most once.
    if step in done:
        return
    needs, cmds, _ = steps[step]
    for need in needs:
        run(need, steps, done)
    for cmd in cmds:
        print("$", " ".join(cmd[2:] if cmd[0] == PY else cmd), flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit(f"Step '{step}' failed.")
    done.add(step)


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline steps, e.g. `python run.py all`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("steps", nargs="*", help="steps to run (see --list)")
    parser.add_argument("--list", action="store_true", help="show every step and exit")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--clients", default=None, help="defaults to <raw-dir>/client_train.csv")
    parser.add_argument("--invoices", default=None, help="defaults to <raw-dir>/invoice_train.csv")
    parser.add_argument(
        "--skip-raw-check", action="store_true", help="skip the checksum check (small test files)"
    )
    parser.add_argument("--interim", default="data/interim")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--features-v0", default="data/processed/features_v0.parquet")
    parser.add_argument("--features-v1", default="data/processed/features_v1.parquet")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--lr-max-iter", type=int, default=15000)
    parser.add_argument("--tree-jobs", type=int, default=2)
    parser.add_argument("--dl-features", choices=["v0", "v1"], default="v1")
    parser.add_argument("--mlp-loss", choices=["bce", "focal", "both"], default="bce")
    parser.add_argument("--pu-m3", default="v0", help="version of the M3 OOF file used by pu")
    parser.add_argument(
        "--pu-weight", default="fold", help='M3 positive-class weight: "fold", "none" or a number'
    )
    args = parser.parse_args()
    args.clients = args.clients or str(Path(args.raw_dir) / "client_train.csv")
    args.invoices = args.invoices or str(Path(args.raw_dir) / "invoice_train.csv")

    steps = commands(args)
    if args.list or not args.steps:
        width = max(map(len, steps))
        for name, (_, _, description) in steps.items():
            print(f"  {name:<{width}}  {description}")
        return
    unknown = [step for step in args.steps if step not in steps]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)} (see --list)")
    done = set()
    for step in args.steps:
        run(step, steps, done)


if __name__ == "__main__":
    main()
