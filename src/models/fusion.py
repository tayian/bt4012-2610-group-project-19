# M10 fusion: stack the models' per-fold percentile ranks, compare model groups on the saved
# folds, then refit the chosen stack and score the test set once.
# Run: python3 -m src.models.fusion cv (then test)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.eval.metrics import bootstrap_ci, bootstrap_values, budget_metrics
from src.eval.oof import write_frames
from src.models.common import save_run

SEED = 4012
VERSION = "v4"
# OOF file stem per model. The sequence, graph and anomaly models keep their earlier feature
# versions because their inputs do not use the v4 additions.
FILES = {
    "M3_tuned": f"M3_tuned_{VERSION}",
    "M9": f"M9_{VERSION}",
    "M7wd": f"M7wd_{VERSION}",
    "M7": "M7_v1",
    "M4": "M4focal_v1",
    "M8sage": "M8sage_v2",
    "M8": "M8_v1",
    "M5": "M5_v1",
    "M6": "M6_v1",
}
ABLATION = [
    ["M3_tuned"],
    ["M3_tuned", "M9"],
    ["M3_tuned", "M7wd"],
    ["M3_tuned", "M7"],
    ["M3_tuned", "M4"],
    ["M3_tuned", "M8sage"],
    ["M3_tuned", "M8"],
    ["M3_tuned", "M5", "M6"],
    ["M3_tuned", "M9", "M7wd"],
    ["M3_tuned", "M9", "M7wd", "M7", "M4"],
    list(FILES),
]
# FINAL is the smallest stack whose gain over tuned M3 is positive in every fold; STACK is the
# stack scored on the test set.
FINAL = ["M3_tuned", "M7wd"]
STACK = ["M3_tuned", "M7wd"]
TEST_MODELS = ["M1", "M3", "M3_tuned", "M9", "M7wd"]
FINAL_NAME = "M10" if len(FINAL) > 1 else FINAL[0]


def load_oof(models):
    # Scores are ranked within each fold, since raw scores from different folds and models are
    # not on the same scale.
    cli = pd.read_parquet("data/interim/client.parquet")[["client_id", "target"]]
    cli["client_id"] = cli["client_id"].astype(str)
    frame = None
    for name in models:
        o = pd.read_parquet(f"results/oof/{FILES[name]}.parquet")[["client_id", "fold", "score"]]
        o = o.rename(columns={"score": name}).assign(client_id=lambda d: d.client_id.astype(str))
        frame = o if frame is None else frame.merge(o.drop(columns="fold"), on="client_id")
    frame = frame.merge(cli, on="client_id", validate="one_to_one")
    ranks = frame.groupby("fold")[models].rank(pct=True)
    return frame, ranks


def stacker():
    return LogisticRegression(C=1.0, max_iter=1000)


def stack_oof(ranks, y, folds, models):
    # Fit on four folds and score the fifth, so every stacked score stays out-of-fold.
    out = np.zeros(len(y))
    for k in range(5):
        fit = stacker().fit(ranks.loc[folds != k, models], y[folds != k])
        out[folds == k] = fit.predict_proba(ranks.loc[folds == k, models])[:, 1]
    return out


def fold_pr_auc(y, scores, folds):
    return np.array([average_precision_score(y[folds == k], scores[folds == k]) for k in range(5)])


def run_cv():
    frame, ranks = load_oof(list(FILES))
    y, folds = frame["target"].to_numpy(), frame["fold"].to_numpy()
    base = fold_pr_auc(y, frame["M3_tuned"].to_numpy(), folds)
    rows = []
    for models in ABLATION:
        for method in ("rank_average", "lr_stack"):
            if method == "rank_average":
                scores = ranks[models].mean(axis=1).to_numpy()
            else:
                scores = stack_oof(ranks, y, folds, models)
            per_fold = fold_pr_auc(y, scores, folds)
            gain = per_fold - base
            rows.append(
                {
                    "models": "+".join(models),
                    "method": method,
                    "pr_auc_mean": per_fold.mean(),
                    "pr_auc_std": per_fold.std(ddof=1),
                    "gain_vs_m3_mean": gain.mean(),
                    "gain_vs_m3_min": gain.min(),
                    "folds_improved": int((gain > 0).sum()),
                }
            )
    table = pd.DataFrame(rows)
    write_frames({Path("results/tables/fusion_ablation.csv"): table})
    print(table.round(4).to_string(index=False))
    scores = stack_oof(ranks, y, folds, STACK)
    save_run(
        frame["client_id"].to_numpy(),
        folds,
        y,
        scores,
        "M10",
        VERSION,
        run_id="M10_lr_stack_" + "+".join(STACK) + "_train_cv",
        combiner="lr_stack_on_fold_ranks",
    )


def test_ranks(models):
    # Raw test scores of each model joined with the labels; they are ranked by the caller.
    frame = None
    for name in models:
        o = pd.read_parquet(f"results/test/{name}_{VERSION}.parquet")[["client_id", "score"]]
        o = o.rename(columns={"score": name}).assign(client_id=lambda d: d.client_id.astype(str))
        frame = o if frame is None else frame.merge(o, on="client_id", validate="one_to_one")
    cli = pd.read_parquet("data/interim/client.parquet")[["client_id", "target"]]
    frame = frame.merge(cli.assign(client_id=cli.client_id.astype(str)), on="client_id")
    return frame


def metric_row(name, y, scores, n_boot=1000):
    m = budget_metrics(y, scores)
    row = {"model": name, "evaluation_split": "test", "n_clients": len(y)}
    for metric in ("pr_auc", "roc_auc"):
        lo, hi = bootstrap_ci(y, scores, metric, n_boot)
        row.update({metric: m[metric], f"{metric}_lo": lo, f"{metric}_hi": hi})
    for key in (
        "precision@1%",
        "recall@1%",
        "precision@5%",
        "recall@5%",
        "precision@10%",
        "recall@10%",
        "brier",
        "net_opt_vc5",
        "net_opt_vc10",
        "net_opt_vc20",
        "k_opt_share_vc10",
    ):
        row[key] = m.get(key, np.nan)
    return row


def paired_gain(y, a, b, n_boot=1000):
    # Bootstrap of PR-AUC(a) - PR-AUC(b) on the same resampled test clients.
    def diff(yy, idx):
        return average_precision_score(yy, a[idx]) - average_precision_score(yy, b[idx])

    vals = bootstrap_values(y, np.arange(len(y)), diff, n_boot)
    point = average_precision_score(y, a) - average_precision_score(y, b)
    return point, *np.percentile(vals, [2.5, 97.5]), float((vals > 0).mean())


def run_test():
    frame, ranks = load_oof(STACK)
    fit = stacker().fit(ranks[STACK], frame["target"].to_numpy())
    test = test_ranks(TEST_MODELS)
    y = test["target"].to_numpy()
    test["M10"] = fit.predict_proba(test[STACK].rank(pct=True))[:, 1]
    rows = [metric_row(name, y, test[name].to_numpy()) for name in TEST_MODELS + ["M10"]]
    table = pd.DataFrame(rows)
    table["final"] = table["model"].eq(FINAL_NAME)
    point, lo, hi, share = paired_gain(y, test["M10"].to_numpy(), test["M3_tuned"].to_numpy())
    gain = pd.DataFrame(
        [
            {
                "comparison": "M10 stack - M3_tuned",
                "pr_auc_gain": point,
                "lo": lo,
                "hi": hi,
                "share_of_resamples_positive": share,
            }
        ]
    )
    final_version = f"{FINAL_NAME}_{VERSION}"
    oof = pd.read_parquet(
        f"results/oof/{FILES[FINAL_NAME] if FINAL_NAME in FILES else final_version}.parquet"
    )
    final = pd.concat(
        [
            oof[["client_id", "score"]].assign(split="train_oof"),
            test[["client_id", FINAL_NAME]]
            .rename(columns={FINAL_NAME: "score"})
            .assign(split="test"),
        ],
        ignore_index=True,
    )
    final["client_id"] = final["client_id"].astype(str)
    final["rank_pct"] = final.groupby("split")["score"].rank(pct=True, ascending=False)
    final["model_version"] = final_version
    config = {
        "model_version": final_version,
        "models": {name: FILES[name] for name in FINAL},
        "selection": "smallest stack with a positive CV gain over tuned M3 in every fold; "
        "none qualified on v3, so tuned M3 alone"
        if len(FINAL) == 1
        else "LR stack on percentile ranks",
        "m3_params": f"results/tables/M3_tuned_params_{VERSION}.json",
        "features": f"data/processed/features_{VERSION}.parquet",
        "runner_up_stack": {
            "models": STACK,
            "coefficients": dict(zip(STACK, fit.coef_[0].round(6).tolist())),
            "intercept": float(fit.intercept_[0]),
        },
        "seed": SEED,
    }
    Path("configs").mkdir(exist_ok=True)
    Path("configs/final.json").write_text(json.dumps(config, indent=2) + "\n")
    write_frames(
        {
            Path("results/tables/test_results.csv"): table,
            Path("results/tables/test_fusion_gain.csv"): gain,
            Path("results/final_scores.parquet"): final[
                ["client_id", "split", "score", "rank_pct", "model_version"]
            ],
        }
    )
    cols = ["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "recall@5%", "precision@5%"]
    print(table[cols].round(4).to_string(index=False))
    print(gain.round(4).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="M10 fusion on the saved folds, then the test set.")
    p.add_argument("step", choices=["cv", "test"])
    a = p.parse_args()
    if a.step == "cv":
        run_cv()
    else:
        run_test()


if __name__ == "__main__":
    main()
