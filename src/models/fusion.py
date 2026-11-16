# M10 fusion: combine the models' out-of-fold scores in parallel, then score the test set once.
#
# Each model's scores are turned into percentile ranks within the fold that produced them
# (scores from different models, and from different folds, are not on the same scale). Two
# combiners are compared on the saved folds:
#   rank average   the mean of the ranks
#   LR stacker     logistic regression on the ranks, fitted on four folds and scored on the
#                  fifth, so every stacked score is still out-of-fold
# The ablation adds one model group at a time to the reference model (tuned M3 on v1).
#
#   python3 -m src.models.fusion cv      -> results/tables/fusion_ablation.csv,
#                                           results/oof/M10_v1.parquet + its main_results row
#   python3 -m src.models.fusion test    -> results/tables/test_results.csv (bootstrap 95% CIs),
#                                           results/tables/test_fusion_gain.csv,
#                                           results/final_scores.parquet, configs/final.json
#
# The test step needs results/test/*.parquet from src.models.test_set. Test ranks are taken
# over the test clients, and the stacker is refitted on all training OOF ranks.

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
# Model name -> OOF/test file stem. M3_tuned on v1 is the reference for every ablation.
FILES = {
    "M3_tuned": "M3_tuned_v1",
    "M9": "M9_v1",
    "M7wd": "M7wd_v1",
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
# The final model: the smallest set whose stacked gain over M3 is positive in every fold.
FINAL = ["M3_tuned", "M7wd"]
TEST_MODELS = ["M1", "M3", "M3_tuned", "M9", "M7wd"]


def load_oof(models):
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
    scores = stack_oof(ranks, y, folds, FINAL)
    save_run(
        frame["client_id"].to_numpy(),
        folds,
        y,
        scores,
        "M10",
        "v1",
        run_id="M10_lr_stack_" + "+".join(FINAL) + "_train_cv",
        combiner="lr_stack_on_fold_ranks",
    )


def test_ranks(models):
    frame = None
    for name in models:
        o = pd.read_parquet(f"results/test/{name}_v1.parquet")[["client_id", "score"]]
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
    frame, ranks = load_oof(FINAL)
    y_tr = frame["target"].to_numpy()
    fit = stacker().fit(ranks[FINAL], y_tr)
    test = test_ranks(TEST_MODELS)
    y = test["target"].to_numpy()
    test_rank = test[FINAL].rank(pct=True)
    test["M10"] = fit.predict_proba(test_rank)[:, 1]
    rows = [metric_row(name, y, test[name].to_numpy()) for name in TEST_MODELS + ["M10"]]
    table = pd.DataFrame(rows)
    point, lo, hi, share = paired_gain(y, test["M10"].to_numpy(), test["M3_tuned"].to_numpy())
    gain = pd.DataFrame(
        [
            {
                "comparison": "M10 - M3_tuned",
                "pr_auc_gain": point,
                "lo": lo,
                "hi": hi,
                "share_of_resamples_positive": share,
            }
        ]
    )
    oof = pd.read_parquet("results/oof/M10_v1.parquet")[["client_id", "score"]]
    final = pd.concat(
        [
            oof.assign(split="train_oof"),
            test[["client_id", "M10"]].rename(columns={"M10": "score"}).assign(split="test"),
        ],
        ignore_index=True,
    )
    final["client_id"] = final["client_id"].astype(str)
    final["rank_pct"] = final.groupby("split")["score"].rank(pct=True, ascending=False)
    final["model_version"] = "M10_v1"
    config = {
        "model_version": "M10_v1",
        "combiner": "logistic regression on percentile ranks",
        "models": {name: FILES[name] for name in FINAL},
        "coefficients": dict(zip(FINAL, fit.coef_[0].round(6).tolist())),
        "intercept": float(fit.intercept_[0]),
        "m3_params": "results/tables/M3_tuned_params_v1.json",
        "features": "data/processed/features_v1.parquet",
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
    run_cv() if a.step == "cv" else run_test()


if __name__ == "__main__":
    main()
