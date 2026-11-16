# Responsible-deployment checks on the final model's ranking.
#
#   python3 -m src.eval.responsible fairness     -> results/tables/fairness.csv
#   python3 -m src.eval.responsible calibration  -> results/tables/calibration_by_group.csv,
#                                                   results/figures/calibration_by_district.png
#   python3 -m src.eval.responsible stability    -> results/tables/explanation_stability.csv
#
# Fairness: who gets inspected at the cost-optimal budget (V/C = 10) compared with where the
# fraud is, per district, region and client category. A group selected at more than twice its
# share of the frauds is flagged; groups under MIN_GROUP clients are reported but not flagged.
# Calibration: does an M10 score of 0.3 mean a 30% fraud rate in every district?
# Stability: the tuned M3 refitted with three seeds on folds 1 to 4, and the top-5 SHAP
# drivers of each fold-0 client compared across seeds (Jaccard overlap; target above 0.6).
# Fairness and calibration read results/final_scores.parquet, on the training OOF scores and,
# once scored, the test scores.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import net_recovery_curve
from src.eval.oof import write_frames

GROUPS = ["district", "region", "client_catg"]
MIN_GROUP = 200
RATIO = 10
SEEDS = [4012, 4013, 4014]


def scored_clients():
    final = pd.read_parquet("results/final_scores.parquet")
    cli = pd.read_parquet("data/interim/client.parquet")[["client_id", "target", *GROUPS]]
    cli["client_id"] = cli["client_id"].astype(str)
    return final.merge(cli, on="client_id", validate="one_to_one")


def optimal_k(y, scores, ratio=RATIO):
    k, net = net_recovery_curve(y, scores, value_per_fraud=ratio, cost_per_visit=1.0)
    return int(k[np.argmax(net)])


def fairness():
    data = scored_clients()
    rows = []
    for split, part in data.groupby("split"):
        y, s = part["target"].to_numpy(), part["score"].to_numpy()
        k = optimal_k(y, s)
        selected = np.zeros(len(part), bool)
        selected[np.argsort(-s, kind="stable")[:k]] = True
        part = part.assign(selected=selected)
        for group in GROUPS:
            for value, g in part.groupby(group):
                frauds = g["target"].sum()
                chosen = g["selected"].sum()
                share_fraud = frauds / part["target"].sum()
                share_selected = chosen / k
                ratio = share_selected / share_fraud if share_fraud else np.nan
                rows.append(
                    {
                        "split": split,
                        "budget_k": k,
                        "group": group,
                        "value": value,
                        "n_clients": len(g),
                        "fraud_rate": frauds / len(g),
                        "share_of_clients": len(g) / len(part),
                        "share_of_frauds": share_fraud,
                        "share_of_selected": share_selected,
                        "selection_rate": chosen / len(g),
                        "precision_in_group": (
                            g.loc[g.selected, "target"].mean() if chosen else np.nan
                        ),
                        "recall_in_group": (
                            g.loc[g.selected, "target"].sum() / frauds if frauds else np.nan
                        ),
                        "selected_vs_fraud_share": ratio,
                        "flag_over_2x": bool(len(g) >= MIN_GROUP and ratio > 2),
                    }
                )
    table = pd.DataFrame(rows)
    write_frames({Path("results/tables/fairness.csv"): table})
    flagged = table[table.flag_over_2x]
    print(f"{len(flagged)} group(s) selected at more than 2x their fraud share")
    print(flagged[["split", "group", "value", "n_clients", "selected_vs_fraud_share"]].to_string())
    return table


def calibration(bins=10):
    import matplotlib.pyplot as plt

    from src.viz import style

    data = scored_clients()
    rows, curves = [], {}
    for split, part in data.groupby("split"):
        for value, g in [("all", part), *part.groupby("district")]:
            y, s = g["target"].to_numpy(), g["score"].to_numpy()
            b = pd.qcut(s, bins, labels=False, duplicates="drop")
            table = (
                pd.DataFrame({"b": b, "y": y, "s": s})
                .groupby("b")
                .agg(mean_score=("s", "mean"), fraud_rate=("y", "mean"), n=("y", "size"))
            )
            ece = float((table.n * (table.mean_score - table.fraud_rate).abs()).sum() / len(g))
            rows.append(
                {
                    "split": split,
                    "district": value,
                    "n_clients": len(g),
                    "fraud_rate": y.mean(),
                    "mean_score": s.mean(),
                    "brier": float(np.mean((s - y) ** 2)),
                    "ece": ece,
                }
            )
            curves[(split, value)] = table
    out = pd.DataFrame(rows)
    write_frames({Path("results/tables/calibration_by_group.csv"): out})
    print(out.round(4).to_string(index=False))
    style.apply()
    split = "test" if "test" in data["split"].unique() else "train_oof"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    top = max(t.mean_score.max() for (sp, _), t in curves.items() if sp == split)
    ax.plot([0, top], [0, top], color=style.GRID, lw=1.5, label="Perfect calibration")
    palette = ["#2a78d6", "#eb6834", "#1f9e74", "#8a5cd1", "#52514e"]
    for colour, ((sp, value), t) in zip(
        palette, [(k, v) for k, v in curves.items() if k[0] == split]
    ):
        label = "All districts" if value == "all" else f"District {value}"
        ax.plot(t.mean_score, t.fraud_rate, marker="o", ms=4, lw=2, color=colour, label=label)
    ax.set_xlabel("Mean M10 score in decile")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(f"M10 calibration by district ({split.replace('_', ' ')})", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/calibration_by_district.png", dpi=style.DPI)
    return out


def stability(n_jobs=4):
    import shap

    from src.eval.explain import positive_shap
    from src.models.test_set import load_split
    from src.models.trees import fit_m3

    (ids, X, y), _ = load_split("v1")
    folds = pd.read_csv("data/splits/folds.csv", dtype={"client_id": "string"}).set_index(
        "client_id"
    )["fold"]
    f = folds.loc[ids].to_numpy()
    params = json.loads(Path("results/tables/M3_tuned_params_v1.json").read_text())
    Xtr, Xva = X[f != 0], X[f == 0]
    tops, scores = [], []
    for seed in SEEDS:
        model = fit_m3(Xtr, y[f != 0], **{**params, "random_state": seed, "n_jobs": n_jobs})
        values = positive_shap(
            shap.TreeExplainer(model.estimator_, feature_perturbation="tree_path_dependent"),
            model._prepare(Xva),
        )
        tops.append(np.argsort(-np.abs(values), axis=1, kind="stable")[:, :5])
        scores.append(model.predict_proba(Xva)[:, 1])
        print(f"seed {seed} done", flush=True)
    high = scores[0] >= np.quantile(scores[0], 0.95)
    rows = []
    for a in range(len(SEEDS)):
        for b in range(a + 1, len(SEEDS)):
            jac = np.array(
                [len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(tops[a], tops[b])]
            )
            rank_corr = pd.Series(scores[a]).corr(pd.Series(scores[b]), method="spearman")
            for subset, mask in (
                ("all fold-0 clients", np.ones(len(jac), bool)),
                ("top 5% by score", high),
            ):
                rows.append(
                    {
                        "seed_a": SEEDS[a],
                        "seed_b": SEEDS[b],
                        "clients": subset,
                        "n": int(mask.sum()),
                        "jaccard_mean": jac[mask].mean(),
                        "jaccard_median": float(np.median(jac[mask])),
                        "share_above_0_6": (jac[mask] > 0.6).mean(),
                        "top1_same_share": (tops[a][mask, 0] == tops[b][mask, 0]).mean(),
                        "score_spearman": rank_corr,
                    }
                )
    table = pd.DataFrame(rows)
    write_frames({Path("results/tables/explanation_stability.csv"): table})
    print(table.round(3).to_string(index=False))
    return table


def main():
    p = argparse.ArgumentParser(description="Fairness, calibration-by-group and SHAP stability.")
    p.add_argument("check", choices=["fairness", "calibration", "stability"])
    p.add_argument("--n-jobs", type=int, default=4)
    a = p.parse_args()
    if a.check == "fairness":
        fairness()
    elif a.check == "calibration":
        calibration()
    else:
        stability(a.n_jobs)


if __name__ == "__main__":
    main()
