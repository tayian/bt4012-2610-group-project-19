# Budget figures and the cost-model table on the test set.
#
#   python3 -m src.viz.budget_figures
#     results/figures/precision_vs_budget.png   report figure F4: precision against the share of
#                                               clients inspected, one line per model
#     results/figures/net_recovery.png          report figure F5: net recovery against budget
#                                               for V/C = 5, 10 and 20, one panel per model
#     results/tables/cost_model.csv             k*, its share of clients, precision, recall and
#                                               net recovery at k*, per model and V/C ratio
#
# Reads results/test/*.parquet and the M10 test scores in results/final_scores.parquet.
# Net recovery uses cost 1 per visit and value V/C per fraud found (docs/eval_protocol.md).

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.metrics import VALUE_COST_RATIOS, net_recovery_curve
from src.eval.oof import write_frames
from src.viz import style

# Fixed colour per model; the final model is the darkest and drawn last.
MODELS = [
    ("M1", "M1 logistic regression", "#8a8984"),
    ("M9", "M9 CatBoost", "#86b6ef"),
    ("M7wd", "M7wd wide-and-deep CNN", "#eb6834"),
    ("M3_tuned", "M3 tuned LightGBM", style.SINGLE),
    ("M10", "M10 fusion (final)", "#0b3d91"),
]
RATIO_COLOURS = {5: "#86b6ef", 10: style.SINGLE, 20: "#0b3d91"}
MAX_SHARE = 0.30


def test_scores():
    cli = pd.read_parquet("data/interim/client.parquet")[["client_id", "target"]]
    cli["client_id"] = cli["client_id"].astype(str)
    final = pd.read_parquet("results/final_scores.parquet")
    frame = final.loc[final.split == "test", ["client_id", "score"]].rename(
        columns={"score": "M10"}
    )
    for name, _, _ in MODELS[:-1]:
        o = pd.read_parquet(f"results/test/{name}_v1.parquet")[["client_id", "score"]]
        frame = frame.merge(o.rename(columns={"score": name}), on="client_id", validate="1:1")
    return frame.merge(cli, on="client_id", validate="1:1")


def precision_curve(y, s):
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    return k / len(y), tp / k


def main():
    style.apply()
    data = test_scores()
    y = data["target"].to_numpy()
    base = y.mean()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, label, colour in MODELS:
        share, precision = precision_curve(y, data[name].to_numpy())
        keep = (share <= MAX_SHARE) & (share >= 0.005)
        ax.plot(share[keep] * 100, precision[keep], color=colour, lw=2, label=label)
    ax.axhline(base, color=style.TEXT_SECONDARY, lw=1, ls="--")
    ax.text(
        MAX_SHARE * 100,
        base,
        f"random ({base:.1%})",
        ha="right",
        va="bottom",
        fontsize=8,
        color=style.TEXT_SECONDARY,
    )
    for budget in (1, 5, 10):
        ax.axvline(budget, color=style.GRID, lw=1)
    ax.set_xlabel("Share of test clients inspected (%)")
    ax.set_ylabel("Precision (share of inspections that find fraud)")
    ax.set_title("Precision at each inspection budget (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/precision_vs_budget.png")
    plt.close(fig)

    rows = []
    panels = [m for m in MODELS if m[0] in ("M3_tuned", "M10")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (name, label, _) in zip(axes, panels):
        s = data[name].to_numpy()
        for ratio in VALUE_COST_RATIOS:
            k, net = net_recovery_curve(y, s, value_per_fraud=ratio, cost_per_visit=1.0)
            best = int(np.argmax(net))
            order = np.argsort(-s, kind="stable")
            found = y[order][: k[best]].sum()
            rows.append(
                {
                    "model": name,
                    "value_cost_ratio": ratio,
                    "k_opt": int(k[best]),
                    "k_opt_share": k[best] / len(y),
                    "precision_at_k_opt": found / k[best],
                    "recall_at_k_opt": found / y.sum(),
                    "net_recovery": float(net[best]),
                    "net_recovery_all_inspected": float(net[-1]),
                    "break_even_precision": 1 / ratio,
                }
            )
            keep = k <= MAX_SHARE * len(y)
            ax.plot(
                k[keep] / len(y) * 100,
                net[keep],
                color=RATIO_COLOURS[ratio],
                lw=2,
                label=f"V/C = {ratio}",
            )
            ax.plot(
                k[best] / len(y) * 100,
                net[best],
                "o",
                ms=8,
                color=RATIO_COLOURS[ratio],
                mec="white",
                mew=2,
            )
            ax.annotate(
                f"k* = {k[best]:,}",
                (k[best] / len(y) * 100, net[best]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=style.TEXT_SECONDARY,
            )
        ax.axhline(0, color=style.TEXT_SECONDARY, lw=1)
        ax.set_title(label)
        ax.set_xlabel("Share of test clients inspected (%)")
    axes[0].set_ylabel("Net recovery (units; cost 1 per visit)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/net_recovery.png")
    plt.close(fig)

    table = pd.DataFrame(rows)
    write_frames({Path("results/tables/cost_model.csv"): table})
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
