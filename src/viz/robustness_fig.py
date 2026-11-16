# Robustness figure: recall@5% clean vs attacked, before and after defence, for one feature
# version of results/tables/robustness.csv (python run.py robustness).
#
#   python3 -m src.viz.robustness_fig --features v0   -> results/figures/robustness_v0.png

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.viz import style

ORDER = [
    "M3",
    "M3 without consumption features",
    "M3 + adv. augmentation",
    "M3 + aug. uniform only",
    "M3 + aug. smooth only",
    "M7",
    "M7 + PGD adv. training",
    "M7 + mimicry augmentation",
]
SHORT = {
    "M3 without consumption features": "M3\nno consumption",
    "M3 + adv. augmentation": "M3\n+ augmentation",
    "M3 + aug. uniform only": "M3 + aug.\nuniform only",
    "M3 + aug. smooth only": "M3 + aug.\nsmooth only",
    "M7 + PGD adv. training": "M7\n+ PGD training",
    "M7 + mimicry augmentation": "M7\n+ mimicry aug.",
}
# Clean in neutral grey, then one hue per attack (lighter to darker as knowledge grows).
ATTACKS = [
    ("clean", "Clean", "#8a8984"),
    ("uniform", "Uniform -30%", "#86b6ef"),
    ("smooth", "Smooth-decline mimicry", style.SINGLE),
    ("pgd", "PGD (white-box)", style.FRAUD),
]


def main():
    p = argparse.ArgumentParser(description="Plot recall@5% clean vs attacked.")
    p.add_argument("--features", default="v0", choices=["v0", "v1"])
    a = p.parse_args()
    style.apply()
    t = pd.read_csv("results/tables/robustness.csv")
    t = t[t["feature_version"] == a.features]
    mean = t.pivot(index="model", columns="attack", values="mean")
    std = t.pivot(index="model", columns="attack", values="std")
    models = [m for m in ORDER if m in mean.index]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.2
    present = [row for row in ATTACKS if row[0] in mean]
    for j, (attack, label, color) in enumerate(present):
        xs, ys, es = [], [], []
        for i, m in enumerate(models):
            if pd.notna(mean.loc[m, attack]):
                xs.append(i + (j - (len(present) - 1) / 2) * width)
                ys.append(mean.loc[m, attack])
                es.append(std.loc[m, attack])
        ax.bar(xs, ys, width, yerr=es, color=color, label=label, capsize=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([SHORT.get(m, m) for m in models])
    ax.set_ylabel(f"Recall@5% (mean of 5 folds, features {a.features})")
    ax.set_title("A fraudster lowers declared consumption by up to 30%: frauds kept in the top 5%")
    ax.legend(ncol=4, loc="upper left", frameon=False)
    ax.set_ylim(0, max(0.5, float(t["mean"].max()) * 1.25))
    fig.savefig(f"results/figures/robustness_{a.features}.png")


if __name__ == "__main__":
    main()
