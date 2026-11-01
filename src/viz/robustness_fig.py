# Report figure F6: recall@5% clean vs attacked, before and after defence.

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = [
    "M3 v1_with_inspection (all groups)",
    "M3 + monotone F6",
    "M3 + adv. augmentation",
    "M3 without F6",
    "M7",
    "M7 + adv. training",
    "M7 + mimicry augmentation",
]
LABEL = {
    "M3 v1_with_inspection (all groups)": "M3",
    "M3 + monotone F6": "M3 + monotone F6",
    "M3 + adv. augmentation": "M3 + adv. augment.",
    "M3 without F6": "M3 without F6",
    "M7": "M7 1D-CNN",
    "M7 + adv. training": "M7 + PGD adv. train",
    "M7 + mimicry augmentation": "M7 + mimicry augment.",
}
ATT = [
    ("clean", "Clean", "#4C4C4C"),
    ("uniform", "Uniform −30%", "#DD8452"),
    ("smooth", "Smooth-decline mimicry", "#C44E52"),
    ("pgd", "PGD (white-box)", "#8172B3"),
]


def main():
    t = pd.read_csv("results/tables/robustness.csv")
    w = t.pivot(index="model", columns="attack", values="mean")
    sd = t.pivot(index="model", columns="attack", values="std")
    models = [m for m in ORDER if m in w.index]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    bw = 0.2
    for j, (a, lab, col) in enumerate(ATT):
        if a not in w:
            continue
        xs, ys, es = [], [], []
        for i, m in enumerate(models):
            if pd.notna(w.loc[m, a]):
                xs.append(i + (j - 1.5) * bw)
                ys.append(w.loc[m, a])
                es.append(sd.loc[m, a])
        ax.bar(xs, ys, bw, yerr=es, color=col, label=lab, capsize=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABEL[m].replace(" + ", "\n+ ") for m in models], fontsize=9)
    ax.set_ylabel("Recall@5% (mean of 5 folds)")
    ax.set_title("Fraudster lowers declared consumption by ≤30%: recall in the top-5% list")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/figures/F6_robustness.png", dpi=300)


if __name__ == "__main__":
    main()
