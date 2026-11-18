# Hypothesis tests and Spearman redundancy pairs for the v1 features, on training clients only.
# With about 108,000 clients nearly every test is significant, so features are ranked by effect size.
# Run: python3 -m src.features.feature_stats

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.split import read_splits
from src.viz import style

CORR_LIMIT = 0.9
# Cohen (1988) cut-offs for small, medium and large effects.
LABELS = {
    "cohen_d": (0.2, 0.5, 0.8),
    "rank_biserial": (0.1, 0.3, 0.5),
    "cramers_v": (0.1, 0.3, 0.5),
    "eta_squared": (0.01, 0.06, 0.14),
}


def effect_label(kind, value):
    if kind not in LABELS or not np.isfinite(value):
        return ""
    small, medium, large = LABELS[kind]
    v = abs(value)
    return (
        "large"
        if v >= large
        else "medium"
        if v >= medium
        else "small"
        if v >= small
        else "negligible"
    )


def row(feature, test, statistic, p_value, effect_kind, effect_size):
    return {
        "feature": feature,
        "test": test,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_measure": effect_kind,
        "effect_size": float(effect_size),
        "effect_label": effect_label(effect_kind, effect_size),
    }


def entropy(p):
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def categorical_tests(feature, x, y):
    x = x.astype("string").fillna("missing")
    table = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    # Information gain in bits: H(target) - H(target | feature).
    prior = entropy(table.sum(axis=0).to_numpy() / n)
    conditional = sum(
        (counts.sum() / n) * entropy(counts.to_numpy() / counts.sum())
        for _, counts in table.iterrows()
    )
    return [
        row(feature, "chi_square", chi2, p, "cramers_v", v),
        row(
            feature,
            "information_gain",
            prior - conditional,
            np.nan,
            "information_gain_bits",
            prior - conditional,
        ),
    ]


def numeric_tests(feature, x, y):
    fraud, other = x[y == 1].dropna(), x[y == 0].dropna()
    if len(fraud) < 2 or len(other) < 2 or x.dropna().nunique() < 2:
        return []
    t, p_t = stats.ttest_ind(fraud, other, equal_var=False)
    pooled = np.sqrt(
        ((len(fraud) - 1) * fraud.var() + (len(other) - 1) * other.var())
        / (len(fraud) + len(other) - 2)
    )
    d = (fraud.mean() - other.mean()) / pooled if pooled > 0 else np.nan
    u, p_u = stats.mannwhitneyu(fraud, other, alternative="two-sided")
    # Positive rank-biserial: fraud clients tend to have larger values.
    r = 2 * u / (len(fraud) * len(other)) - 1
    return [
        row(feature, "welch_t", t, p_t, "cohen_d", d),
        row(feature, "mann_whitney_u", u, p_u, "rank_biserial", r),
    ]


def district_anova(x, district):
    groups = [g.dropna().to_numpy() for _, g in x.groupby(district)]
    groups = [g for g in groups if len(g) > 1]
    f, p = stats.f_oneway(*groups)
    values = np.concatenate(groups)
    total = ((values - values.mean()) ** 2).sum()
    between = sum(len(g) * (g.mean() - values.mean()) ** 2 for g in groups)
    return row("cons_pm_median by district", "anova", f, p, "eta_squared", between / total)


def is_categorical(x):
    return not pd.api.types.is_numeric_dtype(x) or x.dropna().nunique() <= 2


def run_tests(features, y):
    rows = []
    for feature in features.columns:
        x = features[feature]
        tests = categorical_tests if is_categorical(x) else numeric_tests
        rows += tests(feature, x.reset_index(drop=True), y)
    rows.append(district_anova(features["cons_pm_median"], features["district"]))
    out = pd.DataFrame(rows)
    out["rank"] = (
        out.assign(a=out["effect_size"].abs())
        .groupby("test")["a"]
        .rank(ascending=False, method="min")
    )
    out = out.sort_values(["test", "rank"]).reset_index(drop=True)
    out["rank"] = out["rank"].astype("int64")
    return out


def redundant_pairs(rho, strength):
    # Of each pair above the limit, drop the feature with the smaller |rank-biserial|.
    cols = rho.columns
    upper = np.triu(np.abs(rho.to_numpy()) > CORR_LIMIT, k=1)
    rows = []
    for i, j in zip(*np.nonzero(upper)):
        a, b = cols[i], cols[j]
        drop = a if strength.get(a, 0) < strength.get(b, 0) else b
        rows.append({"feature_a": a, "feature_b": b, "spearman_rho": rho.iat[i, j], "drop": drop})
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman_rho", "drop"])
    return out.sort_values("spearman_rho", key=np.abs, ascending=False).reset_index(drop=True)


def plot_heatmap(rho, path):
    style.apply()
    n = len(rho)
    fig, ax = plt.subplots(figsize=(0.19 * n + 3, 0.19 * n + 2))
    image = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n), rho.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n), rho.columns, fontsize=6)
    ax.grid(False)
    ax.set_title("Spearman correlation between v1 features (training clients)")
    fig.colorbar(image, ax=ax, shrink=0.6, label="Spearman rho")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Hypothesis tests and correlation redundancy.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features_v1_with_inspection.parquet"),
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    try:
        clients = pd.read_parquet(args.interim / "client.parquet", columns=["client_id", "target"])
        train, _, _ = read_splits(args.split_dir, clients)
        data = train.merge(clients.astype({"client_id": "string"}), on="client_id").merge(
            pd.read_parquet(args.features).astype({"client_id": "string"}), on="client_id"
        )
        y = data.pop("target").to_numpy()
        features = data.drop(columns=["client_id"])
        tests = run_tests(features, y)

        numeric = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
        rho = features[numeric].corr(method="spearman")
        mw = tests[tests["test"] == "mann_whitney_u"].set_index("feature")["effect_size"].abs()
        pairs = redundant_pairs(rho, mw)

        args.tables.mkdir(parents=True, exist_ok=True)
        args.figures.mkdir(parents=True, exist_ok=True)
        tests.to_csv(args.tables / "stat_tests.csv", index=False)
        pairs.to_csv(args.tables / "corr_pairs.csv", index=False)
        plot_heatmap(rho, args.figures / "corr_heatmap.png")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"feature_stats: {exc}\n")
    print(f"{len(tests)} tests on {len(y):,} training clients -> {args.tables / 'stat_tests.csv'}")
    print(f"{len(pairs)} pairs with |rho| > {CORR_LIMIT} -> {args.tables / 'corr_pairs.csv'}")


if __name__ == "__main__":
    main()
