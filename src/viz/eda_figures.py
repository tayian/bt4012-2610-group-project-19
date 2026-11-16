# Second round of exploratory analysis (training clients only; the test set is never read).
#
#   eda_outliers.csv              boxplot-rule and |z| > 3 outlier counts by class
#   eda_validity_by_class.csv     F6 invalid readings (impossible combinations) by class
#   eda_fraud_rate_heatmap.png    fraud rate by district x client_catg, with client counts (F2)
#   eda_cons_over_time.png        median monthly consumption over time by class, IQR band (F1)
#   eda_top_features_boxplots.png the six features with the largest effect size by class

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

from src.viz import style  # noqa: E402

IQR_K = 1.5
Z_LIMIT = 3.0
N_TOP = 6
START_YEAR = 2005  # earlier months have too few invoices for a stable median
MIN_PERIOD_INVOICES = 500  # per class; thinner quarters are not plotted
INVOICE_COLUMNS = [
    "client_id",
    "invoice_date",
    "counter_type",
    "cons",
    "cons_pm",
    "months_valid",
    "flag_index_backwards",
    "flag_index_rollover",
    "flag_recon_mismatch",
    "flag_bad_months_number",
    "flag_partial_year",
]


def load(interim, split_dir, features):
    train = pd.read_csv(Path(split_dir) / "train_ids.csv", dtype={"client_id": "string"})
    clients = pd.read_parquet(Path(interim) / "client.parquet")
    clients = clients[clients["client_id"].isin(train["client_id"])].reset_index(drop=True)
    inv = pd.read_parquet(Path(interim) / "invoice_clean.parquet", columns=INVOICE_COLUMNS)
    inv = inv[inv["client_id"].isin(clients["client_id"])]
    inv = inv.merge(clients[["client_id", "target"]], on="client_id", validate="many_to_one")
    feats = pd.read_parquet(features)
    feats = clients[["client_id", "target"]].merge(feats, on="client_id", validate="one_to_one")
    return clients, inv, feats


def top_features(stat_tests, n=N_TOP):
    t = pd.read_csv(stat_tests)
    t = t[t["test"].eq("mann_whitney_u")].copy()
    t["abs_effect"] = t["effect_size"].abs()
    return t.nlargest(n, "abs_effect")[["feature", "effect_size", "effect_label"]]


def outlier_table(feats, inv, columns):
    # Fences and z-scores use all training clients (or invoices) pooled, so both classes are
    # judged against the same cut-offs.
    series = {f"{c} (client)": (feats[c], feats["target"]) for c in columns}
    series["cons_pm (invoice)"] = (inv["cons_pm"], inv["target"])
    rows = []
    for name, (x, y) in series.items():
        ok = x.notna()
        x, y = x[ok].astype(float), y[ok]
        q1, q3 = x.quantile([0.25, 0.75])
        low, high = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
        z = (x - x.mean()) / x.std()
        iqr_out, z_out = (x < low) | (x > high), z.abs() > Z_LIMIT
        row = {"variable": name, "lower_fence": low, "upper_fence": high}
        for t in (0, 1):
            label = style.CLASS_LABELS[t].lower().replace("-", "_")
            m = y.eq(t)
            row[f"n_{label}"] = int(m.sum())
            row[f"iqr_outliers_{label}"] = int(iqr_out[m].sum())
            row[f"iqr_share_{label}"] = iqr_out[m].mean()
            row[f"z3_outliers_{label}"] = int(z_out[m].sum())
            row[f"z3_share_{label}"] = z_out[m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def validity_table(inv):
    flags = {
        "index backwards (not a meter rollover)": inv["flag_index_backwards"]
        & ~inv["flag_index_rollover"],
        "index difference x coefficient != cons": inv["flag_recon_mismatch"],
        "months_number <= 0 or > 12": inv["flag_bad_months_number"],
        "impossible period (> 120 months)": inv["months_valid"].isna(),
    }
    flags["any of the above"] = pd.concat(flags, axis=1).any(axis=1)
    rows = []
    for name, flag in flags.items():
        per_client = flag.groupby(inv["client_id"]).any()
        target = inv.groupby("client_id")["target"].first().reindex(per_client.index)
        table = pd.crosstab(target, per_client).reindex(columns=[False, True], fill_value=0)
        p = stats.chi2_contingency(table)[1] if table[True].sum() else np.nan
        row = {"check": name}
        for t in (0, 1):
            label = style.CLASS_LABELS[t].lower().replace("-", "_")
            m = inv["target"].eq(t)
            row[f"invoices_{label}"] = int(flag[m].sum())
            row[f"per_10k_invoices_{label}"] = 1e4 * flag[m].mean()
            row[f"clients_{label}"] = int(per_client[target.eq(t)].sum())
            row[f"client_share_{label}"] = per_client[target.eq(t)].mean()
        row["client_share_chi2_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(clients, path):
    g = clients.groupby(["district", "client_catg"])["target"].agg(["mean", "size"])
    rate = g["mean"].unstack()
    n = g["size"].unstack()
    cmap = LinearSegmentedColormap.from_list("fraud", ["#ffffff", style.FRAUD])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(100 * rate.to_numpy(float), cmap=cmap, vmin=0, aspect="auto")
    top = np.nanmax(100 * rate.to_numpy(float))
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            r, k = rate.iat[i, j], n.iat[i, j]
            if pd.isna(k):
                ax.text(j, i, "no clients", ha="center", va="center", fontsize=8.5)
                continue
            dark = 100 * r > 0.6 * top
            ax.text(
                j,
                i,
                f"{r:.1%}\nn = {int(k):,}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if dark else style.TEXT,
            )
    ax.set_xticks(range(rate.shape[1]), [f"catg {c}" for c in rate.columns])
    ax.set_yticks(range(rate.shape[0]), [f"district {d}" for d in rate.index])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    overall = clients["target"].mean()
    ax.set_title(f"Fraud rate by district and client category (overall {overall:.1%})")
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("fraud rate (%)")
    bar.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    out = pd.DataFrame({"fraud_rate": g["mean"], "n_clients": g["size"]}).reset_index()
    out["n_fraud"] = (out["fraud_rate"] * out["n_clients"]).round().astype(int)
    return out


def monthly_quantiles(inv):
    # Per calendar quarter and class: median and quartiles of cons_pm over ELEC invoices.
    # Quarters rather than months, because a few winter months hold mostly annual catch-up bills.
    elec = inv[inv["counter_type"].eq("ELEC") & inv["cons_pm"].notna()]
    elec = elec[elec["invoice_date"].dt.year >= START_YEAR]
    month = elec["invoice_date"].dt.to_period("Q").dt.to_timestamp()
    g = elec.groupby([month.rename("quarter"), "target"])["cons_pm"]
    out = g.quantile([0.25, 0.5, 0.75]).unstack()
    out.columns = ["q1", "median", "q3"]
    out["n_invoices"] = g.size()
    out = out.reset_index()
    return out[out["n_invoices"] >= MIN_PERIOD_INVOICES]


def plot_over_time(monthly, partial_from, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for t in (0, 1):
        m = monthly[monthly["target"].eq(t)]
        ax.fill_between(
            m["quarter"], m["q1"], m["q3"], color=style.CLASS_COLORS[t], alpha=0.18, lw=0
        )
        ax.plot(
            m["quarter"],
            m["median"],
            color=style.CLASS_COLORS[t],
            lw=1.6,
            label=f"{style.CLASS_LABELS[t]}: median (band = 25th to 75th percentile)",
        )
    if partial_from is not None:
        ax.axvspan(
            partial_from,
            monthly["quarter"].max() + pd.offsets.QuarterEnd(0),
            color=style.GRID,
            alpha=0.6,
            lw=0,
        )
        ax.annotate(
            "2019: incomplete",
            (partial_from, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=style.TEXT_SECONDARY,
        )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("consumption per month (cons_pm)")
    ax.set_xlabel("invoice quarter (electricity invoices, training clients)")
    ax.set_title("Class medians track each other until about 2014, then fraud clients bill more")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def plot_boxplots(feats, top, path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), constrained_layout=True)
    for ax, (_, r) in zip(axes.flat, top.iterrows()):
        data = [feats.loc[feats["target"].eq(t), r["feature"]].dropna() for t in (0, 1)]
        parts = ax.boxplot(
            data,
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": style.TEXT, "lw": 1.4},
            whiskerprops={"color": style.TEXT_SECONDARY},
            capprops={"color": style.TEXT_SECONDARY},
        )
        for box, t in zip(parts["boxes"], (0, 1)):
            box.set(facecolor=style.CLASS_COLORS[t], edgecolor=style.TEXT_SECONDARY, alpha=0.85)
        ax.set_xticks([1, 2], [style.CLASS_LABELS[0], style.CLASS_LABELS[1]])
        if min(d.min() for d in data) >= 0 and max(d.quantile(0.95) for d in data) > 50:
            ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"{r['feature']}\nrank-biserial r = {r['effect_size']:.2f} ({r['effect_label']})",
            fontsize=10,
        )
    fig.suptitle(
        "Six features with the largest class difference (r > 0: fraud higher; whiskers 1.5 IQR, outliers hidden)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(path)
    plt.close(fig)


def run(
    interim="data/interim",
    split_dir="data/splits",
    features="data/processed/features_v1.parquet",
    stat_tests="results/tables/stat_tests.csv",
    tables="results/tables",
    figures="results/figures",
):
    style.apply()
    tables, figures = Path(tables), Path(figures)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    clients, inv, feats = load(interim, split_dir, features)
    top = top_features(stat_tests)

    outliers = outlier_table(feats, inv, top["feature"].tolist())
    validity = validity_table(inv)
    heat = plot_heatmap(clients, figures / "eda_fraud_rate_heatmap.png")
    monthly = monthly_quantiles(inv)
    partial = inv.loc[inv["flag_partial_year"], "invoice_date"].min()
    partial = None if pd.isna(partial) else partial.to_period("Y").to_timestamp()
    plot_over_time(monthly, partial, figures / "eda_cons_over_time.png")
    plot_boxplots(feats, top, figures / "eda_top_features_boxplots.png")

    outliers.to_csv(tables / "eda_outliers.csv", index=False)
    validity.to_csv(tables / "eda_validity_by_class.csv", index=False)
    heat.to_csv(tables / "eda_fraud_rate_heatmap.csv", index=False)
    monthly.to_csv(tables / "eda_cons_over_time.csv", index=False)
    return {"outliers": outliers, "validity": validity, "heatmap": heat, "top": top}


def main():
    parser = argparse.ArgumentParser(description="EDA part 2: outliers, validity and figures.")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_v1.parquet"))
    parser.add_argument("--stat-tests", type=Path, default=Path("results/tables/stat_tests.csv"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    try:
        out = run(
            args.interim, args.split_dir, args.features, args.stat_tests, args.tables, args.figures
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"eda_figures: {exc}\n")
    pd.set_option("display.width", 200)
    for name, table in out.items():
        print(table.to_string(index=False), end="\n\n")


if __name__ == "__main__":
    main()
