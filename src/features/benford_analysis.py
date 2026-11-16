# Benford's law on the invoice data (training clients only).
#
#   1. First digit of positive per-invoice consumption (cons), fraud vs non-fraud: observed vs
#      Benford shares, chi-square goodness of fit and MAD with Nigrini's conformity band.
#   2. The same per district and per district x class, ranked by MAD.
#   3. Drill-down on the worst district x class group: first digit, then first two digits.
#   4. Last digit of new_index: share ending in 0 or 5 against the 20% expected, by class.
#
# Outputs: results/tables/benford_*.csv, results/figures/benford_first_digit.png and
# results/figures/benford_two_digit.png. The test clients are never read.

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.data.load import LEVELS  # noqa: E402
from src.features.behavioural import BENFORD_MIN_READINGS, leading_digits  # noqa: E402
from src.viz import style  # noqa: E402

# Nigrini (2012) MAD bands: upper limits of close, acceptable and marginally acceptable
# conformity; anything above is nonconformity.
NIGRINI = {
    1: [(0.006, "close"), (0.012, "acceptable"), (0.015, "marginal"), (np.inf, "nonconformity")],
    2: [
        (0.0012, "close"),
        (0.0018, "acceptable"),
        (0.0022, "marginal"),
        (np.inf, "nonconformity"),
    ],
}
MIN_GROUP_INVOICES = 1000  # smaller groups are left out of the district ranking
ROUND_EXPECTED = 0.2


def expected_shares(n_digits):
    d = np.arange(10 ** (n_digits - 1), 10**n_digits)
    return pd.Series(np.log10(1 + 1 / d), index=d)


def band(mad, n_digits):
    return next(name for limit, name in NIGRINI[n_digits] if mad <= limit)


def digit_test(values, n_digits=1):
    # Observed counts of the leading digits of positive values (values below 10 are left out of
    # the two-digit test), chi-square goodness of fit against Benford and the MAD.
    v = np.asarray(values)
    v = v[v >= 10 ** (n_digits - 1)]
    expected = expected_shares(n_digits)
    counts = (
        pd.Series(leading_digits(v, n_digits)).value_counts().reindex(expected.index, fill_value=0)
    )
    n = int(counts.sum())
    observed = counts / n
    chi2, p = stats.chisquare(counts, expected * n)
    mad = float((observed - expected).abs().mean())
    summary = {
        "n": n,
        "chi2": float(chi2),
        "dof": len(expected) - 1,
        "p_value": float(p),
        "mad": mad,
        "nigrini_band": band(mad, n_digits),
    }
    detail = pd.DataFrame(
        {"digit": expected.index, "count": counts.to_numpy(), "observed": observed.to_numpy()}
    )
    detail["expected"] = expected.to_numpy()
    detail["difference"] = detail["observed"] - detail["expected"]
    return summary, detail


def expected_mad(n, n_digits=1):
    # Mean |observed - expected| for a sample of n values that follow Benford exactly
    # (normal approximation of each digit's share).
    p = expected_shares(n_digits).to_numpy()
    return float(np.mean(np.sqrt(2 * p * (1 - p) / (np.pi * n))))


def load(interim, split_dir):
    clients = pd.read_parquet(
        Path(interim) / "client.parquet", columns=["client_id", "district", "target"]
    )
    train = pd.read_csv(Path(split_dir) / "train_ids.csv", dtype={"client_id": "string"})
    clients = clients[clients["client_id"].isin(train["client_id"])]
    inv = pd.read_parquet(
        Path(interim) / "invoice_clean.parquet",
        columns=["client_id", "cons", "new_index", *LEVELS],
    )
    inv = inv[inv["client_id"].isin(clients["client_id"])]
    inv = inv.merge(clients, on="client_id", how="left", validate="many_to_one")
    inv["class"] = inv["target"].map(style.CLASS_LABELS)
    return inv


def tests_by(inv, groups, n_digits=1, min_n=0):
    summaries, details = [], []
    for name, part in inv.groupby(groups, sort=True):
        name = name if isinstance(name, tuple) else (name,)
        summary, detail = digit_test(part["cons"][part["cons"] > 0], n_digits)
        if summary["n"] < min_n:
            continue
        keys = dict(zip(groups, name))
        summaries.append({**keys, **summary})
        details.append(detail.assign(**keys))
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def last_digit_test(inv):
    # Readings of 0 are left out (a new meter, not a reading). "distinct readings" also drops
    # bills with zero consumption, whose new_index repeats the previous one.
    rows = []
    for subset, mask in (
        ("new_index > 0", inv["new_index"] > 0),
        ("new_index > 0 and cons > 0", (inv["new_index"] > 0) & (inv["cons"] > 0)),
    ):
        for label, part in inv[mask].groupby("class", sort=False):
            last = part["new_index"].to_numpy() % 10
            counts = np.bincount(last, minlength=10)
            n = int(counts.sum())
            k = int(counts[0] + counts[5])
            lo, hi = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
            chi2, p = stats.chisquare(counts)
            rows.append(
                {
                    "subset": subset,
                    "class": label,
                    "n": n,
                    "share_0_or_5": k / n,
                    "ci_low": lo,
                    "ci_high": hi,
                    "expected": ROUND_EXPECTED,
                    "binom_p_value": stats.binomtest(k, n, ROUND_EXPECTED).pvalue,
                    "chi2_uniform": float(chi2),
                    "chi2_p_value": float(p),
                    **{f"share_digit_{d}": counts[d] / n for d in range(10)},
                }
            )
    return pd.DataFrame(rows)


def caveats(inv):
    # Numbers behind the caveats: how often tariff level 1 is full (cons then starts at the band
    # limit), and how noisy a per-client MAD is with a typical number of readings.
    positive = inv[inv["cons"] > 0]
    per_client = positive.groupby("client_id").size()
    median_n = int(per_client.median())
    return pd.DataFrame(
        [
            ("positive invoices", len(positive)),
            (
                "share with tariff level 1 full (level 2+ > 0)",
                (positive[LEVELS[1:]].sum(1) > 0).mean(),
            ),
            ("median positive invoices per client", median_n),
            (
                "share of clients below the F7 minimum of 20",
                (per_client < BENFORD_MIN_READINGS).mean(),
            ),
            ("expected MAD of a perfect Benford sample of the median size", expected_mad(median_n)),
            ("expected MAD of a perfect Benford sample of 20", expected_mad(BENFORD_MIN_READINGS)),
        ],
        columns=["statistic", "value"],
    )


def plot_first_digit(detail, summary, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    digits = np.arange(1, 10)
    width = 0.38
    for i, label in enumerate(["Non-fraud", "Fraud"]):
        d = detail[detail["class"] == label].set_index("digit").reindex(digits)
        s = summary.set_index("class").loc[label]
        ax.bar(
            digits + (i - 0.5) * width,
            100 * d["observed"],
            width=width - 0.04,
            color=style.CLASS_COLORS[i],
            label=f"{label}: n = {s['n']:,}, MAD = {s['mad']:.4f} ({s['nigrini_band']})",
        )
    ax.plot(
        digits,
        100 * expected_shares(1).to_numpy(),
        color=style.TEXT,
        marker="o",
        markersize=5,
        lw=1.2,
        label="Benford expected",
    )
    ax.set_xticks(digits)
    ax.set_xlabel("first digit of per-invoice consumption (cons > 0)")
    ax.set_ylabel("share of invoices (%)")
    ax.set_title("First digits of consumption against Benford's law (training clients)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def plot_drill_down(first, two, mads, title, path):
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2.6]}, constrained_layout=True
    )
    for ax, detail, n_digits in ((a1, first, 1), (a2, two, 2)):
        ax.bar(
            detail["digit"],
            100 * detail["observed"],
            width=0.8,
            color=style.SINGLE,
            label="observed",
        )
        ax.plot(
            detail["digit"],
            100 * detail["expected"],
            color=style.TEXT,
            lw=1.2,
            marker="o" if n_digits == 1 else None,
            markersize=4,
            label="Benford expected",
        )
        ax.set_ylabel("share of invoices (%)")
    a1.set_xticks(range(1, 10))
    a1.set_xlabel("first digit")
    a1.set_title(f"First digit (MAD {mads[0]:.4f})")
    a2.set_xticks(range(10, 100, 10))
    a2.set_xlim(9, 100)
    a2.set_xlabel("first two digits")
    a2.set_title(f"First two digits, 10 to 99 (MAD {mads[1]:.4f})")
    a2.legend(loc="upper right", fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def run(
    interim="data/interim",
    split_dir="data/splits",
    tables="results/tables",
    figures="results/figures",
):
    style.apply()
    tables, figures = Path(tables), Path(figures)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    inv = load(interim, split_dir)

    by_class, class_detail = tests_by(inv, ["class"])
    by_class = by_class.sort_values("class", ascending=False, ignore_index=True)

    by_district, _ = tests_by(inv, ["district"], min_n=MIN_GROUP_INVOICES)
    by_district_class, _ = tests_by(inv, ["district", "class"], min_n=MIN_GROUP_INVOICES)
    district = pd.concat(
        [by_district.assign(**{"class": "All"}), by_district_class], ignore_index=True
    ).sort_values("mad", ascending=False, ignore_index=True)
    district.insert(0, "rank", np.arange(1, len(district) + 1))

    worst = by_district_class.sort_values("mad", ascending=False).iloc[0]
    group = inv[(inv["district"] == worst["district"]) & (inv["class"] == worst["class"])]
    positive = group["cons"][group["cons"] > 0]
    s1, d1 = digit_test(positive, 1)
    s2, d2 = digit_test(positive, 2)
    drill = pd.DataFrame(
        [{"test": "first digit", **s1}, {"test": "first two digits", **s2}]
    ).assign(district=worst["district"], **{"class": worst["class"]})
    drill_detail = pd.concat(
        [d1.assign(test="first digit"), d2.assign(test="first two digits")], ignore_index=True
    )

    last = last_digit_test(inv)
    notes = caveats(inv)

    by_class.to_csv(tables / "benford_first_digit.csv", index=False)
    class_detail.to_csv(tables / "benford_first_digit_shares.csv", index=False)
    district.to_csv(tables / "benford_district.csv", index=False)
    drill.to_csv(tables / "benford_drilldown.csv", index=False)
    drill_detail.to_csv(tables / "benford_drilldown_shares.csv", index=False)
    last.to_csv(tables / "benford_last_digit.csv", index=False)
    notes.to_csv(tables / "benford_caveats.csv", index=False)

    plot_first_digit(class_detail, by_class, figures / "benford_first_digit.png")
    plot_drill_down(
        d1,
        d2,
        (s1["mad"], s2["mad"]),
        f"District {worst['district']}, {worst['class'].lower()} clients: the group furthest from Benford",
        figures / "benford_two_digit.png",
    )
    return {
        "by_class": by_class,
        "district": district,
        "drill": drill,
        "drill_detail": drill_detail,
        "last_digit": last,
        "caveats": notes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benford and last-digit tests on training invoices."
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    try:
        out = run(args.interim, args.split_dir, args.tables, args.figures)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"benford: {exc}\n")
    pd.set_option("display.width", 160)
    for name in ("by_class", "district", "drill", "caveats"):
        print(out[name].to_string(index=False), end="\n\n")
    cols = ["subset", "class", "n", "share_0_or_5", "ci_low", "ci_high", "binom_p_value"]
    print(out["last_digit"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
