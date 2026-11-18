# Plot fraud rate by the year of a client's last meter replacement (f12_change_year.csv); the
# jump from 2013 is why meter and tariff counts are left out of features_v4.
# Run: python3 -m src.viz.leakage_fig

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.viz import style

FIRST_YEAR = 2005


def main():
    style.apply()
    t = pd.read_csv("results/tables/f12_change_year.csv", dtype={"last_change_year": str})
    t = t[t["field"] == "counter_number"]
    none = t.loc[t["last_change_year"] == "no change", "fraud_rate"].iloc[0]
    t = t[t["last_change_year"] != "no change"].assign(
        year=lambda d: d.last_change_year.astype(int)
    )
    t = t[t["year"] >= FIRST_YEAR]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(
        t["year"], t["fraud_rate"] * 100, color=style.SINGLE, linewidth=2, marker="o", markersize=5
    )
    ax.axhline(none * 100, color=style.TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(
        t["year"].max(),
        none * 100 - 0.8,
        f"clients with no meter change: {none:.1%}",
        color=style.TEXT_SECONDARY,
        fontsize=9,
        ha="right",
        va="top",
    )
    for year in (2012, 2018):
        r = t[t["year"] == year].iloc[0]
        ax.annotate(
            f"{r.fraud_rate:.0%}",
            (year, r.fraud_rate * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=style.TEXT,
            fontsize=9,
        )
    ax.set_xlabel("Year of the client's last meter replacement")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_ylim(0, max(t["fraud_rate"]) * 100 * 1.2)
    ax.set_title(
        "Meter replacements after 2012 mark fraud clients: likely after detection", loc="left"
    )
    fig.savefig("results/figures/meter_change_year.png")
    print("results/figures/meter_change_year.png")


if __name__ == "__main__":
    main()
