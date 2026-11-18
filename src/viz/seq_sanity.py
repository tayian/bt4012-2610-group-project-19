# Sanity plot of the monthly sequences: 10 fraud and 10 non-fraud training clients, plus each
# class's median trajectory, and a small table of summary numbers.
# Run: python3 -m src.viz.seq_sanity

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 4012


def main():
    seq = np.load("data/processed/seq.npy", mmap_mode="r")
    ids = np.load("data/processed/seq_client_ids.npy")
    cli = pd.read_parquet("data/interim/client.parquet").set_index("client_id")
    tr = pd.read_csv("data/splits/train_ids.csv")["client_id"].astype(str)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    y = cli.loc[tr, "target"].values
    rng = np.random.default_rng(SEED)
    # At least 60 observed months, so the lines are readable.
    tr_idx = pos[tr].values
    obs = (seq[tr_idx, :, 2] == 0).sum(1)
    elig = obs >= 60
    fr = rng.choice(tr_idx[(y == 1) & elig], 10, replace=False)
    nf = rng.choice(tr_idx[(y == 0) & elig], 10, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=False)
    x = np.arange(-119, 1)
    for ax, sel, title, col in [
        (axes[0], nf, "10 non-fraud clients", "#4C72B0"),
        (axes[1], fr, "10 fraud clients", "#C44E52"),
    ]:
        for i in sel:
            v = np.where(seq[i, :, 2] == 1, np.nan, seq[i, :, 0])
            ax.plot(x, v, lw=0.9, alpha=0.75, color=col)
        ax.set_title(title)
        ax.set_xlabel("months before last invoice")
        ax.set_ylabel("log1p(ELEC kWh / month)")
    for lab, col, name in [(0, "#4C72B0", "non-fraud"), (1, "#C44E52", "fraud")]:
        s = seq[tr_idx[y == lab]]
        v = np.where(s[:, :, 2] == 1, np.nan, s[:, :, 0])
        axes[2].plot(x, np.nanmedian(v, 0), color=col, label=f"{name} median")
        axes[2].fill_between(
            x, np.nanpercentile(v, 25, 0), np.nanpercentile(v, 75, 0), color=col, alpha=0.15
        )
    axes[2].set_title("All training clients: median and IQR")
    axes[2].set_xlabel("months before last invoice")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig("results/figures/seq_sanity.png", dpi=150)

    # Summary numbers per class.
    s = seq[tr_idx]
    obs_m = (s[:, :, 2] == 0).sum(1)
    elec = np.where(s[:, :, 2] == 1, np.nan, s[:, :, 0])
    zero_share = np.nanmean(np.where(np.isnan(elec), np.nan, elec == 0), 1)
    rows = []
    for lab in (0, 1):
        m = y == lab
        rows.append(
            {
                "target": lab,
                "n": int(m.sum()),
                "observed_months_median": float(np.median(obs_m[m])),
                "elec_level_median": float(np.nanmedian(elec[m])),
                "zero_month_share_mean": float(np.nanmean(zero_share[m])),
                "elec_std_over_time_median": float(np.nanmedian(np.nanstd(elec[m], 1))),
            }
        )
    pd.DataFrame(rows).to_csv("results/tables/seq_sanity_stats.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
