# Sequence tensor (section 6): (n_clients, 120, 3) float32.
#
# Channels
#   0  ELEC  log1p(monthly consumption, kWh) summed over all ELEC counters
#   1  GAZ   log1p(monthly consumption) summed over all GAZ counters
#   2  MASK  1.0 if no invoice of any type covers that month, else 0.0
#
# Method
#   * An invoice dated in month t with months_number m covers months t-m+1 .. t; its
#     consumption is spread evenly (cons / m per month). m is clipped to [1, 24] for
#     spreading (m=0 or absurd values such as 636,624 exist).
#   * Each client is right-aligned on the month of its own last invoice, so the window is the
#     client's last 120 calendar months (most clients end in 2019, but ~15% stop earlier; a
#     global window would leave them blank).
#   * Months before a client's first invoice are masked (MASK=1, values 0).
#
# Works on any invoice dataframe with the raw columns; by default it reads the clean parquet:
#   python3 -m src.features.sequence --invoices data/interim/invoice_clean.parquet

import argparse
import time

import numpy as np
import pandas as pd

LEVELS = [
    "consommation_level_1",
    "consommation_level_2",
    "consommation_level_3",
    "consommation_level_4",
]
T = 120
MAX_SPREAD = 24


def build_sequence(inv: pd.DataFrame, client_ids=None, T=T):
    # Return (seq float32 (n, T, 3), ids np.array[str]).
    inv = inv[["client_id", "invoice_date", "counter_type", "months_number"] + LEVELS]
    if client_ids is None:
        client_ids = np.sort(inv["client_id"].astype(str).unique())
    client_ids = np.asarray(client_ids).astype(str)
    idx_map = pd.Series(np.arange(len(client_ids)), index=client_ids)

    cid = idx_map.reindex(inv["client_id"].astype(str).values).values
    keep = ~np.isnan(cid)
    inv = inv[keep]
    cid = cid[keep].astype(np.int64)

    d = pd.to_datetime(inv["invoice_date"])
    ym = (d.dt.year * 12 + d.dt.month - 1).to_numpy(np.int64)
    m = np.clip(inv["months_number"].to_numpy(np.int64), 1, MAX_SPREAD)
    cons = inv[LEVELS].sum(axis=1).to_numpy(np.float64)
    is_gaz = (inv["counter_type"].astype(str).to_numpy() == "GAZ").astype(np.int64)

    anchor = np.full(len(client_ids), -1, np.int64)
    np.maximum.at(anchor, cid, ym)
    first = np.full(len(client_ids), np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, cid, ym - m + 1)

    # explode each invoice into the months it covers
    rep = np.repeat(np.arange(len(m)), m)
    offset = np.arange(len(rep)) - np.repeat(np.cumsum(m) - m, m)  # 0..m-1
    month = ym[rep] - offset
    pos = (T - 1) - (anchor[cid[rep]] - month)
    ok = pos >= 0
    rep, pos = rep[ok], pos[ok]
    val = (cons / m)[rep]

    raw = np.zeros((len(client_ids), T, 2), np.float64)
    cov = np.zeros((len(client_ids), T), np.int8)
    np.add.at(raw, (cid[rep], pos, is_gaz[rep]), val)
    cov[cid[rep], pos] = 1

    seq = np.zeros((len(client_ids), T, 3), np.float32)
    seq[:, :, :2] = np.log1p(np.clip(raw, 0, None)).astype(np.float32)
    seq[:, :, 2] = 1.0 - cov
    return seq, client_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--invoices", default="data/interim/invoice_clean.parquet")
    p.add_argument("--clients", default="data/interim/client.parquet")
    p.add_argument("--out", default="data/processed/")
    a = p.parse_args()
    t = time.time()
    inv = pd.read_parquet(a.invoices)
    ids = pd.read_parquet(a.clients)["client_id"].astype(str).values
    seq, ids = build_sequence(inv, ids)
    np.save(a.out + "seq.npy", seq)
    np.save(a.out + "seq_client_ids.npy", ids)
    print(f"seq {seq.shape}  masked share {seq[:, :, 2].mean():.3f}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
