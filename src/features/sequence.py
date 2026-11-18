# Build the monthly sequence tensor (n_clients, 120, 3): log1p ELEC and GAZ consumption and a
# mask that is 1 where no invoice covers the month. Each invoice is spread evenly over its months.
# Run: python3 -m src.features.sequence

import argparse
import time

import numpy as np
import pandas as pd

from src.data.load import LEVELS

T = 120
COLUMNS = ["client_id", "invoice_date", "counter_type", "months_valid", *LEVELS]


def _usable(inv):
    # Rows with a valid billing period (months_valid comes from src/data/clean.py).
    inv = inv[COLUMNS]
    return inv[inv["months_valid"].notna()]


def last_invoice_month(inv, client_ids):
    # Calendar month (0 = January) of each client's last usable invoice; -1 if none.
    inv = _usable(inv)
    last = pd.to_datetime(inv["invoice_date"]).groupby(inv["client_id"].astype(str)).max()
    month = last.dt.month.reindex(np.asarray(client_ids).astype(str)) - 1
    return month.fillna(-1).to_numpy(np.int64)


def build_sequence(inv: pd.DataFrame, client_ids=None, T=T):
    # Return (seq float32 (n, T, 3), ids np.array[str]).
    inv = _usable(inv)
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
    m = inv["months_valid"].to_numpy(np.int64)
    cons = inv[LEVELS].sum(axis=1).to_numpy(np.float64)
    is_gaz = (inv["counter_type"].astype(str).to_numpy() == "GAZ").astype(np.int64)

    # Each client is right-aligned on its own last invoice, since many clients stop billing
    # early and a fixed calendar window would leave them as empty rows.
    anchor = np.full(len(client_ids), -1, np.int64)
    np.maximum.at(anchor, cid, ym)

    # Repeat each invoice once for every month it covers.
    rep = np.repeat(np.arange(len(m)), m)
    offset = np.arange(len(rep)) - np.repeat(np.cumsum(m) - m, m)
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
    inv = pd.read_parquet(a.invoices, columns=COLUMNS)
    ids = pd.read_parquet(a.clients)["client_id"].astype(str).values
    seq, ids = build_sequence(inv, ids)
    np.save(a.out + "seq.npy", seq)
    np.save(a.out + "seq_client_ids.npy", ids)
    print(f"seq {seq.shape}  masked share {seq[:, :, 2].mean():.3f}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
