# Read the raw STEG CSVs, strip text fields once and save them as parquet in data/interim.
# The raw client column `disrict` (a typo in the source) becomes `district`.
# Run: python3 -m src.data.load

import argparse
from pathlib import Path

import pandas as pd

from src.data.ids import check_ids

LEVELS = [f"consommation_level_{i}" for i in range(1, 5)]
INVOICE_DTYPES = {
    "client_id": "string",
    "counter_number": "string",
    "counter_statue": "string",
    "reading_remarque": "string",
    "counter_type": "string",
    "tarif_type": "int64",
    "counter_code": "int64",
    "counter_coefficient": "int64",
    "old_index": "int64",
    "new_index": "int64",
    "months_number": "int64",
    **{level: "int64" for level in LEVELS},
}
CLIENT_DTYPES = {
    "client_id": "string",
    "disrict": "string",
    "client_catg": "string",
    "region": "string",
}


def parse_dates(values, dayfirst_formats=("%d/%m/%Y", "%d/%m/%y")):
    # Parse ISO dates first, then day/month dates (some exports use short dates such as 24/3/14).
    text = values.astype("string").str.strip()
    out = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    bad = out.isna() & text.notna()
    for fmt in dayfirst_formats:
        if not bad.any():
            break
        out[bad] = pd.to_datetime(text[bad], format=fmt, errors="coerce").to_numpy()
        bad = out.isna() & text.notna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} unparseable dates, e.g. {text[bad].iloc[0]!r}")
    return out


def normalise_text(frame):
    # Strip every string column and upper-case counter_type, once per distinct value.
    for column in frame.select_dtypes("string"):
        codes, uniques = pd.factorize(frame[column])
        text = pd.Series(uniques, dtype="string").str.strip()
        if column == "counter_type":
            text = text.str.upper()
        frame[column] = text.array.take(codes, allow_fill=True)
    return frame


def read_raw(path, dtypes):
    # Read only the expected columns, so a missing column fails here with a clear message.
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    missing = sorted(set(dtypes) - set(header))
    if missing:
        raise ValueError(f"{path}: missing raw columns {missing}")
    return pd.read_csv(path, dtype=dtypes, encoding="utf-8-sig")


def load_clients(path):
    clients = normalise_text(read_raw(path, CLIENT_DTYPES)).rename(columns={"disrict": "district"})
    clients["client_id"] = check_ids(clients["client_id"], "clients", unique=True)
    clients["creation_date"] = parse_dates(clients["creation_date"])
    clients["target"] = pd.to_numeric(clients["target"], errors="raise").astype("int8")
    return clients


def load_invoices(path):
    invoices = normalise_text(read_raw(path, INVOICE_DTYPES))
    invoices["client_id"] = check_ids(invoices["client_id"], "invoices")
    invoices["invoice_date"] = parse_dates(invoices["invoice_date"])
    return invoices


def main():
    parser = argparse.ArgumentParser(description="Convert the raw STEG CSVs to parquet.")
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--invoices", type=Path, default=Path("data/raw/invoice_train.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()
    try:
        clients = load_clients(args.clients)
        invoices = load_invoices(args.invoices)
        unknown = ~invoices["client_id"].isin(clients["client_id"])
        if unknown.any():
            raise ValueError(f"{int(unknown.sum())} invoices belong to unknown clients.")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        clients.to_parquet(args.out_dir / "client.parquet", index=False)
        invoices.to_parquet(args.out_dir / "invoice.parquet", index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"load: {exc}\n")
    print(f"Saved {len(clients):,} clients and {len(invoices):,} invoices to {args.out_dir}")


if __name__ == "__main__":
    main()
