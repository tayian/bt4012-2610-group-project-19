# Builds data/corpus/source_docs/codebook.md: one entry per code value with facts computed from
# the data and clearly labelled assumed meanings (STEG never published code definitions).
# Fraud rates use training clients only. Run with `python run.py codebook`.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import CODE_FIELDS
from src.data.load import LEVELS

JUNK = {
    "counter_statue": ["46", "A", "618", "769", "269375", "420"],
    "reading_remarque": ["203", "413", "207", "5"],
    "counter_code": ["0", "1"],
}

SHIFT = (
    "The same 34 invoice rows (5 clients; 33 ELEC rows and 1 GAZ row) carry an odd `counter_statue` (46, 618, 769, 269375 or 420), "
    "an odd `reading_remarque` (203, 207, 413 or 5), and `counter_code` 0 or 1. In these rows `reading_remarque` holds a number that looks like a real `counter_code`, "
    "`counter_coefficient` holds what looks like a remark (6, 8 or 9) and the index and consumption fields look moved along as well. That suggests each row was shifted by one column during data entry or export. "
)
JUNK_NOTE = {
    "counter_statue": SHIFT
    + "The value `A` is different: 13 rows of a single client that are not shifted. `src/data/clean.py` maps all of these statue values to `other`. They are kept here so that every raw value has an entry.",
    "reading_remarque": SHIFT
    + "`src/data/clean.py` maps these values to `other`. They are kept here so that every raw value has an entry.",
    "counter_code": SHIFT
    + "`counter_code` 0 (33 rows) and 1 (1 row) occur only in these shifted rows, so they are not real meter codes. They are kept here so that every raw value has an entry.",
}

# (assumed meaning or None, relevance, extra observed note or None)
T = {
    ("counter_statue", "0"): (
        "Normal active service.",
        "Baseline state carried by almost every invoice; on its own it says nothing about fraud.",
        None,
    ),
    ("counter_statue", "1"): (
        "Disconnected / cut-off service.",
        "If the assumed meaning holds, consumption recorded under this status could point to self-reconnection ([TYP-05]). Because most status-1 rows still show consumption, treat it as a weak signal and confirm the account's real service-order history before acting.",
        "Most of these rows (about 72%) still record nonzero consumption, and the code does not appear after 2013, which sits uneasily with a plain 'service is off' reading.",
    ),
    ("counter_statue", "2"): (
        "Meter replaced / exchanged.",
        "A meter exchange can explain index discontinuities; check for one before reading an index jump as a rollback ([TYP-03]). Too few training clients to say anything about fraud rates.",
        None,
    ),
    ("counter_statue", "3"): (
        "Meter under repair / calibration.",
        "Possibly related to defective or jammed meters ([TYP-06]). Too few training clients to interpret the fraud rate.",
        None,
    ),
    ("counter_statue", "4"): (
        "Meter blocked / mechanical failure.",
        "Possibly related to jammed or defective meters ([TYP-06]); the training fraud rate is close to the base rate, so it is not a strong signal by itself.",
        None,
    ),
    ("counter_statue", "5"): (
        "Suspected anomaly or meter defect flagged by the reader.",
        "Among the common status codes this one shows the largest gap between its training fraud rate and the base rate. That is an association in historical data, not proof of tampering; use it as a reason to look closer, together with consumption evidence.",
        None,
    ),
    ("reading_remarque", "6"): (
        "Normal reading taken by the meter reader.",
        "Most frequent remark; its fraud rate is close to the base rate. Not informative on its own.",
        None,
    ),
    ("reading_remarque", "7"): (
        "Reading declared by the customer.",
        "Rare and not seen after 2006; too little data to use as a signal.",
        None,
    ),
    ("reading_remarque", "8"): (
        "Estimated reading (no physical read).",
        "If the assumed meaning holds, long runs of this remark could hide true consumption ([TYP-08]). The training fraud rate is somewhat above the base rate.",
        None,
    ),
    ("reading_remarque", "9"): (
        "Meter inaccessible / not read.",
        "Do not treat this code as evidence that the customer obstructed the meter: its frequency makes the assumed meaning doubtful. Check the reader's own logs before relying on it ([TYP-04], [TYP-08]).",
        "It covers about a third of all invoice rows, a very high share for a 'meter could not be read' meaning, so the assumed meaning may be wrong.",
    ),
    ("tarif_type", "11"): (
        "Standard (residential) low-voltage electricity tariff.",
        "Main electricity tariff; its fraud rate is close to the base rate. Useful as the reference group when comparing consumption with peers ([TYP-07]).",
        None,
    ),
    ("tarif_type", "12"): (
        "Residential dual-rate (day/night) electricity tariff.",
        "Small group with a training fraud rate above the base rate; interpret with care given its size.",
        None,
    ),
    ("tarif_type", "14"): (
        "Agricultural (pumping) electricity tariff.",
        "Small group; a high share of zero-consumption rows may reflect seasonal use rather than fraud.",
        None,
    ),
    ("tarif_type", "15"): (
        "Commercial / professional low-voltage electricity tariff.",
        "Relevant to tariff-misclassification checks ([TYP-07]) only if the assumed meaning holds. The high share of zero-consumption rows needs explaining before it is read as a fraud signal.",
        None,
    ),
    ("tarif_type", "29"): (
        None,
        "An electricity tariff with no known meaning; no specific investigation use.",
        "An earlier draft of this code book called it a domestic natural-gas tariff; the data contradict that, since every row is ELEC.",
    ),
    ("tarif_type", "40"): (
        "The standard gas tariff (most likely residential gas, since nearly all clients are category 11).",
        "Being on this tariff simply means the client has a gas meter. Use it to route gas-specific checks; it is not evidence of a professional or industrial activity.",
        "An earlier draft called it 'small industrial power'; the data contradict that. Every row is GAZ, and it accounts for about 98.7% of all gas invoice rows.",
    ),
    ("tarif_type", "45"): (
        "A commercial or professional gas tariff.",
        "Small gas group with a training fraud rate well above the base rate (based on a few hundred clients); interpret with care.",
        None,
    ),
}
GENERIC_REL = "No known meaning; no specific investigation use beyond noting the value in the client snapshot."
CATG = {
    "11": (
        "Residential (domestic) customer.",
        "Reference category for peer comparison; fraud rate close to the base rate.",
    ),
    "12": (
        "Commercial / professional customer.",
        "Training fraud rate is below the base rate. A residential-looking consumption profile in this category is not by itself suspicious.",
    ),
    "51": (
        None,
        "Training fraud rate is about three times the base rate (over a thousand training clients), a strong association in this data. Its meaning is unknown, so a brief may report the rate but should not describe the category (for example as 'industrial').",
    ),
}


def text_codes(values):
    # Categorical of the values as text (tarif_type and counter_code are integers); each
    # distinct value is converted only once. Values are already stripped by src/data/load.py.
    codes, uniques = pd.factorize(values)
    return pd.Categorical.from_codes(codes, pd.Index(uniques).astype("string"))


def code_stats(interim, splits, tables):
    # Row counts, ELEC/GAZ split, zero-consumption share and training fraud rate per code value.
    inv = pd.read_parquet(
        Path(interim) / "invoice.parquet",
        columns=["client_id", "invoice_date", "counter_type", *CODE_FIELDS, *LEVELS],
    )
    clients = pd.read_parquet(
        Path(interim) / "client.parquet", columns=["client_id", "client_catg", "target"]
    )
    train = set(pd.read_csv(Path(splits) / "train_ids.csv")["client_id"])
    code_values = pd.read_csv(Path(tables) / "code_values.csv", dtype=str)
    target = clients[clients.client_id.isin(train)].set_index("client_id")["target"]
    if inv["client_id"].isna().any():
        raise ValueError("invoices with a missing client_id cannot be grouped")
    client, ids = pd.factorize(inv["client_id"])
    kind = inv["counter_type"]
    per_row = pd.DataFrame(
        {
            "client": client,
            "elec": kind.eq("ELEC").to_numpy(bool, na_value=False),
            "gaz": kind.eq("GAZ").to_numpy(bool, na_value=False),
            "zero": inv[LEVELS].sum(axis=1).eq(0).to_numpy(),
            "year": inv["invoice_date"].dt.year.to_numpy(),
        }
    )
    # Target of each client code; NaN for clients outside the training split.
    client_target = pd.Series(ids).map(target).to_numpy(np.float64, na_value=np.nan)
    n = len(inv)
    rows = []
    for field in CODE_FIELDS:
        value = text_codes(inv[field])
        stats = per_row.groupby(value, observed=True).agg(
            count=("client", "size"),
            elec=("elec", "sum"),
            gaz=("gaz", "sum"),
            zero=("zero", "mean"),
            first=("year", "min"),
            last=("year", "max"),
        )
        pairs = pd.DataFrame({"value": value, "client": client}).drop_duplicates()
        pairs["target"] = client_target[pairs["client"].to_numpy()]
        trained = (
            pairs.dropna(subset=["target"])
            .groupby("value", observed=False)["target"]
            .agg(["size", "sum", "mean"])
        )
        for v in code_values[code_values.field == field]["value"]:
            r, t = stats.loc[v], trained.loc[v]
            rows.append(
                dict(
                    field=field,
                    value=v,
                    count=int(r["count"]),
                    pct=100 * int(r["count"]) / n,
                    elec=int(r["elec"]),
                    gaz=int(r["gaz"]),
                    zero_share=float(r["zero"]),
                    train_n=int(t["size"]),
                    train_fraud=int(t["sum"]),
                    train_rate=float(t["mean"]),
                    years=f"{int(r['first'])}-{int(r['last'])}",
                )
            )
    catg = clients.client_catg.astype(str)
    row_catg = pd.Series(ids).map(dict(zip(clients.client_id, catg))).to_numpy(object)[client]
    is_train = clients.client_id.isin(train)
    for v in CATG:
        count = int((catg == v).sum())
        t = clients.target[(catg == v) & is_train]
        sub = per_row[row_catg == v]
        rows.append(
            dict(
                field="client_catg",
                value=v,
                count=count,
                pct=100 * count / len(clients),
                elec=int(sub["elec"].sum()),
                gaz=int(sub["gaz"].sum()),
                zero_share=float(sub["zero"].mean()),
                train_n=len(t),
                train_fraud=int(t.sum()),
                train_rate=float(t.mean()),
            )
        )
    return {
        "n_rows": n,
        "train_n": int(len(target)),
        "train_rate": float(target.mean()),
        "n_clients": int(len(clients)),
        "rows": rows,
    }


def render(d):
    # Turn the statistics into the markdown code book.
    N, BASE, TRN = d["n_rows"], d["train_rate"], d["train_n"]
    rows = {(r["field"], r["value"]): r for r in d["rows"]}

    def pct(x, k=2):
        return f"{x:.{k}f}%"

    def share(x):
        return "<0.0001%" if x < 0.0001 else (pct(x, 4) if x < 0.01 else pct(x))

    def nrows(n):
        return f"{n:,} invoice row" + ("" if n == 1 else "s")

    def ctype(r):
        if r["gaz"] == 0:
            return "ELEC"
        if r["elec"] == 0:
            return "GAZ"
        return "both"

    def split(r):
        tot = r["elec"] + r["gaz"]
        return (
            f"{r['elec']:,} ELEC / {r['gaz']:,} GAZ ({pct(100 * r['elec'] / tot, 1)} ELEC)"
            if tot
            else "no rows"
        )

    def fraud(r):
        if r["train_n"] == 0:
            return "No training client carries this value, so there is no fraud rate."
        s = f"Training clients carrying it at least once: {r['train_n']:,}, of which fraud: {r['train_fraud']:,} ({pct(100 * r['train_rate'])}; base rate {pct(100 * BASE)})."
        if r["train_n"] < 30:
            s += " Too few clients to interpret."
        return s

    def chunk(field, v, counter_type, obs, assumed, meaning, rel):
        meta = f"`doc: codebook` | `field: {field}` | `value: {v}` | `counter_type: {counter_type}` | `jurisdiction: TN`"
        if assumed:
            meta += " | `assumed: true`"
            meaning = f"* **Assumed meaning (unverified — STEG has not published code definitions):** {assumed}"
        out = [
            f"### [CODE-{field}-{v}] {field} = {v}",
            f"* **Metadata:** {meta}",
            f"* **Observed in data:** {obs}",
            meaning,
            f"* **Investigation relevance:** {rel}",
        ]
        return "\n".join(out) + "\n"

    def entry(field, v, junk=False):
        r = rows[(field, v)]
        assumed, rel, note = T.get((field, v), (None, GENERIC_REL, None))
        meaning = "* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption."
        if junk:
            assumed = None
            rel = "Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything."
            meaning = (
                "* **Meaning:** Invalid value outside the field's normal set (see the note above)."
            )
            if (field, v) == ("counter_statue", "A"):
                note = "These rows belong to a single client and are not column-shifted (their `counter_code` and remark look normal); the letter has no known meaning."
            else:
                note = "Row appears column-shifted (see the note above), so its consumption and index fields are unreliable."
        obs = (
            f"{nrows(r['count'])} ({share(r['pct'])} of {N:,}); "
            f"{split(r)}; {pct(100 * r['zero_share'], 1)} of these rows have zero total consumption; seen {r['years']}. {fraud(r)}"
        )
        if note:
            obs += f" {note}"
        return chunk(field, v, ctype(r), obs, assumed, meaning, rel)

    def catg_entry(v):
        r = rows[("client_catg", v)]
        assumed, rel = CATG[v]
        obs = (
            f"{r['count']:,} clients ({pct(r['pct'])} of {d['n_clients']:,}); their invoices split {split(r)}; "
            f"{pct(100 * r['zero_share'], 1)} of their invoice rows have zero total consumption. "
            f"{r['train_n']:,} are training clients, of whom {r['train_fraud']:,} are fraud ({pct(100 * r['train_rate'])}; base rate {pct(100 * BASE)})."
        )
        meaning = "* **Meaning:** Unknown. STEG has not published category definitions and the data give no basis for an assumption."
        return chunk("client_catg", v, "both", obs, assumed, meaning, rel)

    values = {f: [r["value"] for r in d["rows"] if r["field"] == f] for f in CODE_FIELDS}
    n_entries = sum(len(v) for v in values.values()) + len(CATG)
    gas_rows = sum(r["gaz"] for r in d["rows"] if r["field"] == "tarif_type")

    L = []
    L.append(f"""# STEG Code Book (`data/corpus/source_docs/codebook.md`)

**Document ID:** `DOC-02-CODEBOOK`  
**Corpus layer:** Categorical code reference  
**Status:** Data-derived facts; code meanings are assumptions  
**Entries:** {n_entries} (one per value)

---

## About this code book

STEG has never published definitions for the codes in this dataset. Everything in an entry's **Observed in data** line is computed from the project data. Everything in an **Assumed meaning** line is a guess made by the project team. The guesses come from field names and common utility practice, are unverified, and are kept only where the data do not contradict them. Entries carrying a guess are tagged `assumed: true` in their metadata. An Inspector Brief that cites a code meaning must say that the meaning is assumed.

There is one entry for every value of `counter_statue`, `reading_remarque`, `tarif_type` and `counter_code` listed in `results/tables/code_values.csv`, and one for each `client_catg` value ({", ".join(CATG)}). Each entry is one retrieval chunk, identified by `[CODE-<field>-<value>]`. BM25 matches these exact IDs and code strings better than dense embeddings do.

## How the facts were computed

* **Rows and share:** raw values from `data/interim/invoice.parquet` ({N:,} invoice rows). Raw values are used because `invoice_clean.parquet` maps unexpected `counter_statue` / `reading_remarque` values to `other`.
* **ELEC/GAZ split:** `counter_type` of the same rows (stripped, upper-cased).
* **Zero-consumption share:** rows where `consommation_level_1..4` sum to 0.
* **Fraud rate:** training clients only (`data/splits/train_ids.csv`, {TRN:,} clients, base fraud rate {pct(100 * BASE)} from `target` in `data/interim/client.parquet`). It is the share of fraud clients among training clients who carry the value on at least one invoice. Clients with longer histories are more likely to carry a rare value at some point, so this is a descriptive association, not a causal or per-invoice rate. Rates based on fewer than 30 clients are flagged.
* For `client_catg` the counts are clients from `client.parquet`, not invoice rows.

## Index

| Field | Values | Notes |
| :--- | :--- | :--- |
| `counter_statue` | {len(values["counter_statue"])} | 0–5 are the normal set; {", ".join(JUNK["counter_statue"])} are invalid ({sum(rows[("counter_statue", v)]["count"] for v in JUNK["counter_statue"])} rows) |
| `reading_remarque` | {len(values["reading_remarque"])} | 6–9 are the normal set; {", ".join(JUNK["reading_remarque"])} are invalid ({sum(rows[("reading_remarque", v)]["count"] for v in JUNK["reading_remarque"])} rows) |
| `tarif_type` | {len(values["tarif_type"])} | each value is used with only one counter type |
| `counter_code` | {len(values["counter_code"])} | each value is used with only one counter type; no meanings assumed; 0 and 1 are invalid (34 rows) |
| `client_catg` | {len(CATG)} | client-level field |

---
""")

    def section(field, title, intro):
        L.append(f"## {title}\n\n{intro}\n")
        junk = JUNK.get(field, [])
        for v in values[field]:
            if v not in junk:
                L.append(entry(field, v))
        if junk:
            L.append(f"## {field}: invalid or rare values\n\n{JUNK_NOTE[field]}\n")
            for v in junk:
                L.append(entry(field, v, junk=True))

    section(
        "counter_statue",
        "counter_statue (meter status)",
        "The field name suggests a meter or contract status. The values 0–5 have assumed meanings.",
    )
    section(
        "reading_remarque",
        "reading_remarque (reading remark)",
        "The field name suggests a remark made by the meter reader. The values 6–9 have assumed meanings.",
    )
    section(
        "tarif_type",
        "tarif_type (tariff)",
        f"Every tariff value is used with only one counter type, so the ELEC/GAZ split is a hard fact even where the meaning is unknown. Gas rows total {gas_rows:,}.",
    )
    section(
        "counter_code",
        "counter_code (meter code)",
        "The field name suggests a meter model or family. Every value is used with only one counter type. No per-value meanings are assumed, so these entries record only what the data show.",
    )
    L.append(
        "## client_catg (client category)\n\nA client-level field from `client.parquet`, not an invoice field.\n"
    )
    for v in CATG:
        L.append(catg_entry(v))
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="Build the RAG code book from the data.")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--out", type=Path, default=Path("data/corpus/source_docs/codebook.md"))
    args = parser.parse_args()
    try:
        text = render(code_stats(args.interim, args.splits, args.tables))
        args.out.write_text(text, encoding="utf-8")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"codebook: {exc}\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
