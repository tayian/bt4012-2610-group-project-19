# Clean invoices, flag impossible readings and write the data quality table.
#
# Nothing suspicious is deleted: impossible readings are often the fraud signal, so they are
# kept and flagged. Only exact duplicate rows are dropped.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import LEVELS

STATUE_STATES = ["0", "1", "2", "3", "4", "5"]
REMARQUE_STATES = ["6", "7", "8", "9"]
MAX_MONTHS_VALID = 12  # plan rule F6: months_number <= 0 or > 12 is invalid
MAX_MONTHS_PLAUSIBLE = 120  # beyond 10 years the period is meaningless: months_valid is NaN
PARTIAL_MONTH_RATIO = 0.5  # a month below half the previous year's median volume is incomplete
CODE_FIELDS = ["counter_statue", "reading_remarque", "tarif_type", "counter_code"]
COLUMNS = [
    "client_id",
    "invoice_date",
    "tarif_type",
    "counter_number",
    "counter_statue",
    "counter_code",
    "reading_remarque",
    "counter_coefficient",
    "old_index",
    "new_index",
    "months_number",
    "counter_type",
    "cons",
    "cons_pm",
    "months_valid",
    "idx_diff",
    "recon_gap",
    "flag_index_backwards",
    "flag_index_rollover",
    "flag_recon_mismatch",
    "flag_bad_months_number",
    "flag_partial_year",
    *LEVELS,
]


def standardise_codes(text, states):
    # Map codes outside the known states to 'other'.
    return text.where(text.isin(states), "other").astype("string")


def final_year_status(dates, ratio=PARTIAL_MONTH_RATIO):
    # Return (final_year, is_complete, worst_month, worst_ratio).
    #
    # Each month of the final year is compared with the median monthly invoice count of the
    # year before it.
    years = dates.dt.year
    final = int(years.max())
    baseline = dates[years == final - 1].dt.month.value_counts().median()
    counts = dates[years == final].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    ratios = counts / baseline
    worst = int(ratios.idxmin())
    return final, bool(ratios.min() >= ratio), worst, float(ratios.min())


def clean_invoices(df, report=None):
    # Return the cleaned invoice table; append data quality rows to `report` if given.
    # Expects the output of src/data/load.py, whose text fields are already normalised.
    n_raw = len(df)
    rows = [] if report is None else report

    def log(issue, count, action):
        rows.append(
            {
                "issue": issue,
                "count": int(count),
                "pct_of_rows": round(100 * count / n_raw, 4),
                "action": action,
            }
        )

    inv = df.drop_duplicates().reset_index(drop=True)
    log("exact duplicate invoice rows", n_raw - len(inv), "dropped")
    key = ["client_id", "invoice_date", "counter_number", "counter_type"]
    log(
        "rows sharing client, date, counter and type (not exact duplicates)",
        inv.duplicated(key).sum(),
        "kept: different tariff lines or index readings",
    )

    for field, states in (("counter_statue", STATUE_STATES), ("reading_remarque", REMARQUE_STATES)):
        raw = inv[field]
        inv[field] = standardise_codes(raw, states)
        other = inv[field].eq("other")
        log(
            f"{field} outside {{{', '.join(states)}}} ({', '.join(sorted(raw[other].unique()))})",
            other.sum(),
            "mapped to 'other'",
        )

    inv["cons"] = inv[LEVELS].sum(axis=1)
    months = inv["months_number"]
    inv["flag_bad_months_number"] = months.le(0) | months.gt(MAX_MONTHS_VALID)
    log(
        f"months_number <= 0 or > {MAX_MONTHS_VALID}",
        inv["flag_bad_months_number"].sum(),
        "kept, flag_bad_months_number",
    )
    # The one months rule for every later step: features use months_valid, never months_number.
    implausible = months.le(0) | months.gt(MAX_MONTHS_PLAUSIBLE)
    inv["months_valid"] = months.where(~implausible).astype("float64")
    inv["cons_pm"] = (inv["cons"] / inv["months_valid"]).astype("float64")
    log(
        f"months_number <= 0 or > {MAX_MONTHS_PLAUSIBLE} (impossible period)",
        implausible.sum(),
        "kept, cons_pm set to NaN",
    )
    log("zero total consumption", inv["cons"].eq(0).sum(), "kept (a bypassed meter reads zero)")

    # A 5-digit meter passing 99999 restarts at 0; that is not a backwards reading.
    inv["idx_diff"] = inv["new_index"] - inv["old_index"]
    inv["flag_index_backwards"] = inv["idx_diff"].lt(0)
    # Number of digits of old_index (0 counts as one digit).
    digits = np.searchsorted(10 ** np.arange(1, 19), inv["old_index"].abs(), side="right") + 1
    wrapped = inv["idx_diff"] + np.power(10, np.minimum(digits, 18))
    inv["flag_index_rollover"] = inv["flag_index_backwards"] & (
        wrapped * inv["counter_coefficient"]
    ).eq(inv["cons"])
    log("new_index < old_index", inv["flag_index_backwards"].sum(), "kept, flag_index_backwards")
    log(
        "  of which an exact meter rollover",
        inv["flag_index_rollover"].sum(),
        "kept, flag_index_rollover; reconciled with the wrapped difference",
    )

    effective = inv["idx_diff"].where(~inv["flag_index_rollover"], wrapped)
    inv["recon_gap"] = (effective * inv["counter_coefficient"] - inv["cons"]).abs()
    inv["flag_recon_mismatch"] = inv["recon_gap"].gt(0)
    log(
        "index difference x counter_coefficient != total consumption",
        inv["flag_recon_mismatch"].sum(),
        "kept, flag_recon_mismatch and recon_gap",
    )
    log(
        "counter_coefficient == 0",
        inv["counter_coefficient"].eq(0).sum(),
        "kept (always a reconciliation mismatch)",
    )
    log("counter_number == '0' (placeholder)", inv["counter_number"].eq("0").sum(), "kept")

    final, complete, month, ratio = final_year_status(inv["invoice_date"])
    inv["flag_partial_year"] = inv["invoice_date"].dt.year.eq(final) & (not complete)
    log(
        f"invoices in final year {final} "
        f"({'complete' if complete else f'incomplete: month {month} has {ratio:.0%} of {final - 1} median volume'})",
        inv["flag_partial_year"].sum() if not complete else 0,
        "kept, flag_partial_year (exclude from year-on-year features)" if not complete else "none",
    )
    return inv[COLUMNS]


def code_values(inv_raw):
    # Every distinct raw value and its count for the code fields (used by the RAG code book),
    # most frequent first; ties are ordered by value so the file is the same on every run.
    frames = []
    for field in CODE_FIELDS:
        counts = inv_raw[field].value_counts(dropna=False)
        frame = pd.DataFrame(
            {"field": field, "value": counts.index.astype("string"), "count": counts.to_numpy()}
        )
        frames.append(frame.sort_values(["count", "value"], ascending=[False, True]))
    out = pd.concat(frames, ignore_index=True)
    out["pct_of_rows"] = (100 * out["count"] / len(inv_raw)).round(4)
    return out


def main():
    parser = argparse.ArgumentParser(description="Clean invoices and write the data quality table.")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    try:
        raw = pd.read_parquet(args.interim / "invoice.parquet")
        clients = pd.read_parquet(args.interim / "client.parquet")
        report = []
        clean = clean_invoices(raw, report)
        created = clean["client_id"].map(clients.set_index("client_id")["creation_date"])
        lag = (created - clean["invoice_date"]).dt.days
        n_early = int(lag.gt(0).sum())
        report.append(
            {
                "issue": "invoice dated before client creation_date",
                "count": n_early,
                "pct_of_rows": round(100 * n_early / len(raw), 4),
                "action": f"kept (max lag {int(lag.max())} days: likely data-entry lag)",
            }
        )
        args.tables.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(args.interim / "invoice_clean.parquet", index=False)
        pd.DataFrame(report).to_csv(args.tables / "data_quality.csv", index=False)
        code_values(raw).to_csv(args.tables / "code_values.csv", index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"clean: {exc}\n")
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Saved {len(clean):,} clean invoices to {args.interim / 'invoice_clean.parquet'}")


if __name__ == "__main__":
    main()
