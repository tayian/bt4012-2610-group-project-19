# Data loading and cleaning rules

How `python run.py interim` turns the two raw course files into `data/interim/`. Counts for every rule are in `results/tables/data_quality.csv`.

## Loading (`src/data/load.py`)

- Both CSVs are read with explicit column types and saved as parquet (`client.parquet`, `invoice.parquet`).
- Every text field is stripped of spaces once, here, and `counter_type` is upper-cased.
- The raw client column `disrict` (a typo in the source data) is renamed `district` and kept as text.
- Every invoice must belong to a known client, and client IDs must be unique.

## Cleaning (`src/data/clean.py`)

- **Duplicates:** only exact duplicate rows are dropped (the training data has none). Impossible readings are kept and flagged, because they are often the fraud signal itself.
- **Codes:** `counter_statue` is standardised to `0`–`5` and `reading_remarque` to `6`–`9`; anything else becomes `other`.
- **Consumption:** `cons` is the sum of the four consumption levels. `months_valid` is `months_number`, set to missing when it is ≤ 0 or > 120, and `cons_pm = cons / months_valid`. Every feature step uses `months_valid`, so this rule lives only here.
- **Flags:**
  - `flag_bad_months_number`: `months_number` ≤ 0 or > 12.
  - `flag_index_backwards`: `new_index` < `old_index`.
  - `flag_index_rollover`: a backwards reading that is an exact meter wrap past 99999, so not suspicious (2,252 of the 2,264 backwards readings).
  - `flag_recon_mismatch` and `recon_gap`: index difference × `counter_coefficient` ≠ `cons`, after allowing for rollovers.
  - `flag_partial_year`: the invoice falls in the incomplete final year.
- **Final year:** 2019 is incomplete (September to December run at 26–40% of the 2018 median monthly volume). Its rows are kept but flagged, so year-on-year features should exclude it.

## Code values

`results/tables/code_values.csv` lists every raw value of `counter_statue`, `reading_remarque`, `tarif_type` and `counter_code` with its count, most frequent first (ties by value). It feeds the RAG code book (`python run.py codebook`).
