# data/

| Folder | In git? | Contents | How to get it |
|---|---|---|---|
| `raw/` | no | `client_train.csv`, `invoice_train.csv` (the two course files) | Copy the provided data files here; `python run.py check-raw` compares them with `checksums.txt` |
| `interim/` | no | `client.parquet`, `invoice.parquet`, `invoice_clean.parquet` | `python run.py interim` |
| `processed/` | no | `features_v0.parquet`, `features_v1.parquet`, `seq.npy`, `seq_client_ids.npy` | `python run.py features`, `python run.py features-v1`, `python run.py seq` |
| `splits/` | yes | `train_ids.csv`, `test_ids.csv`, `folds.csv` | Committed once; never re-split |
| `corpus/` | yes | RAG source documents and specs | Written by hand, except `source_docs/codebook.md` (`python run.py codebook`) |
| `stubs/` | yes | Fake versions of every contract file, for testing code before real files exist | `python run.py stubs` |
| `checksums.txt` | yes | SHA-256, line count and size of each raw file | Written once |

`features_v1.parquet` is the leak-free table every model uses by default; it keeps its name
as more feature groups are added as columns. Features built from the inspector-written columns
(`counter_statue` / `reading_remarque` shares) go in a separate
`features_v1_with_inspection.parquet`, which appears next to it once those features exist; the
leakage comparison scores v1 against it.

The test set is the 20% of `client_train` listed in `splits/test_ids.csv`.
