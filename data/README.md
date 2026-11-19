# data/

| Folder | In git? | Contents | How to get it |
|---|---|---|---|
| `raw/` | gzipped only | `client_train.csv.gz`, `invoice_train.csv.gz` (the two course files) | Committed; `python run.py check-raw` unpacks them to `.csv` and compares them with `checksums.txt` |
| `interim/` | no | `client.parquet`, `invoice.parquet`, `invoice_clean.parquet` | `python run.py interim` |
| `processed/` | no | `features_v0.parquet`, `features_v1[_with_inspection].parquet`, `features_v2[_with_inspection].parquet`, `meter_tariff.parquet`, `features_v3.parquet`, `features_v4.parquet`, `peer_residual.parquet`, `graph_features.parquet`, `knn_edges.parquet`, `seq.npy`, `seq_client_ids.npy` | `python run.py features`, `features-v1`, `features-v2`, `features-v3`, `seq` |
| `splits/` | yes | `train_ids.csv`, `test_ids.csv`, `folds.csv` | Committed once; never re-split |
| `corpus/` | yes | RAG source documents, specs, case cards and chunks | Written by hand, except `source_docs/codebook.md` (`python run.py codebook`) and `case_cards.jsonl` / `chunks.jsonl` (`python run.py corpus`) |
| `checksums.txt` | yes | SHA-256, line count and size of each raw file | Written once |

`features_v1.parquet` (F1 to F7) is the leak-free base table, and `features_v2.parquet` adds the
peer-residual and graph features. `features_v4.parquet` is the final model's table: v1 plus the
F12 meter, tariff and recency features (`meter_tariff.parquet`) that a meter replacement after
an inspection cannot move. `features_v3.parquet` keeps all of F12 and is used only to measure
that leakage (docs/decisions.md, section 8). Features built from the
inspector-written columns (`counter_statue` / `reading_remarque` shares) are only in the
`_with_inspection` versions, which the leakage comparison scores against. Every column is
described in [docs/feature_list.md](../docs/feature_list.md).

The test set is the 20% of `client_train` listed in `splits/test_ids.csv`.
