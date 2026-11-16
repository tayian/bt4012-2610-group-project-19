# data/

| Folder | In git? | Contents | How to get it |
|---|---|---|---|
| `raw/` | no | `client_train.csv`, `invoice_train.csv` (the two course files) | Copy the provided data files here; `python run.py check-raw` compares them with `checksums.txt` |
| `interim/` | no | `client.parquet`, `invoice.parquet`, `invoice_clean.parquet` | `python run.py interim` |
| `processed/` | no | `features_v0.parquet`, `features_v1[_with_inspection].parquet`, `features_v2[_with_inspection].parquet`, `peer_residual.parquet`, `graph_features.parquet`, `knn_edges.parquet`, `seq.npy`, `seq_client_ids.npy` | `python run.py features`, `features-v1`, `features-v2`, `seq` |
| `splits/` | yes | `train_ids.csv`, `test_ids.csv`, `folds.csv` | Committed once; never re-split |
| `corpus/` | yes | RAG source documents and specs | Written by hand, except `source_docs/codebook.md` (`python run.py codebook`) |
| `stubs/` | yes | Fake versions of every contract file, for testing code before real files exist | `python run.py stubs` |
| `checksums.txt` | yes | SHA-256, line count and size of each raw file | Written once |

`features_v1.parquet` (F1 to F7) is the leak-free table every model uses by default, and
`features_v2.parquet` adds the peer-residual and graph features. Features built from the
inspector-written columns (`counter_statue` / `reading_remarque` shares) are only in the
`_with_inspection` versions, which the leakage comparison scores against. Every column is
described in [docs/feature_list.md](../docs/feature_list.md).

The test set is the 20% of `client_train` listed in `splits/test_ids.csv`.
