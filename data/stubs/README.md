# Fake STEG fixtures (stubs)

Everything in this folder is fake except the 1,000 client IDs, which are taken from
data/raw/client_train.csv. Labels and raw invoices are random. The interim files,
features_v0, features_v1 and features_v1_with_inspection are built from those fake raw rows by the real load, clean
and feature code, so they have the real schemas. seq.npy is random with the agreed shape
(clients x 120 months x 3 channels). features_v2 and features_v2_with_inspection have the
real columns, with random peer-residual and graph values. Splits, scores, SHAP
values, corpus chunks and briefs are random values with the real column names. Use them
only for interface tests, never as results. The split here is synthetic; the official split
is in data/splits/.

Regenerate with `python run.py stubs` (`python3 -m src.data.make_stubs`). It reads only the client
IDs of data/raw/client_train.csv. manifest.json lists every file.
