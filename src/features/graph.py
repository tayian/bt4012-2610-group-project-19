# Graph features: kNN edges on standardised behaviour features plus shared-meter edges, then
# out-of-fold neighbour, community and meter fraud rates (no client sees its own label).
# Run: python3 -m src.features.graph

import argparse
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data.split import N_FOLDS, SEED, read_splits

K = 10
MAX_METER_CLIENTS = 10  # larger groups on one meter number are placeholders, not shared meters
PRIOR_WEIGHT = 10  # community_fraud_rate counts the overall rate as this many extra clients
# F2 to F5 columns of features_v1 (consumption, billing, trajectory, robust deviation).
EMBED_PREFIXES = (
    "cons_pm_",
    "share_level_2_plus",
    "n_invoices",
    "invoices_per_year",
    "months_number_",
    "days_since_last_invoice_",
    "has_missing_months",
    "yoy_drop_max",
    "slope_last3y",
    "sudden_spike_share",
    "zero_",
    "robust_z_max",
    "n_robust_z_gt3",
)
PLACEHOLDER = re.compile(r"^(\d)\1*$")  # one repeated digit: 0, 0000, 1111, ...
FEATURES = [
    "nbr_fraud_rate_k10",
    "degree",
    "community_size",
    "community_fraud_rate",
    "n_meter_neighbours",
    "meter_fraud_rate",
]


def embed_columns(columns):
    return [c for c in columns if c.startswith(EMBED_PREFIXES)]


def embedding(features, train_mask):
    # Signed log1p for heavy tails, standardised on training clients; missing becomes 0.
    x = features.to_numpy(np.float64)
    x = np.sign(x) * np.log1p(np.abs(x))
    mean = np.nanmean(x[train_mask], axis=0)
    std = np.nanstd(x[train_mask], axis=0)
    std[~(std > 0)] = 1.0
    z = (x - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def knn_edges(z, k=K):
    # Directed edges i -> its k nearest other clients, with cosine similarity.
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    dist, idx = nn.fit(z).kneighbors(z)
    rows = np.repeat(np.arange(len(z)), k + 1)
    keep = idx.ravel() != rows
    # Drop self; if duplicates push self out of the list, keep the first k others.
    src, dst, sim = rows[keep], idx.ravel()[keep], 1 - dist.ravel()[keep]
    first = pd.DataFrame({"src": src}).groupby("src").cumcount().to_numpy() < k
    return pd.DataFrame({"src": src[first], "dst": dst[first], "similarity": sim[first]})


def meter_edges(meters, node):
    # Undirected client pairs (saved in both directions) that share a real meter number.
    m = meters[~meters["counter_number"].astype(str).str.match(PLACEHOLDER)]
    m = m.drop_duplicates(["client_id", "counter_number", "counter_type"])
    size = m.groupby(["counter_number", "counter_type"])["client_id"].transform("nunique")
    m = m[(size > 1) & (size <= MAX_METER_CLIENTS)]
    m = m.assign(node=node.reindex(m["client_id"]).to_numpy())
    pairs = m.merge(m, on=["counter_number", "counter_type"])
    pairs = pairs[pairs["node_x"] != pairs["node_y"]]
    out = pd.DataFrame({"src": pairs["node_x"].to_numpy(), "dst": pairs["node_y"].to_numpy()})
    return out.drop_duplicates().assign(similarity=np.nan)


def visible_labels(n, train_nodes, folds, target):
    # labels[g] = node labels visible to group g (folds 0..4, then the test set); NaN = hidden.
    groups = []
    for k in [*range(N_FOLDS), None]:
        y = np.full(n, np.nan)
        seen = train_nodes if k is None else train_nodes[folds != k]
        y[seen] = target[seen]
        groups.append(y)
    return groups


def neighbour_rate(edges, y_visible, n):
    # Mean visible label over each node's out-neighbours; NaN where none is visible.
    y = y_visible[edges["dst"].to_numpy()]
    seen = ~np.isnan(y)
    total = np.bincount(edges["src"].to_numpy()[seen], weights=y[seen], minlength=n)
    count = np.bincount(edges["src"].to_numpy()[seen], minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def community_rate(community, y_visible, prior):
    # Visible fraud rate per community, shrunk towards the prior so small ones are not extreme.
    seen = ~np.isnan(y_visible)
    frauds = pd.Series(y_visible[seen]).groupby(community[seen]).sum()
    counts = pd.Series(1, index=np.flatnonzero(seen)).groupby(community[seen]).sum()
    rate = (frauds + PRIOR_WEIGHT * prior) / (counts + PRIOR_WEIGHT)
    return rate.reindex(community).fillna(prior).to_numpy()


def graph_features(edges, n, node_group, labels, prior):
    # node_group[i] = which visibility group scores node i (fold index, or N_FOLDS for test).
    knn = edges[edges["kind"] == "knn"]
    meter = edges[edges["kind"] == "meter"]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(zip(edges["src"].to_numpy(), edges["dst"].to_numpy()))
    communities = nx.community.louvain_communities(g, seed=SEED)
    community = np.empty(n, dtype=np.int64)
    # Number communities largest first so the IDs are stable for a given graph.
    for cid, members in enumerate(sorted(communities, key=lambda c: (-len(c), min(c)))):
        community[list(members)] = cid
    out = pd.DataFrame(
        {
            "degree": np.array([d for _, d in sorted(g.degree())], dtype=np.int64),
            "community_id": community,
            "community_size": np.bincount(community)[community],
            "n_meter_neighbours": np.bincount(meter["src"].to_numpy(), minlength=n),
        }
    )
    for name in ["nbr_fraud_rate_k10", "community_fraud_rate", "meter_fraud_rate"]:
        out[name] = np.nan
    for group, y in enumerate(labels):
        rows = node_group == group
        out.loc[rows, "nbr_fraud_rate_k10"] = neighbour_rate(knn, y, n)[rows]
        out.loc[rows, "community_fraud_rate"] = community_rate(community, y, prior)[rows]
        out.loc[rows, "meter_fraud_rate"] = neighbour_rate(meter, y, n)[rows]
    return out


def main():
    parser = argparse.ArgumentParser(description="Build the kNN client graph and its features.")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_v1.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    try:
        clients = pd.read_parquet(args.interim / "client.parquet", columns=["client_id", "target"])
        clients["client_id"] = clients["client_id"].astype("string")
        _, _, folds = read_splits(args.split_dir, clients)
        v1 = pd.read_parquet(args.features)
        v1["client_id"] = v1["client_id"].astype("string")
        v1 = v1.set_index("client_id").reindex(clients["client_id"])
        n = len(v1)
        node = pd.Series(np.arange(n), index=v1.index)
        fold_of = folds.set_index("client_id")["fold"].reindex(v1.index)
        train_nodes = np.flatnonzero(fold_of.notna().to_numpy())
        node_group = fold_of.fillna(N_FOLDS).astype(int).to_numpy()
        target = clients.set_index("client_id")["target"].reindex(v1.index).to_numpy(float)

        z = embedding(v1[embed_columns(v1.columns)], fold_of.notna().to_numpy())
        meters = pd.read_parquet(
            args.interim / "invoice_clean.parquet",
            columns=["client_id", "counter_number", "counter_type"],
        )
        meters["client_id"] = meters["client_id"].astype("string")
        edges = pd.concat(
            [
                knn_edges(z).assign(kind="knn"),
                meter_edges(meters, node).assign(kind="meter"),
            ],
            ignore_index=True,
        )
        labels = visible_labels(n, train_nodes, node_group[train_nodes], target)
        features = graph_features(edges, n, node_group, labels, target[train_nodes].mean())
        features.insert(0, "client_id", v1.index.to_numpy())

        ids = v1.index.to_numpy()
        saved_edges = pd.DataFrame(
            {
                "src": ids[edges["src"].to_numpy()],
                "dst": ids[edges["dst"].to_numpy()],
                "kind": edges["kind"].to_numpy(),
                "similarity": edges["similarity"].to_numpy(),
            }
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        saved_edges.to_parquet(args.out_dir / "knn_edges.parquet", index=False)
        features.to_parquet(args.out_dir / "graph_features.parquet", index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"graph: {exc}\n")
    counts = saved_edges["kind"].value_counts()
    print(
        f"{counts.get('knn', 0):,} kNN and {counts.get('meter', 0):,} meter edges -> "
        f"{args.out_dir / 'knn_edges.parquet'}; "
        f"{features['community_id'].nunique():,} communities"
    )


if __name__ == "__main__":
    main()
