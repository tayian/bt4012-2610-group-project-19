# M8 label spreading on the client graph (data/processed/knn_edges.parquet), out-of-fold.
#
# The same update as scikit-learn's LabelSpreading (alpha = 0.2, symmetric normalised graph,
# at most 30 iterations), run on our saved graph instead of one rebuilt from raw features, so
# every graph model uses the same edges. For fold k only the labels of training clients in the
# other folds are given; the fold-k clients' scores are their spread fraud share.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.split import N_FOLDS
from src.models.common import load_train, save_run

ALPHA = 0.2
MAX_ITER = 30
TOL = 1e-3


def normalised_adjacency(src, dst, n):
    # Undirected, unweighted graph; S = D^-1/2 W D^-1/2.
    w = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    w = ((w + w.T) > 0).astype(np.float64)
    d = np.asarray(w.sum(axis=1)).ravel()
    inv = np.where(d > 0, 1 / np.sqrt(d), 0.0)
    return sp.diags(inv) @ w @ sp.diags(inv)


def spread(s, labelled, y, alpha=ALPHA):
    # F <- alpha * S F + (1 - alpha) * Y, with Y one-hot on the labelled nodes.
    n = s.shape[0]
    base = np.zeros((n, 2))
    base[labelled, y.astype(int)] = 1.0
    f = base.copy()
    for _ in range(MAX_ITER):
        new = alpha * (s @ f) + (1 - alpha) * base
        done = np.abs(new - f).sum() < TOL
        f = new
        if done:
            break
    mass = f.sum(axis=1)
    prior = y.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0, f[:, 1] / mass, prior)


def main():
    parser = argparse.ArgumentParser(description="M8 label spreading on the client graph.")
    parser.add_argument("--edges", type=Path, default=Path("data/processed/knn_edges.parquet"))
    args = parser.parse_args()
    ids, _, y, folds = load_train("v1")
    edges = pd.read_parquet(args.edges)
    nodes = pd.Index(pd.unique(np.concatenate([ids, edges["src"], edges["dst"]])))
    s = normalised_adjacency(
        nodes.get_indexer(edges["src"]), nodes.get_indexer(edges["dst"]), len(nodes)
    )
    train = nodes.get_indexer(ids)
    scores = np.empty(len(ids))
    for k in range(N_FOLDS):
        seen = folds != k
        spread_scores = spread(s, train[seen], y[seen])
        scores[~seen] = spread_scores[train[~seen]]
    save_run(ids, folds, y, scores, "M8", "v1", "M8_label_spreading_v1")


if __name__ == "__main__":
    main()
