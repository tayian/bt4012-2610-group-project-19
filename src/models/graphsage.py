# M8sage: GraphSAGE on the client graph (data/processed/knn_edges.parquet), out-of-fold.
#
# Two SAGEConv layers (64 hidden) and a linear head, trained full-graph with weighted BCE
# (pos_weight = negatives / positives). Node features are the behaviour columns of features_v2
# (consumption, billing, trajectory, validity, digits), standardised as in src/features/graph.py.
# The label-based graph columns and the peer residuals are left out, so labels only enter
# through the loss. For fold k the loss uses training clients of the other folds (10% of them
# held out for early stopping on PR-AUC, as in src/models/dl.py); fold-k clients are scored.

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv

from src.data.split import N_FOLDS
from src.features.graph import FEATURES as GRAPH_FEATURES
from src.features.graph import embedding
from src.features.peer import COLUMNS as PEER_FEATURES
from src.models.common import load_train, save_run
from src.models.dl import SEED, make_loss, set_seed

HIDDEN = 64
DROPOUT = 0.3
LR = 0.01
MAX_EPOCHS = 200
PATIENCE = 20


class GraphSAGE(nn.Module):
    def __init__(self, d_in, hidden=HIDDEN):
        super().__init__()
        self.conv1 = SAGEConv(d_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), DROPOUT, self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), DROPOUT, self.training)
        return self.head(h).squeeze(-1)


def node_columns(columns):
    skip = {"client_id", "client_catg", "region", *GRAPH_FEATURES, *PEER_FEATURES}
    return [c for c in columns if c not in skip]


def edge_index(edges, nodes):
    src, dst = nodes.get_indexer(edges["src"]), nodes.get_indexer(edges["dst"])
    pairs = np.unique(np.concatenate([np.c_[src, dst], np.c_[dst, src]]), axis=0)
    return torch.as_tensor(pairs.T, dtype=torch.long)


def fit_predict(x, ei, train_nodes, y_train, score_nodes, seed=SEED):
    set_seed(seed)
    tr, va = train_test_split(
        np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=seed
    )
    model = GraphSAGE(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    crit = make_loss("bce", y_train[tr])
    target = torch.as_tensor(y_train[tr], dtype=torch.float32)
    fit_nodes = torch.as_tensor(train_nodes[tr])
    best, best_state, bad = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = crit(model(x, ei)[fit_nodes], target)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, ei)
        ap = average_precision_score(y_train[va], logits[train_nodes[va]].numpy())
        if ap > best + 1e-4:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, ei)[score_nodes]).numpy(), best


def main():
    parser = argparse.ArgumentParser(description="M8sage GraphSAGE on the client graph.")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_v2.parquet"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/knn_edges.parquet"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ids, _, y, folds = load_train("v2")
    table = pd.read_parquet(args.features)
    table["client_id"] = table["client_id"].astype(str)
    table = table.set_index("client_id")
    nodes = pd.Index(table.index)
    train = nodes.get_indexer(ids)
    is_train = np.zeros(len(nodes), dtype=bool)
    is_train[train] = True
    columns = node_columns(table.columns)
    x = torch.as_tensor(embedding(table[columns], is_train), dtype=torch.float32)
    edges = pd.read_parquet(args.edges)
    ei = edge_index(edges.astype({"src": str, "dst": str}), nodes)
    print(
        f"GraphSAGE: {len(nodes):,} nodes, {ei.shape[1]:,} directed edges, {len(columns)} features"
    )
    scores = np.empty(len(ids))
    for k in range(N_FOLDS):
        seen = folds != k
        scores[~seen], best = fit_predict(x, ei, train[seen], y[seen], train[~seen])
        print(f"  fold {k}: inner validation PR-AUC {best:.4f}", flush=True)
    save_run(ids, folds, y, scores, "M8sage", "v2", "M8_graphsage_v2")


if __name__ == "__main__":
    main()
