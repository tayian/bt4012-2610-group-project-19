# Small hand-made graphs for the peer-residual, graph and v2 feature steps: labels must stay
# out-of-fold and the placeholder meter numbers must not create edges.

import unittest

import numpy as np
import pandas as pd

from src.features import combine, graph, peer
from src.models import label_spreading


class GraphTests(unittest.TestCase):
    def test_meter_edges_skip_placeholders_and_big_groups(self):
        meters = pd.DataFrame(
            {
                "client_id": ["a", "b", "c", "d"] + [f"x{i}" for i in range(12)],
                "counter_number": ["123", "123", "0000", "0000"] + ["777"] * 12,
                "counter_type": "ELEC",
            }
        )
        node = pd.Series(range(len(meters)), index=meters["client_id"])
        edges = graph.meter_edges(meters, node)
        self.assertEqual(sorted(map(tuple, edges[["src", "dst"]].to_numpy())), [(0, 1), (1, 0)])

    def test_visible_labels_hide_own_fold_and_test(self):
        target = np.array([1, 0, 1, 0, 1.0])
        train_nodes = np.array([0, 1, 2, 3])  # node 4 is a test client
        folds = np.array([0, 1, 2, 3])
        labels = graph.visible_labels(5, train_nodes, folds, target)
        self.assertTrue(np.isnan(labels[0][0]))  # fold 0 cannot see node 0
        self.assertEqual(labels[0][2], 1)
        self.assertTrue(np.isnan(labels[-1][4]))  # nobody sees the test client
        self.assertEqual(np.isnan(labels[-1][:4]).sum(), 0)  # test clients see all train labels

    def test_neighbour_rate_uses_visible_labels_only(self):
        edges = pd.DataFrame({"src": [0, 0, 0, 1], "dst": [1, 2, 3, 0]})
        y = np.array([np.nan, 1, 0, np.nan])
        rate = graph.neighbour_rate(edges, y, 4)
        self.assertEqual(rate[0], 0.5)  # neighbour 3's label is hidden
        self.assertTrue(np.isnan(rate[1]))  # its only neighbour is hidden

    def test_community_rate_shrinks_small_communities(self):
        community = np.array([0, 0, 1])
        y = np.array([1, 1, np.nan])
        rate = graph.community_rate(community, y, prior=0.1)
        self.assertAlmostEqual(rate[0], (2 + graph.PRIOR_WEIGHT * 0.1) / (2 + graph.PRIOR_WEIGHT))
        self.assertAlmostEqual(rate[2], 0.1)  # no visible label: the overall rate


class PeerTests(unittest.TestCase):
    def test_fold_groups_fit_on_other_folds_non_fraud_only(self):
        folds = pd.DataFrame({"client_id": list("abcdef"), "fold": [0, 0, 1, 2, 3, 4]})
        labels = pd.DataFrame({"client_id": list("abcdefg"), "target": [0, 1, 0, 1, 0, 0, 0]})
        groups = peer.fold_groups(folds, labels, ["g"])
        score, fit = groups[0]
        self.assertEqual(score, {"a", "b"})
        self.assertEqual(fit, {"c", "e", "f"})  # d is fraud, a and b are the scored fold
        self.assertEqual(groups[-1], ({"g"}, {"a", "c", "e", "f"}))


class SpreadingTests(unittest.TestCase):
    def test_labels_spread_to_the_closer_side(self):
        # Chain 0 - 1 - 2 - 3 - 4 with node 0 fraud and node 4 not.
        s = label_spreading.normalised_adjacency(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]), 5)
        scores = label_spreading.spread(s, np.array([0, 4]), np.array([1.0, 0.0]))
        self.assertGreater(scores[1], scores[3])
        self.assertAlmostEqual(scores[2], 0.5)


class CombineTests(unittest.TestCase):
    def test_combine_drops_and_merges(self):
        v1 = pd.DataFrame(
            {"client_id": ["a", "b"], "district": [1, 2], "x": [1.0, 2.0], "y": [3, 4]}
        )
        peer_t = pd.DataFrame({"client_id": ["b", "a"], **{c: [0.5, 0.1] for c in peer.COLUMNS}})
        graph_t = pd.DataFrame({"client_id": ["a", "b"], **{c: [1, 2] for c in graph.FEATURES}})
        out = combine.combine(v1, peer_t, graph_t, drop=["y"])
        self.assertEqual(list(out.columns), ["client_id", "x", *peer.COLUMNS, *graph.FEATURES])
        self.assertEqual(out.loc[out.client_id == "a", "peer_resid_mean"].item(), 0.1)


if __name__ == "__main__":
    unittest.main()
