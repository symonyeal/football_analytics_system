"""Tests for the graph theory module (Part 1)."""

import numpy as np
import pandas as pd

from fas.data.schema import validate_actions
from fas.graph import (
    build_pass_network,
    centrality_table,
    network_entropy,
    pagerank,
)
from fas.graph.min_cut_pressing import pressing_assignment


def _toy_actions():
    # A cycle of completed passes 1->2->3->1 plus a hub 4.
    rows = []
    seq = [1, 2, 3, 1, 4, 2, 4, 3]
    for k, p in enumerate(seq):
        rows.append({
            "match_id": 1, "period": 1, "timestamp_ms": k * 1000,
            "player_id": p, "team_id": 100, "action_type": "pass",
            "x_start": 50.0, "y_start": 40.0, "x_end": 60.0, "y_end": 40.0,
            "outcome": True,
        })
    return validate_actions(pd.DataFrame(rows))


def test_pass_network_edges():
    net = build_pass_network(_toy_actions(), team_id=100)
    assert net.n == 4
    # at least the 1->2 edge exists
    i, j = net.players.index(1), net.players.index(2)
    assert net.W[i, j] >= 1


def test_pagerank_is_distribution():
    net = build_pass_network(_toy_actions(), team_id=100)
    pr = pagerank(net)
    assert abs(sum(pr.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in pr.values())


def test_centrality_table_columns():
    net = build_pass_network(_toy_actions(), team_id=100)
    tbl = centrality_table(net)
    for col in ["degree", "betweenness", "closeness", "pagerank", "clustering"]:
        assert col in tbl.columns


def test_entropy_nonnegative():
    net = build_pass_network(_toy_actions(), team_id=100)
    assert network_entropy(net) >= 0.0


def test_min_cut_returns_targets():
    net = build_pass_network(_toy_actions(), team_id=100)
    targets = pressing_assignment(net, source=1, sinks=[3])
    # cut may be empty if fully connected, but call must not error and be sorted
    pri = [t.priority for t in targets]
    assert pri == sorted(pri, reverse=True)
