"""Centrality measures on pass networks (Part 1.1).

Implements all four required measures:

    1. Degree    C_D(i) = (d_in + d_out) / (2(n-1))
    2. Betweenness  C_B via Brandes (delegated to networkx, O(VE))
    3. Closeness  on reciprocal-weight distances
    4. PageRank  stationary distribution of the Markov matrix P

Betweenness and closeness use ``1/W`` as edge *distance* (a frequent pass =
a short hop). PageRank is computed directly from P as the dominant left
eigenvector, with uniform teleportation for dangling rows.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from fas.graph.pass_network import PassNetwork


def degree_centrality(net: PassNetwork) -> dict[int, float]:
    """C_D(i) = (d_in(i) + d_out(i)) / (2(n-1)), counting distinct neighbours."""
    n = net.n
    if n <= 1:
        return {p: 0.0 for p in net.players}
    A = (net.W > 0).astype(float)
    d_out = A.sum(axis=1)
    d_in = A.sum(axis=0)
    cd = (d_in + d_out) / (2 * (n - 1))
    return {p: float(cd[k]) for k, p in enumerate(net.players)}


def betweenness_centrality(net: PassNetwork) -> dict[int, float]:
    """Brandes betweenness on reciprocal-weight distances (networkx)."""
    g = _distance_graph(net)
    if g.number_of_nodes() == 0:
        return {}
    return nx.betweenness_centrality(g, weight="distance", normalized=True)


def closeness_centrality(net: PassNetwork) -> dict[int, float]:
    """C_C(i) = (n-1) / sum_j d(i, j) on the reciprocal-weight graph.

    networkx ``closeness_centrality`` uses incoming distance by default; we
    pass the distance graph so shorter (more frequent) connections raise the
    score. Unreachable nodes contribute via the Wasserman-Faust correction.
    """
    g = _distance_graph(net)
    if g.number_of_nodes() == 0:
        return {}
    return nx.closeness_centrality(g, distance="distance", wf_improved=True)


def pagerank(net: PassNetwork, alpha: float = 0.85, tol: float = 1e-12) -> dict[int, float]:
    """Stationary distribution pi = pi P with damping (Part 1.1 measure 4).

    Solved by power iteration on the Google matrix
    ``alpha * P + (1-alpha)/n * 1 1^T``; dangling rows (no outgoing passes)
    are redistributed uniformly. Returns ``pi(i)`` = steady-state ball-touch
    probability for player ``i``.
    """
    n = net.n
    if n == 0:
        return {}
    if n == 1:
        return {net.players[0]: 1.0}

    P = net.markov_matrix()
    dangling = (P.sum(axis=1) == 0)
    P = P.copy()
    P[dangling] = 1.0 / n  # uniform redistribution from dead-ends

    pi = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)
    for _ in range(1000):
        new = alpha * (pi @ P) + (1 - alpha) * teleport
        if np.abs(new - pi).sum() < tol:
            pi = new
            break
        pi = new
    pi = pi / pi.sum()
    return {p: float(pi[k]) for k, p in enumerate(net.players)}


def centrality_table(net: PassNetwork) -> pd.DataFrame:
    """All four centralities + clustering as a per-player DataFrame."""
    g = _distance_graph(net)
    clustering = (
        nx.clustering(g.to_undirected(), weight=None) if g.number_of_nodes() else {}
    )
    deg = degree_centrality(net)
    bet = betweenness_centrality(net)
    clo = closeness_centrality(net)
    pr = pagerank(net)
    rows = [
        {
            "player_id": p,
            "degree": deg.get(p, 0.0),
            "betweenness": bet.get(p, 0.0),
            "closeness": clo.get(p, 0.0),
            "pagerank": pr.get(p, 0.0),
            "clustering": clustering.get(p, 0.0),
        }
        for p in net.players
    ]
    return pd.DataFrame(rows).set_index("player_id")


def _distance_graph(net: PassNetwork) -> nx.DiGraph:
    """Copy of the pass graph with edge attribute ``distance = 1 / weight``."""
    g = nx.DiGraph()
    g.add_nodes_from(net.players)
    for u, v, data in net.graph.edges(data=True):
        w = data.get("weight", 1.0)
        g.add_edge(u, v, weight=w, distance=1.0 / w if w > 0 else np.inf)
    return g
