"""Pass-network construction and structural metrics (Part 1.1, 1.2).

A pass network is a weighted directed graph ``G = (V, E, W)`` where ``W[i,j]``
counts completed passes from player ``i`` to player ``j``. We expose both a
``networkx.DiGraph`` (for centrality algorithms) and the raw weight matrix
``W`` plus the row-stochastic Markov matrix ``P`` used by PageRank/entropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(slots=True)
class PassNetwork:
    """Container for a single team's pass network over a time window.

    Attributes
    ----------
    players : ordered list of player_ids → matrix index
    W : (n, n) completed-pass count matrix, W[i, j] = i → j
    graph : networkx.DiGraph with edge attribute ``weight`` = W[i, j]
    """

    players: list[int]
    W: np.ndarray
    graph: nx.DiGraph

    @property
    def n(self) -> int:
        return len(self.players)

    def markov_matrix(self) -> np.ndarray:
        """Row-stochastic transition matrix P[i,j] = W[i,j] / sum_k W[i,k].

        Rows with no outgoing passes (dangling nodes) are left as zeros; the
        PageRank routine handles them with uniform teleportation.
        """
        row = self.W.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            P = np.where(row > 0, self.W / row, 0.0)
        return P


def build_pass_network(
    actions: pd.DataFrame,
    team_id: int,
    *,
    period: int | None = None,
    t_lo_ms: int | None = None,
    t_hi_ms: int | None = None,
) -> PassNetwork:
    """Build a :class:`PassNetwork` for one team within an optional time window.

    A pass edge ``i -> j`` requires a *completed* pass action by ``i`` whose
    immediate next event in the same team/possession is an action by ``j``
    (the receiver). Because the canonical schema does not store an explicit
    ``recipient_id``, we infer the receiver as the next on-ball actor of the
    same team within the window — the standard event-stream heuristic.
    """
    df = actions[(actions["team_id"] == team_id)].copy()
    if period is not None:
        df = df[df["period"] == period]
    if t_lo_ms is not None:
        df = df[df["timestamp_ms"] >= t_lo_ms]
    if t_hi_ms is not None:
        df = df[df["timestamp_ms"] < t_hi_ms]
    df = df.sort_values(["period", "timestamp_ms"]).reset_index(drop=True)

    if df.empty:
        return PassNetwork([], np.zeros((0, 0)), nx.DiGraph())

    # Receiver = next same-team on-ball actor after a completed pass.
    passer = df["player_id"].to_numpy()
    is_completed_pass = (
        (df["action_type"] == "pass") & df["outcome"].astype(bool)
    ).to_numpy()
    receiver = np.roll(passer, -1)
    receiver[-1] = -1  # last action has no successor

    edges: dict[tuple[int, int], int] = {}
    for i in range(len(df)):
        if is_completed_pass[i] and receiver[i] != -1 and receiver[i] != passer[i]:
            key = (int(passer[i]), int(receiver[i]))
            edges[key] = edges.get(key, 0) + 1

    players = sorted(set(passer.tolist()) | {r for r in receiver.tolist() if r != -1})
    if not players:
        return PassNetwork([], np.zeros((0, 0)), nx.DiGraph())
    idx = {p: k for k, p in enumerate(players)}
    n = len(players)
    W = np.zeros((n, n))
    g = nx.DiGraph()
    g.add_nodes_from(players)
    for (u, v), w in edges.items():
        W[idx[u], idx[v]] = w
        g.add_edge(u, v, weight=float(w))
    return PassNetwork(players, W, g)


def network_entropy(net: PassNetwork) -> float:
    """Shannon entropy of the normalized pass distribution (Part 1.1).

        H(G) = - sum_{i,j} P[i,j] log P[i,j]

    where P is the *global* normalization W / sum(W). High entropy ==
    distributed, unpredictable passing; low == centralized.
    """
    total = net.W.sum()
    if total <= 0:
        return 0.0
    P = net.W / total
    nz = P[P > 0]
    return float(-(nz * np.log(nz)).sum())


def phase_snapshots(
    actions: pd.DataFrame,
    team_id: int,
    *,
    n_phases: int = 6,
    minutes_per_phase: int = 15,
) -> list[PassNetwork]:
    """Sequence of pass networks over K phases (Part 1.2).

    Default: K=6 phases of 15 minutes each across a 90-minute match.
    """
    snaps: list[PassNetwork] = []
    for k in range(n_phases):
        lo = k * minutes_per_phase * 60_000
        hi = (k + 1) * minutes_per_phase * 60_000
        snaps.append(build_pass_network(actions, team_id, t_lo_ms=lo, t_hi_ms=hi))
    return snaps


def network_velocity(prev: PassNetwork, curr: PassNetwork) -> float:
    """Normalized Frobenius-norm change between consecutive snapshots (Part 1.2).

        dG = ||A_k - A_{k-1}||_F / ||A_{k-1}||_F

    Adjacency matrices are aligned on the union of players so the difference
    is well-defined even when the on-pitch set changes.
    """
    players = sorted(set(prev.players) | set(curr.players))
    A_prev = _aligned(prev, players)
    A_curr = _aligned(curr, players)
    denom = np.linalg.norm(A_prev, "fro")
    if denom == 0:
        return float("inf") if np.linalg.norm(A_curr, "fro") > 0 else 0.0
    return float(np.linalg.norm(A_curr - A_prev, "fro") / denom)


def _aligned(net: PassNetwork, players: list[int]) -> np.ndarray:
    idx = {p: k for k, p in enumerate(net.players)}
    n = len(players)
    A = np.zeros((n, n))
    for a, pa in enumerate(players):
        for b, pb in enumerate(players):
            if pa in idx and pb in idx:
                A[a, b] = net.W[idx[pa], idx[pb]]
    return A
