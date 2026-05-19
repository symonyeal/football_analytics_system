"""Opponent vulnerability mapping via min-cut (Part 1.3).

Given an opponent pass network, the minimum s-t cut (source = goalkeeper,
sink = attacking-third players) identifies the smallest set of passing lanes
whose disruption disconnects ball circulation to the attack. By the max-flow
min-cut theorem we obtain it from a max-flow computation; networkx uses a
preflow-push (push-relabel) implementation, matching the spec's O(V^2 sqrt E).

Output: a ranked pressing assignment of (presser, target, priority) triples,
where priority is the flow carried by the cut edge — the lanes worth pressing.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from fas.graph.pass_network import PassNetwork

# An attacking-third action starts beyond this x on the [0,120] pitch.
ATTACKING_THIRD_X = 80.0


@dataclass(slots=True)
class PressingTarget:
    presser: int          # our pressing player (placeholder = the cut edge tail)
    target: int           # opponent player whose outlet we cut (edge head)
    priority: float       # capacity (pass frequency) on the cut edge


def pressing_assignment(
    net: PassNetwork,
    source: int,
    sinks: list[int],
) -> list[PressingTarget]:
    """Min-cut pressing targets from opponent network ``net``.

    Parameters
    ----------
    source : goalkeeper player_id (build-up origin)
    sinks  : attacking-third player_ids (circulation destinations)

    A super-sink is added with infinite-capacity edges from each sink so a
    single s-t max-flow yields the min cut separating GK from the attack.
    """
    if source not in net.players or not sinks:
        return []
    g = nx.DiGraph()
    for u, v, data in net.graph.edges(data=True):
        g.add_edge(u, v, capacity=float(data.get("weight", 1.0)))
    super_sink = "__SINK__"
    big = float(net.W.sum() + 1)
    for s in sinks:
        if s in net.players:
            g.add_edge(s, super_sink, capacity=big)
    if source not in g or super_sink not in g:
        return []

    cut_value, partition = nx.minimum_cut(
        g, source, super_sink, flow_func=nx.algorithms.flow.preflow_push
    )
    reachable, non_reachable = partition
    targets: list[PressingTarget] = []
    for u in reachable:
        for v in g.successors(u):
            if v in non_reachable and v != super_sink:
                cap = g[u][v]["capacity"]
                targets.append(PressingTarget(int(u), int(v), float(cap)))
    targets.sort(key=lambda t: t.priority, reverse=True)
    return targets


def attacking_third_players(net: PassNetwork, actions, team_id: int) -> list[int]:
    """Helper: players whose mean start-x lies in the attacking third."""
    df = actions[actions["team_id"] == team_id]
    mean_x = df.groupby("player_id")["x_start"].mean()
    return [int(p) for p in net.players if mean_x.get(p, 0.0) >= ATTACKING_THIRD_X]


def goalkeeper_node(net: PassNetwork, actions, team_id: int) -> int | None:
    """Heuristic GK = player with the smallest mean start-x (deepest)."""
    df = actions[actions["team_id"] == team_id]
    mean_x = df.groupby("player_id")["x_start"].mean()
    candidates = [(mean_x.get(p, np.inf), p) for p in net.players]
    if not candidates:
        return None
    return int(min(candidates)[1])
