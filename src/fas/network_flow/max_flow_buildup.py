"""Pitch-zone flow model and max-flow build-up analysis (Part 2.1-2.2).

The pitch is partitioned into 18 zones (6 vertical bands x 3 horizontal
thirds). A directed zone graph carries ball progressions between zones with
capacity = number of successful progressive actions. Min-cost max-flow from
defensive-third zones to attacking-third zones gives the team's *build-up
potency*; the flow decomposition reveals the productive corridors.

We use ``networkx.max_flow_min_cost`` (successive shortest paths) on integer
capacities, with edge cost = -round(scaled mean xT gain) so min-cost equals
max-reward, exactly as specified.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from fas.data.schema import PITCH_LENGTH, PITCH_WIDTH

N_BANDS = 6      # vertical bands along x
N_ROWS = 3       # horizontal thirds along y
N_ZONES = N_BANDS * N_ROWS  # = 18


def zone_of(x: float, y: float) -> int:
    """Map a pitch coordinate to a zone index in ``[0, 18)``.

    Zones are numbered ``band * N_ROWS + row``; band increases up the pitch
    (toward the attack), so zones 0-2 are deepest and 15-17 are most advanced.
    """
    band = min(int(x / PITCH_LENGTH * N_BANDS), N_BANDS - 1)
    row = min(int(y / PITCH_WIDTH * N_ROWS), N_ROWS - 1)
    return band * N_ROWS + row


def defensive_zones() -> list[int]:
    """Zones in the defensive third (first two bands)."""
    return [b * N_ROWS + r for b in (0, 1) for r in range(N_ROWS)]


def attacking_zones() -> list[int]:
    """Zones in the attacking third (last two bands)."""
    return [b * N_ROWS + r for b in (4, 5) for r in range(N_ROWS)]


@dataclass(slots=True)
class ZoneGraph:
    """Directed zone graph with capacity and value (mean xT gain) per edge."""

    capacity: np.ndarray   # (18, 18) progressive-action counts
    value: np.ndarray      # (18, 18) mean xT gain per action

    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for i in range(N_ZONES):
            for j in range(N_ZONES):
                if self.capacity[i, j] > 0:
                    g.add_edge(i, j, capacity=int(self.capacity[i, j]),
                               value=float(self.value[i, j]))
        return g


def build_zone_graph(
    actions: pd.DataFrame,
    team_id: int,
    *,
    xt_model=None,
    min_count: int = 5,
) -> ZoneGraph:
    """Build the zone graph for one team (Part 2.1).

    An edge ``i -> j`` exists when >= ``min_count`` successful progressive
    actions moved the ball from zone i to zone j. Edge value is the mean
    xT gain across those actions (zero if no xT model is supplied).
    """
    df = actions[
        (actions["team_id"] == team_id)
        & actions["action_type"].isin(["pass", "carry"])
        & actions["outcome"].astype(bool)
    ].dropna(subset=["x_end", "y_end"]).copy()

    df["z0"] = df.apply(lambda r: zone_of(r["x_start"], r["y_start"]), axis=1)
    df["z1"] = df.apply(lambda r: zone_of(r["x_end"], r["y_end"]), axis=1)
    df = df[df["z0"] != df["z1"]]

    if xt_model is not None:
        df["gain"] = df.apply(
            lambda r: xt_model.value(r["x_end"], r["y_end"])
            - xt_model.value(r["x_start"], r["y_start"]),
            axis=1,
        )
    else:
        df["gain"] = 0.0

    cap = np.zeros((N_ZONES, N_ZONES))
    val = np.zeros((N_ZONES, N_ZONES))
    grp = df.groupby(["z0", "z1"])
    for (i, j), sub in grp:
        if len(sub) >= min_count:
            cap[i, j] = len(sub)
            val[i, j] = sub["gain"].mean()
    return ZoneGraph(capacity=cap, value=val)


@dataclass(slots=True)
class BuildupResult:
    flow_value: int             # total units of flow pushed D-third -> A-third
    total_value: float          # accumulated xT reward along the optimal flow
    corridors: dict[tuple[int, int], int]  # edge -> flow used


def buildup_potency(zg: ZoneGraph, *, value_scale: int = 1000) -> BuildupResult:
    """Min-cost max-flow from defensive to attacking zones (Part 2.2).

    A super-source feeds all defensive-third zones and a super-sink drains all
    attacking-third zones (large capacity). Edge cost = ``-round(value*scale)``
    so the min-cost flow maximizes accumulated xT reward (SSP algorithm).
    """
    g = zg.graph()
    src, snk = "__SRC__", "__SNK__"
    big = int(zg.capacity.sum() + 1)
    for z in defensive_zones():
        g.add_edge(src, z, capacity=big, weight=0)
    for z in attacking_zones():
        g.add_edge(z, snk, capacity=big, weight=0)
    for u, v, data in list(g.edges(data=True)):
        if "value" in data:
            data["weight"] = -int(round(data["value"] * value_scale))
        else:
            data.setdefault("weight", 0)

    if src not in g or snk not in g:
        return BuildupResult(0, 0.0, {})

    flow = nx.max_flow_min_cost(g, src, snk, capacity="capacity", weight="weight")
    flow_value = sum(flow[src].values())
    corridors: dict[tuple[int, int], int] = {}
    total_value = 0.0
    for u, dests in flow.items():
        for v, f in dests.items():
            if f > 0 and isinstance(u, int) and isinstance(v, int):
                corridors[(u, v)] = f
                total_value += f * zg.value[u, v]
    return BuildupResult(int(flow_value), float(total_value), corridors)
