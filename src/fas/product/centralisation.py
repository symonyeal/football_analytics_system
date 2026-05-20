"""Centralisation: how concentrated is a team's passing influence?

Formation and control are different things. A team can keep the same shape
while the main hub moves from a midfielder to a forward, or while influence
spreads out. We track a Freeman-style centralisation index on the PageRank
distribution of the pass network, plus the identity of the dominant hub.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.graph import build_pass_network, centrality_table, network_entropy
from fas.graph.pass_network import PassNetwork


@dataclass(slots=True)
class CentralisationResult:
    """Centralisation summary for one team in one window."""

    index: float            # Freeman centralisation in [0, 1]
    hub_player: int | None  # PageRank-dominant player
    hub_share: float        # the hub's PageRank mass
    entropy: float          # Shannon entropy of the pass distribution
    n_players: int
    n_passes: int


def freeman_centralisation(scores: np.ndarray) -> float:
    """Freeman centralisation of a centrality vector that sums to one.

    For a probability-like centrality (PageRank), the maximum of
    ``sum_i (c_max - c_i)`` over distributions summing to one is ``n - 1``
    (all mass on a single node). Dividing by that bound puts the index in
    ``[0, 1]``: 0 = perfectly even, 1 = a single hub touches everything.
    """
    v = np.asarray(scores, dtype=float)
    n = len(v)
    if n <= 1:
        return 0.0
    total = v.sum()
    if total <= 0:
        return 0.0
    v = v / total
    return float((v.max() - v).sum() / (n - 1))


def centralisation_from_network(net: PassNetwork, *, exclude: set[int] | None = None
                                ) -> CentralisationResult:
    """Compute centralisation from an existing :class:`PassNetwork`."""
    exclude = exclude or set()
    if net.n == 0:
        return CentralisationResult(0.0, None, 0.0, 0.0, 0, 0)
    cents = centrality_table(net)
    pr = cents["pagerank"]
    keep = pr.index.difference(list(exclude)) if exclude else pr.index
    pr = pr.loc[keep]
    if pr.empty:
        return CentralisationResult(0.0, None, 0.0, 0, 0)
    idx = freeman_centralisation(pr.to_numpy())
    hub = int(pr.idxmax())
    share = float(pr.max() / pr.sum()) if pr.sum() > 0 else 0.0
    return CentralisationResult(
        index=idx,
        hub_player=hub,
        hub_share=share,
        entropy=network_entropy(net),
        n_players=int(pr.shape[0]),
        n_passes=int(net.W.sum()),
    )


def centralisation_by_phase(
    actions: pd.DataFrame,
    team_id: int,
    *,
    minutes_per_phase: int = 15,
    n_phases: int = 6,
    exclude_keeper_id: int | None = None,
) -> pd.DataFrame:
    """Centralisation per 15-minute phase for one team.

    Returns one row per phase with the index, dominant hub, hub share, entropy,
    and sample size, so hub-identity drift over a match is visible.
    """
    exclude = {exclude_keeper_id} if exclude_keeper_id is not None else set()
    rows = []
    for k in range(n_phases):
        lo = k * minutes_per_phase * 60_000
        hi = (k + 1) * minutes_per_phase * 60_000
        # Pass network within this minute window (period-aware via timestamp).
        net = build_pass_network(actions, team_id, t_lo_ms=lo, t_hi_ms=hi)
        res = centralisation_from_network(net, exclude=exclude)
        rows.append({
            "phase": f"{k * minutes_per_phase}-{(k + 1) * minutes_per_phase}",
            "phase_index": k,
            "minute_start": k * minutes_per_phase,
            "minute_end": (k + 1) * minutes_per_phase,
            "centralisation": res.index,
            "hub_player": res.hub_player,
            "hub_share": res.hub_share,
            "entropy": res.entropy,
            "n_players": res.n_players,
            "n_passes": res.n_passes,
        })
    return pd.DataFrame(rows)
