"""Graph Attention Network for collective valuation (Part 1.4).

A GAT over possession sequences predicts ``Delta EPV`` per action; the mean
received attention weight per player, aggregated over a season, is their
*graph contribution score* (a feature in the PVS pipeline, Part 4.1).

This module is a STUB: it requires ``torch`` + ``torch-geometric`` (extras:
``gnn``), whose wheels may lag on Python 3.14. The class/loss/attention design
is documented inline so it can be implemented when those wheels are available.
The graph-contribution feature can be sourced from centralities until then.
"""

from __future__ import annotations

_REQUIRES = "torch, torch-geometric (pip install fas[gnn])"


class TacticalGAT:
    """Two-layer GAT (planned).

    Architecture (Part 1.4):
        conv1 = GATConv(in=d, out=64, heads=4)
        conv2 = GATConv(64*4, out=1, heads=1)
        node features: coords, velocity, role encoding, in-possession flag
        edge features:  Euclidean distance, angular separation, pass freq
        attention:  alpha_ij = softmax_j(LeakyReLU(a^T[Wh_i || Wh_j || e_ij]))
        message:    h_i' = sigma(sum_j alpha_ij W h_j)
        target:     Delta EPV;   loss = MSE + lambda||theta||^2
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"TacticalGAT requires {_REQUIRES}. See module docstring for the "
            "full specification; use graph centralities as the interim "
            "collective-valuation feature."
        )
