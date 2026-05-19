"""Expected Possession Value U-Net (Part 3.1) — STUB (requires torch).

EPV(s) = P(team scores next | s) - P(opponent scores next | s).

Spatial-grid representation I in R^{C x 80 x 120}:
    C1 attacking-team density (Gaussian KDE)   C2 defending-team density
    C3 ball-position indicator                 C4 velocity vectors (4 ch)
    C5 time remaining + score diff (broadcast constant channels)

Model: 4-block U-Net encoder/decoder with skip connections; output head =
global-average-pool -> FC -> tanh, scaled to [-1, 1].

Training:
    label  EPV(s_t) = gamma^{T-t} * outcome   (gamma=0.99, T = next goal/half)
    loss   sum_t (EPV_theta - EPV_label)^2 + l1||theta||^2 + l2 TV(EPV_theta)
    target ECE <= 0.02, AUC >= 0.72 on held-out competitions.

Implement when torch wheels are available on the target interpreter
(pip install fas[ml]).
"""

from __future__ import annotations


def build_state_tensor(*args, **kwargs):
    raise NotImplementedError("EPV U-Net requires torch (pip install fas[ml]).")


class EPVUNet:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPV U-Net requires torch (pip install fas[ml]).")
