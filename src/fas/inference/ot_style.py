"""Optimal transport for style distances (v3 Part E.3)."""

from __future__ import annotations

import numpy as np


def sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    *,
    epsilon: float = 0.05,
    n_iter: int = 300,
) -> float:
    """Entropic OT/Sinkhorn distance between two discrete measures."""
    a = _prob(a)
    b = _prob(b)
    C = np.asarray(cost, dtype=float)
    K = np.exp(-C / max(epsilon, 1e-12))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / np.maximum(K @ v, 1e-12)
        v = b / np.maximum(K.T @ u, 1e-12)
    P = (u[:, None] * K) * v[None, :]
    return float(np.sum(P * C))


def sinkhorn_barycenter(
    measures: list[np.ndarray],
    cost: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    epsilon: float = 0.05,
    n_iter: int = 100,
) -> np.ndarray:
    """Approximate Wasserstein barycenter of style measures."""
    if not measures:
        return np.array([])
    A = np.vstack([_prob(m) for m in measures])
    weights = np.ones(len(A)) / len(A) if weights is None else _prob(weights)
    C = np.asarray(cost, dtype=float)
    K = np.exp(-C / max(epsilon, 1e-12))
    q = _prob(np.average(A, axis=0, weights=weights))
    for _ in range(n_iter):
        transports = []
        for a in A:
            u = np.ones_like(a)
            v = np.ones_like(q)
            for _ in range(30):
                u = a / np.maximum(K @ v, 1e-12)
                v = q / np.maximum(K.T @ u, 1e-12)
            transports.append(v * (K.T @ u))
        q = _prob(np.exp(np.sum(weights[:, None] * np.log(np.maximum(transports, 1e-12)), axis=0)))
    return q


def grid_cost(n: int) -> np.ndarray:
    """Simple absolute-distance ground cost for ordered style bins."""
    x = np.arange(n)
    return np.abs(x[:, None] - x[None, :]).astype(float)


def _prob(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return x / max(x.sum(), 1e-12)
