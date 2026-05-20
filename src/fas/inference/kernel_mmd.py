"""Kernel embeddings and two-sample tests (v3 Part E.4)."""

from __future__ import annotations

import numpy as np


def rbf_kernel(X: np.ndarray, Y: np.ndarray | None = None, *, gamma: float | None = None) -> np.ndarray:
    """RBF kernel matrix."""
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    if gamma is None:
        Z = np.vstack([X, Y])
        d = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)
    dxy = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * dxy)


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, *, gamma: float | None = None) -> float:
    """Unbiased squared Maximum Mean Discrepancy."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Kxx = rbf_kernel(X, gamma=gamma)
    Kyy = rbf_kernel(Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    n, m = len(X), len(Y)
    a = (Kxx.sum() - np.trace(Kxx)) / max(n * (n - 1), 1)
    b = (Kyy.sum() - np.trace(Kyy)) / max(m * (m - 1), 1)
    c = Kxy.mean()
    return float(a + b - 2.0 * c)


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_perm: int = 200,
    gamma: float | None = None,
    random_state: int = 0,
) -> dict[str, float]:
    """Permutation test for style/player distribution shift."""
    rng = np.random.default_rng(random_state)
    observed = mmd2_unbiased(X, Y, gamma=gamma)
    Z = np.vstack([X, Y])
    n = len(X)
    null = np.zeros(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(len(Z))
        null[b] = mmd2_unbiased(Z[perm[:n]], Z[perm[n:]], gamma=gamma)
    p = (1.0 + np.sum(null >= observed)) / (n_perm + 1.0)
    return {"mmd2": float(observed), "p_value": float(p)}


def kernel_ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    gamma: float | None = None,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Nonparametric kernel ridge prediction."""
    K = rbf_kernel(X_train, gamma=gamma)
    alpha = np.linalg.solve(K + ridge * np.eye(len(K)), np.asarray(y_train, dtype=float))
    return rbf_kernel(X_test, X_train, gamma=gamma) @ alpha
