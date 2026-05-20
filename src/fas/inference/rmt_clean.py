"""Random matrix theory covariance cleaning (v3 Part E.2)."""

from __future__ import annotations

import numpy as np


def marchenko_pastur_bounds(n_samples: int, n_features: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Return Marchenko-Pastur noise eigenvalue bounds."""
    q = n_features / max(n_samples, 1)
    root = np.sqrt(q)
    return float(sigma2 * (1.0 - root) ** 2), float(sigma2 * (1.0 + root) ** 2)


def clean_covariance(X: np.ndarray, *, assume_centered: bool = False) -> np.ndarray:
    """Spectrally clean a sample covariance matrix using MP filtering."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2 and X.shape[0] != X.shape[1]:
        Xc = X if assume_centered else X - X.mean(axis=0, keepdims=True)
        cov = np.cov(Xc, rowvar=False)
        n_samples, n_features = X.shape
    else:
        cov = X
        n_features = cov.shape[0]
        n_samples = max(n_features * 2, n_features + 1)
    vals, vecs = np.linalg.eigh(cov)
    sigma2 = float(np.median(vals))
    _, lam_plus = marchenko_pastur_bounds(n_samples, n_features, sigma2=sigma2)
    signal = vals > lam_plus
    if signal.any():
        noise_mean = float(vals[~signal].mean()) if (~signal).any() else sigma2
        cleaned_vals = np.where(signal, vals, noise_mean)
    else:
        cleaned_vals = np.full_like(vals, vals.mean())
    cleaned = (vecs * cleaned_vals) @ vecs.T
    return (cleaned + cleaned.T) / 2.0
