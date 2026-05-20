"""Topological data analysis of team shape (v3 Part E.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


@dataclass(slots=True)
class PersistenceSummary:
    """Persistence diagrams and vector features for a point cloud."""

    h0: np.ndarray
    h1: np.ndarray
    features: dict[str, float]
    method: str = "Vietoris-Rips fallback summary"
    math: str = "persistent homology / algebraic topology"


def persistence_features(points: np.ndarray) -> PersistenceSummary:
    """Compute lightweight H0 and proxy H1 persistence features.

    If ripser/gudhi are installed they can replace this function. The fallback
    is still useful: H0 lifetimes are exactly MST edge lengths of the Rips
    filtration, and the H1 proxy captures excess long edges beyond the tree.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return PersistenceSummary(np.zeros((0, 2)), np.zeros((0, 2)), {"compactness": 0.0, "hole_proxy": 0.0})
    D = squareform(pdist(pts))
    mst = minimum_spanning_tree(D).toarray()
    lifetimes = mst[mst > 0]
    h0 = np.column_stack([np.zeros(len(lifetimes)), np.sort(lifetimes)])
    all_edges = D[np.triu_indices_from(D, k=1)]
    thresh = np.median(lifetimes) if len(lifetimes) else np.median(all_edges)
    cycle_edges = all_edges[all_edges > thresh]
    h1_life = np.maximum(cycle_edges - thresh, 0.0)
    h1 = np.column_stack([np.full(len(h1_life), thresh), thresh + h1_life])
    features = {
        "compactness": float(np.mean(lifetimes)) if len(lifetimes) else 0.0,
        "max_connectivity_lifetime": float(np.max(lifetimes)) if len(lifetimes) else 0.0,
        "hole_proxy": float(np.sum(h1_life)),
    }
    return PersistenceSummary(h0=h0, h1=h1, features=features)


def persistence_image(diagram: np.ndarray, *, n_x: int = 16, n_y: int = 16, sigma: float = 1.0) -> np.ndarray:
    """Vectorize a persistence diagram as a Gaussian persistence image."""
    diag = np.asarray(diagram, dtype=float)
    img = np.zeros((n_x, n_y))
    if diag.size == 0:
        return img
    birth = diag[:, 0]
    pers = np.maximum(diag[:, 1] - diag[:, 0], 0.0)
    xs = np.linspace(birth.min(initial=0.0), birth.max(initial=1.0) + 1e-6, n_x)
    ys = np.linspace(0.0, pers.max(initial=1.0) + 1e-6, n_y)
    for b, p in zip(birth, pers):
        img += p * np.exp(-0.5 * (((xs[:, None] - b) ** 2 + (ys[None, :] - p) ** 2) / sigma**2))
    return img
