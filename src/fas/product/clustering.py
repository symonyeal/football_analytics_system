"""Pass clustering for spatial style discovery.

Passes are clustered on their start/end coordinates plus length and angle.
We prefer density-based clustering (DBSCAN) when there is enough data, which
lets us report *noise* (unclustered passes) honestly; we fall back to KMeans
on sparse samples. Cluster labels are assigned from the geometry only — after
inspecting the centroid — never invented up front.

Mirroring is supported: with ``mirror=True`` passes are folded onto the lower
half of the pitch (``y -> min(y, 80 - y)``) so left/right equivalents cluster
together for symmetry analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from fas.data.schema import PITCH_WIDTH


@dataclass(slots=True)
class PassClusters:
    """Clustering output: per-pass labels and per-cluster summaries."""

    labels: np.ndarray          # cluster id per pass, -1 = noise
    summary: pd.DataFrame        # one row per cluster
    method: str
    n_noise: int


def _channel(y: float) -> str:
    if y < PITCH_WIDTH / 3:
        return "left"
    if y > 2 * PITCH_WIDTH / 3:
        return "right"
    return "central"


def _label_cluster(x0: float, y0: float, x1: float, y1: float) -> str:
    """Human-readable label from a cluster centroid's geometry."""
    dx, dy = x1 - x0, y1 - y0
    start_ch, end_ch = _channel(y0), _channel(y1)
    third = "deep" if x0 < 40 else ("middle" if x0 < 80 else "final-third")

    if x1 > 102 and abs(dy) < 12:
        return "through ball in behind"
    if x0 > 90 and x1 < x0 and abs(dy) > 8:
        return "final-third cutback"
    if start_ch != end_ch and abs(dy) > 20:
        return f"{start_ch} half-space switch"
    if dx > 18 and abs(dy) < 12:
        side = start_ch if start_ch != "central" else "central"
        return f"{side} wide progression" if side != "central" else "central progression"
    if abs(dx) < 10 and abs(dy) < 10:
        return f"{third} {start_ch} short link"
    if dx < -8:
        return f"{third} recycle back"
    return f"{third} {start_ch} build-up"


def cluster_passes(
    actions: pd.DataFrame,
    *,
    team_id: int | None = None,
    mirror: bool = False,
    eps: float = 0.55,
    min_samples: int = 12,
    max_kmeans: int = 8,
) -> PassClusters:
    """Cluster completed passes by ``(x_start, y_start, x_end, y_end, len, angle)``."""
    df = actions[(actions["action_type"] == "pass") & actions["outcome"].astype(bool)]
    df = df.dropna(subset=["x_end", "y_end"])
    if team_id is not None:
        df = df[df["team_id"] == team_id]
    if len(df) < min_samples:
        return PassClusters(np.array([], dtype=int), _empty_summary(), "insufficient-data", 0)

    x0 = df["x_start"].to_numpy(float)
    y0 = df["y_start"].to_numpy(float)
    x1 = df["x_end"].to_numpy(float)
    y1 = df["y_end"].to_numpy(float)
    if mirror:
        y0 = np.minimum(y0, PITCH_WIDTH - y0)
        y1 = np.minimum(y1, PITCH_WIDTH - y1)

    length = np.hypot(x1 - x0, y1 - y0)
    angle = np.arctan2(y1 - y0, x1 - x0)
    feats = np.column_stack([x0, y0, x1, y1, length, np.cos(angle), np.sin(angle)])
    feats = StandardScaler().fit_transform(feats)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)
    n_clusters = len(set(labels) - {-1})
    method = "DBSCAN (density)"
    if n_clusters < 3:
        # Sparse / diffuse data: fall back to KMeans so the atlas still works.
        k = int(min(max_kmeans, max(3, len(df) // 60)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feats)
        method = f"KMeans (k={k}, fallback)"

    rows = []
    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        cx0, cy0, cx1, cy1 = x0[mask].mean(), y0[mask].mean(), x1[mask].mean(), y1[mask].mean()
        rows.append({
            "cluster_id": int(cid),
            "label": _label_cluster(cx0, cy0, cx1, cy1),
            "size": int(mask.sum()),
            "x_start": float(cx0), "y_start": float(cy0),
            "x_end": float(cx1), "y_end": float(cy1),
            "mean_length": float(length[mask].mean()),
            "mean_angle_deg": float(np.degrees(np.arctan2(
                (y1[mask] - y0[mask]).mean(), (x1[mask] - x0[mask]).mean()))),
        })
    summary = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return PassClusters(labels, summary, method, int((labels == -1).sum()))


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "cluster_id", "label", "size", "x_start", "y_start",
        "x_end", "y_end", "mean_length", "mean_angle_deg",
    ])
