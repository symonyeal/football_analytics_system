"""Team style on an information manifold (v3 Part C.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from fas.entities import TeamSeason
from fas.network_flow.max_flow_buildup import zone_of


def team_style_distribution(
    actions: pd.DataFrame,
    *,
    team_id: int | None = None,
    action_col: str = "action_type",
    zone_count: int = 18,
) -> pd.Series:
    """Represent a team-match as q(action_type, zone)."""
    df = actions.copy()
    if team_id is not None:
        df = df[df["team_id"] == team_id]
    if df.empty:
        return pd.Series(dtype=float, name="style")
    df["zone"] = df.apply(lambda r: zone_of(r["x_start"], r["y_start"]), axis=1)
    counts = df.groupby([action_col, "zone"]).size().astype(float)
    action_types = sorted(actions[action_col].astype(str).unique())
    index = pd.MultiIndex.from_product([action_types, range(zone_count)], names=[action_col, "zone"])
    q = counts.reindex(index).fillna(0.0)
    q = q / max(q.sum(), 1e-12)
    q.name = "style"
    return q


def fisher_rao_distance(p: pd.Series | np.ndarray, q: pd.Series | np.ndarray) -> float:
    """Fisher-Rao geodesic distance on the probability simplex."""
    p_arr, q_arr = _align_prob(p, q)
    inner = float(np.sum(np.sqrt(p_arr * q_arr)))
    return float(2.0 * np.arccos(np.clip(inner, 0.0, 1.0)))


def frechet_mean(distributions: list[pd.Series]) -> pd.Series:
    """Approximate Fisher-Rao Frechet mean via the square-root embedding."""
    if not distributions:
        return pd.Series(dtype=float)
    index = distributions[0].index
    X = np.vstack([d.reindex(index).fillna(0.0).to_numpy(dtype=float) for d in distributions])
    S = np.sqrt(np.maximum(X, 0.0)).mean(axis=0)
    p = S**2
    p = p / max(p.sum(), 1e-12)
    return pd.Series(p, index=index, name="frechet_mean")


def cluster_styles(distributions: list[pd.Series], *, n_clusters: int = 4, random_state: int = 0) -> np.ndarray:
    """Cluster team styles in the Fisher-Rao square-root embedding."""
    if not distributions:
        return np.array([], dtype=int)
    index = distributions[0].index
    X = np.vstack([
        np.sqrt(d.reindex(index).fillna(0.0).to_numpy(dtype=float))
        for d in distributions
    ])
    n_clusters = max(1, min(n_clusters, len(distributions)))
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X)


def enrich(team: TeamSeason, distribution: pd.Series) -> TeamSeason:
    """Attach square-root manifold coordinates to a team entity."""
    coord = np.sqrt(distribution.to_numpy(dtype=float))
    perf = dict(team.performance)
    perf["style_manifold"] = {
        "dimension": int(coord.size),
        "math": "information geometry / Fisher-Rao metric",
    }
    return team.with_updates(style_manifold_coord=coord, performance=perf)


def _align_prob(p: pd.Series | np.ndarray, q: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(p, pd.Series) and isinstance(q, pd.Series):
        idx = p.index.union(q.index)
        p_arr = p.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        q_arr = q.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    else:
        p_arr = np.asarray(p, dtype=float)
        q_arr = np.asarray(q, dtype=float)
    p_arr = np.maximum(p_arr, 0.0)
    q_arr = np.maximum(q_arr, 0.0)
    p_arr = p_arr / max(p_arr.sum(), 1e-12)
    q_arr = q_arr / max(q_arr.sum(), 1e-12)
    return p_arr, q_arr
