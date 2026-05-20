"""Information-flow causality between players (v3 Part E.5)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


def granger_causality(x: np.ndarray, y: np.ndarray, *, lag: int = 2) -> dict[str, float]:
    """Test whether x helps predict y beyond y's own lags."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Y, Xr, Xu = _lag_design(x, y, lag)
    if len(Y) <= 2 * lag + 1:
        return {"f_stat": 0.0, "p_value": 1.0}
    br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, Y, rcond=None)[0]
    rss_r = float(np.sum((Y - Xr @ br) ** 2))
    rss_u = float(np.sum((Y - Xu @ bu) ** 2))
    df1 = lag
    df2 = max(len(Y) - Xu.shape[1], 1)
    f_stat = ((rss_r - rss_u) / df1) / max(rss_u / df2, 1e-12)
    p = float(1.0 - f_dist.cdf(max(f_stat, 0.0), df1, df2))
    return {"f_stat": float(max(f_stat, 0.0)), "p_value": p}


def transfer_entropy_discrete(x: np.ndarray, y: np.ndarray, *, lag: int = 1, n_bins: int = 4) -> float:
    """Discrete transfer entropy T_{x->y} with histogram bins."""
    xq = np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    yq = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1]))
    total = 0.0
    n = len(yq)
    counts = {}
    for t in range(lag, n):
        key = (yq[t], yq[t - lag], xq[t - lag])
        counts[key] = counts.get(key, 0) + 1
    for (yt, yp, xp), c in counts.items():
        p_xyz = c / max(n - lag, 1)
        p_y_p_x = sum(v for (a, b, d), v in counts.items() if b == yp and d == xp) / max(n - lag, 1)
        p_y_yp = sum(v for (a, b, d), v in counts.items() if a == yt and b == yp) / max(n - lag, 1)
        p_yp = sum(v for (a, b, d), v in counts.items() if b == yp) / max(n - lag, 1)
        total += p_xyz * np.log((p_xyz * p_yp + 1e-12) / (p_y_p_x * p_y_yp + 1e-12))
    return float(max(total, 0.0))


def influence_network(series: pd.DataFrame, *, lag: int = 2, alpha: float = 0.05) -> nx.DiGraph:
    """Build directed player influence network from value time-series."""
    g = nx.DiGraph()
    for src in series.columns:
        g.add_node(src)
    for src in series.columns:
        for dst in series.columns:
            if src == dst:
                continue
            stat = granger_causality(series[src].to_numpy(), series[dst].to_numpy(), lag=lag)
            if stat["p_value"] <= alpha:
                g.add_edge(src, dst, weight=stat["f_stat"], p_value=stat["p_value"])
    return g


def _lag_design(x: np.ndarray, y: np.ndarray, lag: int):
    rows_y, rows_r, rows_u = [], [], []
    for t in range(lag, len(y)):
        y_lags = [y[t - k] for k in range(1, lag + 1)]
        x_lags = [x[t - k] for k in range(1, lag + 1)]
        rows_y.append(y[t])
        rows_r.append([1.0] + y_lags)
        rows_u.append([1.0] + y_lags + x_lags)
    return np.asarray(rows_y), np.asarray(rows_r), np.asarray(rows_u)
