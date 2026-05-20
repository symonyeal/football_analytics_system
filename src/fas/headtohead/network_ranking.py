"""Network ranking of head-to-head results (v3 Part D.2)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def results_graph(
    matches: pd.DataFrame,
    *,
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
) -> nx.DiGraph:
    """Build directed graph winner -> loser weighted by margin."""
    g = nx.DiGraph()
    for row in matches.itertuples(index=False):
        h = getattr(row, home_col)
        a = getattr(row, away_col)
        hg = getattr(row, home_goals_col)
        ag = getattr(row, away_goals_col)
        g.add_node(h)
        g.add_node(a)
        if hg == ag:
            continue
        winner, loser = (h, a) if hg > ag else (a, h)
        margin = abs(float(hg - ag))
        if g.has_edge(winner, loser):
            g[winner][loser]["weight"] += margin
        else:
            g.add_edge(winner, loser, weight=margin)
    return g


def massey_ratings(matches: pd.DataFrame, *, teams: list[int] | None = None) -> pd.Series:
    """Least-squares Massey ratings."""
    if teams is None:
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    M = np.zeros((len(teams), len(teams)))
    b = np.zeros(len(teams))
    for row in matches.itertuples(index=False):
        i, j = idx[row.home_team], idx[row.away_team]
        diff = float(row.home_goals - row.away_goals)
        M[i, i] += 1
        M[j, j] += 1
        M[i, j] -= 1
        M[j, i] -= 1
        b[i] += diff
        b[j] -= diff
    M[-1, :] = 1.0
    b[-1] = 0.0
    r = np.linalg.lstsq(M, b, rcond=None)[0]
    return pd.Series(r, index=teams, name="massey")


def colley_ratings(matches: pd.DataFrame, *, teams: list[int] | None = None) -> pd.Series:
    """Colley linear ratings with draws as half-wins."""
    if teams is None:
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    C = 2.0 * np.eye(len(teams))
    b = np.ones(len(teams))
    for row in matches.itertuples(index=False):
        i, j = idx[row.home_team], idx[row.away_team]
        C[i, i] += 1
        C[j, j] += 1
        C[i, j] -= 1
        C[j, i] -= 1
        if row.home_goals > row.away_goals:
            b[i] += 0.5
            b[j] -= 0.5
        elif row.home_goals < row.away_goals:
            b[i] -= 0.5
            b[j] += 0.5
    r = np.linalg.solve(C, b)
    return pd.Series(r, index=teams, name="colley")


def pagerank_results(g: nx.DiGraph, *, alpha: float = 0.85) -> pd.Series:
    """Perron-Frobenius/PageRank rating over the result graph."""
    if len(g) == 0:
        return pd.Series(dtype=float, name="pagerank")
    pr = nx.pagerank(g, alpha=alpha, weight="weight")
    return pd.Series(pr, name="pagerank").sort_values(ascending=False)


def competitiveness_spectral_gap(g: nx.DiGraph) -> float:
    """Spectral gap of the row-stochastic result matrix."""
    nodes = list(g.nodes())
    if len(nodes) < 2:
        return 0.0
    A = nx.to_numpy_array(g, nodelist=nodes, weight="weight")
    row = A.sum(axis=1, keepdims=True)
    P = np.where(row > 0, A / row, 1.0 / len(nodes))
    eig = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return float(eig[0] - eig[1]) if len(eig) > 1 else 0.0
