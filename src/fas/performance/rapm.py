"""Regularized adjusted plus-minus for football (v3 Part B.1).

Stints with fixed on-pitch players become rows of a signed design matrix
``Z``. The response is a value differential such as goal differential per
possession, delta EPV, or VAEP/SPADL action value:

    y_k = sum_i alpha_i z_ki + eps_k

The estimator is ridge/Tikhonov regression, with lambda selected by generalized
cross-validation when not supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.entities import PlayerSeason


@dataclass(slots=True)
class RAPMResult:
    """Fitted RAPM impacts and diagnostics."""

    player_ids: list[int]
    alpha: pd.Series
    lambda_: float
    fitted: np.ndarray
    gcv: pd.Series
    method: str = "ridge RAPM"
    math: str = "penalized least squares / Gaussian MAP"


def stint_design_matrix(
    stints: pd.DataFrame,
    *,
    player_ids: list[int] | None = None,
    home_col: str = "home_players",
    away_col: str = "away_players",
) -> tuple[np.ndarray, list[int]]:
    """Build signed player-presence design matrix from stint rows."""
    if player_ids is None:
        players: set[int] = set()
        for col in (home_col, away_col):
            for vals in stints[col]:
                players.update(int(v) for v in vals)
        player_ids = sorted(players)
    idx = {p: i for i, p in enumerate(player_ids)}
    Z = np.zeros((len(stints), len(player_ids)))
    for r, row in enumerate(stints.itertuples(index=False)):
        home = getattr(row, home_col)
        away = getattr(row, away_col)
        for p in home:
            if int(p) in idx:
                Z[r, idx[int(p)]] += 1.0
        for p in away:
            if int(p) in idx:
                Z[r, idx[int(p)]] -= 1.0
    return Z, player_ids


def fit_rapm(
    stints: pd.DataFrame,
    *,
    player_ids: list[int] | None = None,
    response_col: str = "value_diff",
    home_col: str = "home_players",
    away_col: str = "away_players",
    weight_col: str | None = None,
    lambda_: float | None = None,
    lambdas: np.ndarray | None = None,
) -> RAPMResult:
    """Fit ridge RAPM on football stints."""
    Z, players = stint_design_matrix(stints, player_ids=player_ids, home_col=home_col, away_col=away_col)
    y = stints[response_col].to_numpy(dtype=float)
    if weight_col is not None:
        w = np.sqrt(stints[weight_col].to_numpy(dtype=float))
        Z = Z * w[:, None]
        y = y * w
    if lambdas is None:
        lambdas = np.logspace(-3, 3, 25)
    if lambda_ is None:
        scores = {float(lam): _gcv_score(Z, y, float(lam)) for lam in lambdas}
        lambda_ = min(scores, key=scores.get)
    else:
        scores = {float(lambda_): _gcv_score(Z, y, float(lambda_))}

    alpha = _ridge_solution(Z, y, float(lambda_))
    fitted = Z @ alpha
    return RAPMResult(
        player_ids=players,
        alpha=pd.Series(alpha, index=players, name="rapm"),
        lambda_=float(lambda_),
        fitted=fitted,
        gcv=pd.Series(scores, name="gcv"),
    )


def action_rapm(
    actions: pd.DataFrame,
    values: pd.Series | np.ndarray,
    *,
    player_ids: list[int] | None = None,
    lambda_: float | None = None,
) -> RAPMResult:
    """Action-RAPM where each action contributes one player-indicator row."""
    vals = pd.Series(values, index=actions.index)
    rows = pd.DataFrame({
        "home_players": actions["player_id"].map(lambda p: [int(p)]),
        "away_players": [[] for _ in range(len(actions))],
        "value_diff": vals.loc[actions.index].to_numpy(dtype=float),
    })
    return fit_rapm(rows, player_ids=player_ids, lambda_=lambda_)


def enrich(player: PlayerSeason, result: RAPMResult) -> PlayerSeason:
    """Attach RAPM impact to a :class:`PlayerSeason`."""
    value = result.alpha.get(player.player_uid)
    perf = dict(player.performance)
    perf["rapm"] = {
        "value": None if value is None else float(value),
        "lambda": result.lambda_,
        "math": result.math,
    }
    return player.with_updates(rapm=None if value is None else float(value), performance=perf)


def _ridge_solution(Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    p = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ y)


def _gcv_score(Z: np.ndarray, y: np.ndarray, lam: float) -> float:
    if Z.size == 0:
        return 0.0
    alpha = _ridge_solution(Z, y, lam)
    resid = y - Z @ alpha
    s = np.linalg.svd(Z, compute_uv=False)
    df = float(np.sum(s**2 / (s**2 + lam)))
    n = max(len(y), 1)
    denom = max((n - df) / n, 1e-8)
    return float(np.mean(resid**2) / denom**2)
