"""Latent contested-action skill via Item Response Theory (v3 Part B.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from fas.entities import PlayerSeason


@dataclass(slots=True)
class IRTResult:
    """Fitted two-parameter IRT model."""

    theta: pd.Series
    difficulty: pd.Series
    discrimination: pd.Series
    log_likelihood: float
    method: str = "2PL IRT alternating gradient"
    math: str = "latent-variable item response theory"

    def predict(self, player_id: int, item_id: str | int) -> float:
        a = float(self.discrimination.loc[item_id])
        b = float(self.difficulty.loc[item_id])
        th = float(self.theta.loc[player_id])
        return float(expit(a * (th - b)))


def fit_irt_2pl(
    responses: pd.DataFrame,
    *,
    player_col: str = "player_id",
    item_col: str = "item_id",
    outcome_col: str = "outcome",
    max_iter: int = 400,
    lr: float = 0.03,
    ridge: float = 0.05,
) -> IRTResult:
    """Fit a lightweight 2PL IRT model for duels/dribbles/pressured actions."""
    df = responses[[player_col, item_col, outcome_col]].copy()
    players = sorted(df[player_col].unique())
    items = sorted(df[item_col].unique())
    pidx = {p: i for i, p in enumerate(players)}
    iidx = {it: j for j, it in enumerate(items)}
    p = df[player_col].map(pidx).to_numpy()
    it = df[item_col].map(iidx).to_numpy()
    y = df[outcome_col].astype(float).to_numpy()

    theta = _safe_logit(df.groupby(player_col)[outcome_col].mean().reindex(players).to_numpy())
    b = -_safe_logit(df.groupby(item_col)[outcome_col].mean().reindex(items).to_numpy())
    a = np.ones(len(items))

    for _ in range(max_iter):
        z = a[it] * (theta[p] - b[it])
        prob = expit(np.clip(z, -30.0, 30.0))
        resid = y - prob
        g_theta = np.zeros_like(theta)
        g_b = np.zeros_like(b)
        g_a = np.zeros_like(a)
        np.add.at(g_theta, p, a[it] * resid)
        np.add.at(g_b, it, -a[it] * resid)
        np.add.at(g_a, it, (theta[p] - b[it]) * resid)
        theta += lr * (g_theta - ridge * theta) / max(len(y), 1)
        b += lr * (g_b - ridge * b) / max(len(y), 1)
        a += lr * (g_a - ridge * (a - 1.0)) / max(len(y), 1)
        a = np.clip(a, 0.2, 5.0)
        theta -= theta.mean()

    ll = float(np.sum(y * np.log(prob + 1e-12) + (1.0 - y) * np.log(1.0 - prob + 1e-12)))
    return IRTResult(
        theta=pd.Series(theta, index=players, name="theta"),
        difficulty=pd.Series(b, index=items, name="difficulty"),
        discrimination=pd.Series(a, index=items, name="discrimination"),
        log_likelihood=ll,
    )


def enrich(player: PlayerSeason, result: IRTResult) -> PlayerSeason:
    """Attach IRT ability to a player entity."""
    if player.player_uid not in result.theta.index:
        return player
    payload = {
        "theta": float(result.theta.loc[player.player_uid]),
        "method": result.method,
        "math": result.math,
    }
    perf = dict(player.performance)
    perf["irt"] = payload
    return player.with_updates(irt_skill=payload, performance=perf)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), 0.02, 0.98))
