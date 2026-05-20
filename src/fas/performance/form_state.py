"""Player form as a state-space process (v3 Part B.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.entities import PlayerSeason


@dataclass(slots=True)
class FormStateResult:
    """Kalman filtered and smoothed form trajectories."""

    states: pd.DataFrame
    innovations: pd.DataFrame
    A: float
    process_var: float
    obs_var: float
    method: str = "Kalman filter and Rauch-Tung-Striebel smoother"
    math: str = "linear Gaussian state-space model"


def kalman_form(
    observations: pd.DataFrame,
    *,
    player_col: str = "player_id",
    time_col: str = "t",
    value_col: str = "performance",
    A: float = 0.85,
    process_var: float = 0.08,
    obs_var: float = 0.25,
) -> FormStateResult:
    """Smooth per-player latent form from match performance observations."""
    state_rows = []
    innov_rows = []
    for pid, grp in observations.sort_values(time_col).groupby(player_col):
        y = grp[value_col].to_numpy(dtype=float)
        t = grp[time_col].to_numpy()
        m_f, p_f, innov = _kalman_filter(y, A, process_var, obs_var)
        m_s, p_s = _rts_smoother(m_f, p_f, A, process_var)
        for k, idx in enumerate(grp.index):
            state_rows.append({
                "player_id": int(pid),
                "t": t[k],
                "state": float(m_s[k]),
                "variance": float(p_s[k]),
                "lo95": float(m_s[k] - 1.96 * np.sqrt(max(p_s[k], 0.0))),
                "hi95": float(m_s[k] + 1.96 * np.sqrt(max(p_s[k], 0.0))),
                "index": idx,
            })
            innov_rows.append({"player_id": int(pid), "t": t[k], "innovation": float(innov[k])})
    return FormStateResult(
        states=pd.DataFrame(state_rows),
        innovations=pd.DataFrame(innov_rows),
        A=A,
        process_var=process_var,
        obs_var=obs_var,
    )


def enrich(player: PlayerSeason, result: FormStateResult) -> PlayerSeason:
    """Attach one player's form trajectory to a :class:`PlayerSeason`."""
    sub = result.states[result.states["player_id"] == player.player_uid].copy()
    if sub.empty:
        return player
    perf = dict(player.performance)
    perf["form_state"] = {"latest": float(sub["state"].iloc[-1]), "math": result.math}
    return player.with_updates(form_state=sub, performance=perf)


def _kalman_filter(y: np.ndarray, A: float, q: float, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    m = np.zeros(n)
    p = np.zeros(n)
    innov = np.zeros(n)
    m_prev = 0.0
    p_prev = 1.0
    for k in range(n):
        m_pred = A * m_prev
        p_pred = A * A * p_prev + q
        innov[k] = y[k] - m_pred
        s = p_pred + r
        gain = p_pred / s
        m[k] = m_pred + gain * innov[k]
        p[k] = (1.0 - gain) * p_pred
        m_prev, p_prev = m[k], p[k]
    return m, p, innov


def _rts_smoother(m: np.ndarray, p: np.ndarray, A: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(m)
    ms = m.copy()
    ps = p.copy()
    for k in range(n - 2, -1, -1):
        p_pred = A * A * p[k] + q
        c = p[k] * A / max(p_pred, 1e-12)
        ms[k] = m[k] + c * (ms[k + 1] - A * m[k])
        ps[k] = p[k] + c * c * (ps[k + 1] - p_pred)
    return ms, ps
