"""Pitch control as a differential field (v3 Part C.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from fas.data.schema import PITCH_LENGTH, PITCH_WIDTH
from fas.entities import TeamSeason


@dataclass(slots=True)
class PitchControlSurface:
    """Continuous pitch-control grid and gradients."""

    x: np.ndarray
    y: np.ndarray
    control: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    team_id: int
    method: str = "Spearman-style logistic pitch control"
    math: str = "potential fields / logistic spatial model / vector calculus"

    def value_at(self, x: float, y: float) -> float:
        i = int(np.clip(np.searchsorted(self.x, x), 0, len(self.x) - 1))
        j = int(np.clip(np.searchsorted(self.y, y), 0, len(self.y) - 1))
        return float(self.control[i, j])


def pitch_control_surface(
    players: pd.DataFrame,
    *,
    team_id: int,
    x_col: str = "x",
    y_col: str = "y",
    team_col: str = "team_id",
    grid_x: int = 24,
    grid_y: int = 16,
    max_speed: float = 5.5,
    tau: float = 0.45,
) -> PitchControlSurface:
    """Estimate a logistic control field from freeze-frame/player positions."""
    xs = np.linspace(0.0, PITCH_LENGTH, grid_x)
    ys = np.linspace(0.0, PITCH_WIDTH, grid_y)
    grid = np.zeros((grid_x, grid_y))
    atk = players[players[team_col] == team_id]
    dfn = players[players[team_col] != team_id]
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ta = _min_arrival(atk, x, y, x_col, y_col, max_speed)
            td = _min_arrival(dfn, x, y, x_col, y_col, max_speed)
            grid[i, j] = expit((td - ta) / tau)
    gx, gy = np.gradient(grid, xs, ys, edge_order=1)
    return PitchControlSurface(xs, ys, grid, gx, gy, team_id=int(team_id))


def integrate_control_xt(surface: PitchControlSurface, xt_model) -> float:
    """Integrate pitch control against an xT surface."""
    total = 0.0
    for i, x in enumerate(surface.x):
        for j, y in enumerate(surface.y):
            total += surface.control[i, j] * xt_model.value(float(x), float(y))
    return float(total / surface.control.size)


def enrich(team: TeamSeason, surface: PitchControlSurface, *, xt_model=None) -> TeamSeason:
    """Attach pitch-control value to a team entity."""
    value = float(surface.control.mean()) if xt_model is None else integrate_control_xt(surface, xt_model)
    perf = dict(team.performance)
    perf["pitch_control"] = {"value": value, "math": surface.math}
    return team.with_updates(pitch_control_value=value, performance=perf)


def _min_arrival(players: pd.DataFrame, x: float, y: float, x_col: str, y_col: str, max_speed: float) -> float:
    if players.empty:
        return 1e6
    xy = players[[x_col, y_col]].to_numpy(dtype=float)
    dist = np.sqrt((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2)
    return float(dist.min() / max(max_speed, 1e-6))
