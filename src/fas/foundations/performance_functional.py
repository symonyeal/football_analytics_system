"""Performance measure space (v3 Part A.2).

All downstream ratings are projections or estimators of the same object:

    Pi(e, W) = int_W phi(s) dV_e(s)

where ``V_e`` is the entity contribution measure and ``phi`` is a value density
such as an xT/EPV gradient. This file implements the shared finite-sample
version used by the v3 modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from fas.entities import PlayerSeason, TeamSeason

ValueDensity = Callable[[pd.Series], float]


@dataclass(slots=True)
class ContributionMeasure:
    """Finite measure over value-generating actions for one entity."""

    entity_id: int
    action_index: np.ndarray
    weights: np.ndarray
    window: tuple[float, float] | None = None

    @property
    def mass(self) -> float:
        return float(self.weights.sum())


@dataclass(slots=True)
class PerformanceEstimate:
    """Estimate of ``Pi(e, W)`` plus method metadata."""

    entity_id: int
    value: float
    mass: float
    method: str = "finite performance functional"
    math: str = "measure-theoretic performance functional"


def value_density_from_xt(xt_model) -> ValueDensity:
    """Return ``phi(s) = xT(end) - xT(start)`` for completed movements."""

    def density(row: pd.Series) -> float:
        if pd.isna(row.get("x_end")) or pd.isna(row.get("y_end")):
            return 0.0
        return float(
            xt_model.value(row["x_end"], row["y_end"])
            - xt_model.value(row["x_start"], row["y_start"])
        )

    return density


def entity_contribution_measure(
    actions: pd.DataFrame,
    entity_id: int,
    *,
    entity_col: str = "player_id",
    weight_col: str | None = None,
    t_lo_ms: int | None = None,
    t_hi_ms: int | None = None,
) -> ContributionMeasure:
    """Build ``V_e`` as weighted mass on actions belonging to one entity."""
    df = actions[actions[entity_col] == entity_id]
    if t_lo_ms is not None:
        df = df[df["timestamp_ms"] >= t_lo_ms]
    if t_hi_ms is not None:
        df = df[df["timestamp_ms"] < t_hi_ms]
    if weight_col is None:
        weights = np.ones(len(df), dtype=float)
    else:
        weights = df[weight_col].to_numpy(dtype=float)
    window = None if t_lo_ms is None and t_hi_ms is None else (float(t_lo_ms or 0), float(t_hi_ms or 0))
    return ContributionMeasure(
        entity_id=int(entity_id),
        action_index=df.index.to_numpy(),
        weights=weights,
        window=window,
    )


def performance_functional(
    actions: pd.DataFrame,
    measure: ContributionMeasure,
    value_density: ValueDensity | pd.Series | np.ndarray,
) -> PerformanceEstimate:
    """Evaluate the finite-sample performance functional."""
    if len(measure.action_index) == 0:
        return PerformanceEstimate(measure.entity_id, 0.0, 0.0)
    rows = actions.loc[measure.action_index]
    if callable(value_density):
        phi = rows.apply(value_density, axis=1).to_numpy(dtype=float)
    else:
        phi = pd.Series(value_density).loc[measure.action_index].to_numpy(dtype=float)
    value = float(np.dot(measure.weights, phi))
    return PerformanceEstimate(entity_id=measure.entity_id, value=value, mass=measure.mass)


def enrich(obj: PlayerSeason | TeamSeason, estimate: PerformanceEstimate) -> PlayerSeason | TeamSeason:
    """Attach a performance-functional estimate to an entity record."""
    perf = dict(obj.performance)
    perf["performance_functional"] = {
        "value": estimate.value,
        "mass": estimate.mass,
        "method": estimate.method,
        "math": estimate.math,
    }
    return obj.with_updates(performance=perf)
