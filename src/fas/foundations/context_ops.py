"""Context operators for adjusted performance (v3 Part A.3).

Raw performance is a measure. Context adjustment is represented as composing
linear operators over that measure: opponent strength, game state, venue, and
fatigue. In finite samples these are diagonal weight operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from fas.foundations.performance_functional import PerformanceEstimate


@dataclass(slots=True)
class ContextOperator:
    """A linear context operator represented by per-row multipliers."""

    name: str
    weights: Callable[[pd.DataFrame], np.ndarray]
    math: str = "linear operator on a contribution measure"

    def apply(self, values: pd.Series | np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.weights(frame)


def opponent_strength_operator(strength: dict[int, float], *, team_col: str = "opponent_team_id") -> ContextOperator:
    """Operator ``S``: rescale actions by opponent Bradley-Terry strength."""

    def weights(frame: pd.DataFrame) -> np.ndarray:
        if team_col not in frame:
            return np.ones(len(frame))
        raw = frame[team_col].map(lambda t: np.exp(strength.get(int(t), 0.0))).to_numpy(dtype=float)
        return raw / (raw.mean() if raw.size and raw.mean() > 0 else 1.0)

    return ContextOperator("opponent_strength", weights)


def game_state_operator(*, score_col: str = "score_diff", scale: float = 0.08) -> ContextOperator:
    """Operator ``G``: down/up-weight actions by score state."""

    def weights(frame: pd.DataFrame) -> np.ndarray:
        if score_col not in frame:
            return np.ones(len(frame))
        return np.exp(-scale * frame[score_col].to_numpy(dtype=float))

    return ContextOperator("game_state", weights)


def venue_operator(*, venue_col: str = "venue", away_multiplier: float = 1.04) -> ContextOperator:
    """Operator ``H``: small away-context adjustment."""

    def weights(frame: pd.DataFrame) -> np.ndarray:
        if venue_col not in frame:
            return np.ones(len(frame))
        return np.where(frame[venue_col].astype(str).str.lower().to_numpy() == "away", away_multiplier, 1.0)

    return ContextOperator("venue", weights)


def fatigue_operator(*, minute_col: str = "minute", k_f: float = 0.004) -> ContextOperator:
    """Operator ``F``: Banister-style exponential fatigue correction."""

    def weights(frame: pd.DataFrame) -> np.ndarray:
        if minute_col in frame:
            minute = frame[minute_col].to_numpy(dtype=float)
        elif "timestamp_ms" in frame:
            minute = frame["timestamp_ms"].to_numpy(dtype=float) / 60_000.0
        else:
            minute = np.zeros(len(frame))
        return np.exp(k_f * np.maximum(minute - 60.0, 0.0))

    return ContextOperator("fatigue", weights, math="Banister impulse-response fatigue operator")


def compose_context(*ops: ContextOperator) -> ContextOperator:
    """Compose context operators in the supplied order."""

    def weights(frame: pd.DataFrame) -> np.ndarray:
        w = np.ones(len(frame))
        for op in ops:
            w *= op.weights(frame)
        return w

    name = " ".join(op.name for op in ops) or "identity"
    return ContextOperator(name=name, weights=weights)


def adjusted_performance(
    values: pd.Series | np.ndarray,
    frame: pd.DataFrame,
    estimate: PerformanceEstimate,
    *ops: ContextOperator,
) -> PerformanceEstimate:
    """Apply ``S G H F`` to action-level values and return adjusted ``Pi``."""
    operator = compose_context(*ops)
    adjusted = operator.apply(values, frame)
    return PerformanceEstimate(
        entity_id=estimate.entity_id,
        value=float(np.sum(adjusted)),
        mass=estimate.mass,
        method=f"adjusted by {operator.name}",
        math="composition of linear context operators",
    )
