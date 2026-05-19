"""Canonical action schema (Part 0.2).

The whole system speaks one vocabulary: an *action* on a standardized
StatsBomb pitch ``[0,120] x [0,80]``. Every data source (StatsBomb events,
SPADL, Wyscout) is mapped into this schema so downstream graph / flow /
optimization modules never need to know the provenance.

    A = (t, player_id, team_id, action_type,
         x_start, y_start, x_end, y_end, outcome, freeze_frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Standardized StatsBomb pitch dimensions (metres-ish units used by the feed).
PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0

# Closed vocabulary for action_type. Sources are mapped onto this set.
ACTION_TYPES: tuple[str, ...] = (
    "pass",
    "carry",
    "shot",
    "dribble",
    "pressure",
    "tackle",
    "interception",
    "clearance",
    "block",
    "recovery",
    "foul",
    "goalkeeper",
)


@dataclass(slots=True)
class Action:
    """One on-ball (or pressing) action in canonical coordinates.

    Parameters
    ----------
    match_id, period, timestamp_ms : event location in time
    player_id, team_id : actors
    action_type : one of :data:`ACTION_TYPES`
    x_start, y_start, x_end, y_end : pitch coordinates in [0,120]x[0,80]
    outcome : bool — True == successful (pass completed, tackle won, ...)
    freeze_frame : optional (n, 4) array of (x, y, vx, vy) for visible players
    """

    match_id: int
    period: int
    timestamp_ms: int
    player_id: int
    team_id: int
    action_type: str
    x_start: float
    y_start: float
    x_end: Optional[float] = None
    y_end: Optional[float] = None
    outcome: bool = True
    freeze_frame: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                f"action_type {self.action_type!r} not in {ACTION_TYPES}"
            )
        _check_xy(self.x_start, self.y_start, "start")
        if self.x_end is not None and self.y_end is not None:
            _check_xy(self.x_end, self.y_end, "end")
        if self.freeze_frame is not None:
            ff = np.asarray(self.freeze_frame, dtype=float)
            if ff.ndim != 2 or ff.shape[1] != 4:
                raise ValueError("freeze_frame must have shape (n, 4): (x,y,vx,vy)")
            self.freeze_frame = ff


def _check_xy(x: float, y: float, which: str) -> None:
    if not (0.0 <= x <= PITCH_LENGTH):
        raise ValueError(f"{which} x={x} outside [0,{PITCH_LENGTH}]")
    if not (0.0 <= y <= PITCH_WIDTH):
        raise ValueError(f"{which} y={y} outside [0,{PITCH_WIDTH}]")


# Canonical column order for the tabular form used everywhere downstream.
COLUMNS: tuple[str, ...] = (
    "match_id",
    "period",
    "timestamp_ms",
    "player_id",
    "team_id",
    "action_type",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "outcome",
)


def actions_to_frame(actions: list[Action]) -> pd.DataFrame:
    """Materialize a list of :class:`Action` into a tidy DataFrame.

    ``freeze_frame`` is dropped from the tabular view (kept only in the
    object form / JSONB column); the scalar columns are what the graph and
    flow modules consume.
    """
    rows = [
        tuple(getattr(a, c) for c in COLUMNS)
        for a in actions
    ]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def validate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a DataFrame to the canonical schema.

    Raises ``ValueError`` on missing columns or out-of-range coordinates.
    Returns the frame with dtypes coerced.
    """
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing canonical columns: {sorted(missing)}")

    bad_type = set(df["action_type"].unique()) - set(ACTION_TYPES)
    if bad_type:
        raise ValueError(f"unknown action_type values: {sorted(bad_type)}")

    out = df.copy()
    for c in ("x_start", "y_start", "x_end", "y_end"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Range checks on start coordinates (ends may be NaN for e.g. pressures).
    if ((out["x_start"] < 0) | (out["x_start"] > PITCH_LENGTH)).any():
        raise ValueError("x_start out of range")
    if ((out["y_start"] < 0) | (out["y_start"] > PITCH_WIDTH)).any():
        raise ValueError("y_start out of range")
    out["outcome"] = out["outcome"].astype(bool)
    return out
