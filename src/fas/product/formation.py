"""Formation inference: rule-based and interpretable.

Two signals, kept deliberately separate:

1. **Starting formation** from lineup positions — high confidence. Outfield
   players are grouped into vertical lines by their stable pitch x-coordinate,
   and the line sizes become a string like ``4-3-3``.
2. **Phase formation** from average action locations — lower confidence, and
   labelled as such. Where players actually operated, not where they lined up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.data.schema import PITCH_LENGTH
from fas.product.synthetic import ROLE_COORDS


@dataclass(slots=True)
class FormationResult:
    """An inferred formation plus the per-player coordinates it was built from."""

    formation: str
    confidence: str          # "lineup" (high) or "phase" (lower)
    lines: list[int]         # outfield counts per line, deep -> advanced
    coords: dict[int, tuple[float, float]]


def _group_into_lines(xs: np.ndarray, *, gap: float = 13.0, max_lines: int = 4) -> list[int]:
    """Split sorted x-coordinates into lines wherever a gap exceeds ``gap``."""
    order = np.sort(xs)
    if len(order) == 0:
        return []
    lines = [1]
    for prev, cur in zip(order[:-1], order[1:]):
        if cur - prev > gap:
            lines.append(1)
        else:
            lines[-1] += 1
    # Merge the smallest adjacent lines until within the line budget.
    while len(lines) > max_lines:
        i = int(np.argmin([lines[k] + lines[k + 1] for k in range(len(lines) - 1)]))
        lines[i] += lines.pop(i + 1)
    return lines


def infer_formation(coords: dict[int, tuple[float, float]], *,
                    confidence: str = "lineup") -> FormationResult:
    """Infer a formation string from player coordinates.

    The goalkeeper (deepest player) is dropped; the remaining outfield players
    are grouped into lines along the pitch and reported deep-to-advanced.
    """
    if not coords:
        return FormationResult("unknown", confidence, [], {})
    items = sorted(coords.items(), key=lambda kv: kv[1][0])
    gk = items[0][0]
    outfield = {pid: xy for pid, xy in items if pid != gk}
    xs = np.array([xy[0] for xy in outfield.values()], dtype=float)
    lines = _group_into_lines(xs)
    formation = "-".join(str(n) for n in lines) if lines else "unknown"
    return FormationResult(formation, confidence, lines, coords)


def starting_formation_from_lineup(lineup: pd.DataFrame) -> FormationResult:
    """Infer the starting formation from a lineup table.

    ``lineup`` must have ``player_id`` and ``position`` columns; positions are
    mapped to their stable pitch coordinates in :data:`ROLE_COORDS`.
    """
    coords = {}
    for row in lineup.itertuples(index=False):
        xy = ROLE_COORDS.get(getattr(row, "position", None))
        if xy is not None:
            coords[int(row.player_id)] = xy
    return infer_formation(coords, confidence="lineup")


def phase_formation_from_actions(actions: pd.DataFrame, team_id: int, *,
                                 keeper_id: int | None = None) -> FormationResult:
    """Lower-confidence formation from average on-ball action locations.

    This reflects where players *operated* during the window, which can differ
    from how they lined up. It is explicitly tagged ``confidence="phase"``.
    """
    df = actions[(actions["team_id"] == team_id)]
    if df.empty:
        return FormationResult("unknown", "phase", [], {})
    avg = df.groupby("player_id")[["x_start", "y_start"]].mean()
    coords = {int(pid): (float(r.x_start), float(r.y_start)) for pid, r in avg.iterrows()}
    if keeper_id is not None:
        coords.pop(keeper_id, None)
    elif coords:
        # Drop the single deepest player as the de-facto keeper.
        gk = min(coords, key=lambda p: coords[p][0])
        if coords[gk][0] < 0.25 * PITCH_LENGTH:
            coords.pop(gk, None)
    return infer_formation(coords, confidence="phase")
