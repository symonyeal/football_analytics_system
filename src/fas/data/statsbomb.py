"""StatsBomb open-data loader → canonical action schema (Part 0, Appendix B).

Maps StatsBomb event DataFrames (as returned by ``statsbombpy.sb.events``)
onto the canonical schema in :mod:`fas.data.schema`. ``statsbombpy`` is an
optional import so the rest of the package works without network access.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from fas.data.schema import COLUMNS, validate_actions

# StatsBomb event "type" name → canonical action_type. Unmapped types dropped.
_TYPE_MAP: dict[str, str] = {
    "Pass": "pass",
    "Carry": "carry",
    "Shot": "shot",
    "Dribble": "dribble",
    "Pressure": "pressure",
    "Tackle": "tackle",
    "Interception": "interception",
    "Clearance": "clearance",
    "Block": "block",
    "Ball Recovery": "recovery",
    "Foul Committed": "foul",
    "Goal Keeper": "goalkeeper",
}


def load_competitions():  # pragma: no cover - network
    """Return StatsBomb competitions table (requires ``statsbombpy``)."""
    sb = _import_sb()
    return sb.competitions()


def load_match_events(match_id: int):  # pragma: no cover - network
    """Fetch and canonicalize one match's events from StatsBomb."""
    sb = _import_sb()
    events = sb.events(match_id=match_id)
    return events_to_actions(events, match_id=match_id)


def events_to_actions(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Convert a StatsBomb events DataFrame to the canonical action frame.

    Handles the nested ``location`` / ``pass`` / ``carry`` / ``shot`` columns
    documented in Appendix B. Robust to columns being absent.
    """
    ev = events.copy()
    ev = ev[ev["type"].isin(_TYPE_MAP)].reset_index(drop=True)

    start = ev["location"].apply(_xy)
    ev["x_start"] = start.apply(lambda p: p[0])
    ev["y_start"] = start.apply(lambda p: p[1])

    end = ev.apply(_end_location, axis=1)
    ev["x_end"] = end.apply(lambda p: p[0])
    ev["y_end"] = end.apply(lambda p: p[1])

    ev["action_type"] = ev["type"].map(_TYPE_MAP)
    ev["outcome"] = ev.apply(_outcome, axis=1)

    ev["match_id"] = match_id
    ev["timestamp_ms"] = _timestamp_ms(ev)
    ev["player_id"] = _coalesce_id(ev, "player_id", "player")
    ev["team_id"] = _coalesce_id(ev, "team_id", "team")
    if "period" not in ev:
        ev["period"] = 1

    out = ev[list(COLUMNS)].dropna(subset=["x_start", "y_start", "player_id"])
    out = out.astype({"player_id": "int64", "team_id": "int64", "period": "int64"})
    return validate_actions(out)


# --- helpers ---------------------------------------------------------------

def _import_sb():  # pragma: no cover - network
    try:
        from statsbombpy import sb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "statsbombpy is required for live loading: pip install statsbombpy"
        ) from exc
    return sb


def _xy(loc: Any) -> tuple[Optional[float], Optional[float]]:
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return (np.nan, np.nan)


def _end_location(row: pd.Series) -> tuple[Optional[float], Optional[float]]:
    for col in ("pass", "carry", "shot"):
        val = row.get(col)
        if isinstance(val, dict):
            el = val.get("end_location")
            if isinstance(el, (list, tuple)) and len(el) >= 2:
                return float(el[0]), float(el[1])
    return (np.nan, np.nan)


def _outcome(row: pd.Series) -> bool:
    """Success convention: a pass/action is successful iff no failure outcome.

    StatsBomb encodes failure via a nested ``outcome`` dict; its *absence*
    means success (a completed pass has no ``pass.outcome``).
    """
    val = row.get("pass")
    if isinstance(val, dict):
        return val.get("outcome") is None
    val = row.get("shot")
    if isinstance(val, dict):
        oc = val.get("outcome", {})
        name = oc.get("name") if isinstance(oc, dict) else oc
        return name == "Goal"
    return True


def _timestamp_ms(ev: pd.DataFrame) -> pd.Series:
    minute = pd.to_numeric(ev.get("minute", 0), errors="coerce").fillna(0)
    second = pd.to_numeric(ev.get("second", 0), errors="coerce").fillna(0)
    return ((minute * 60 + second) * 1000).astype("int64")


def _coalesce_id(ev: pd.DataFrame, id_col: str, name_col: str) -> pd.Series:
    """Prefer a numeric id column; fall back to hashing the name string.

    statsbombpy sometimes returns names rather than ids; we synthesize a
    stable integer id from the name so the graph layer always has ints.
    """
    if id_col in ev and pd.api.types.is_numeric_dtype(ev[id_col]):
        return ev[id_col].fillna(-1)
    if name_col in ev:
        return ev[name_col].astype("category").cat.codes
    return pd.Series([-1] * len(ev))
