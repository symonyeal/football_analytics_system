"""Canonical entity spine for v2/v3 enrichment.

The v3 performance modules all write into these immutable records. That keeps
the system composable: graph, flow, valuation, inference, UI, and MILP code pass
the same objects around and each module adds only the fields it owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd


def _series() -> pd.Series:
    return pd.Series(dtype=float)


def _frame() -> pd.DataFrame:
    return pd.DataFrame()


def _array() -> np.ndarray:
    return np.array([], dtype=float)


@dataclass(frozen=True, slots=True)
class MatchMeta:
    """Small serializable metadata record for a match."""

    competition: str | None = None
    season: str | None = None
    date: str | None = None
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatchObject:
    """One canonical match plus all match-level enrichments."""

    match_id: int
    actions: pd.DataFrame = field(default_factory=_frame)
    pass_networks: dict[int, Any] = field(default_factory=dict)
    centrality: dict[int, pd.DataFrame] = field(default_factory=dict)
    xt_added: pd.Series = field(default_factory=_series)
    zone_flow: dict[int, Any] = field(default_factory=dict)
    epv_timeline: pd.DataFrame | None = None
    intensity_surface: Any | None = None
    momentum: dict[str, Any] = field(default_factory=dict)
    shape_features: dict[int, Any] = field(default_factory=dict)
    insights: list[Any] = field(default_factory=list)
    meta: MatchMeta = field(default_factory=MatchMeta)

    def with_updates(self, **updates: Any) -> "MatchObject":
        """Return a copy with selected fields replaced."""
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class PlayerSeason:
    """One player-season record consumed by valuation, scouting, and MILP."""

    player_uid: int
    league: str = ""
    season: str = ""
    team_id: int | None = None
    position: str | None = None
    minutes: int = 0
    features_90: pd.Series = field(default_factory=_series)
    graph_features: pd.Series = field(default_factory=_series)
    epv_added_90: float = 0.0
    pvs: float = 0.0
    pvs_distribution: np.ndarray = field(default_factory=_array)
    fair_value_eur: float = 0.0
    market_value_eur: float = 0.0
    dev_curve: tuple[float, float, float] | None = None
    rapm: float | None = None
    skill_posterior: dict[str, Any] = field(default_factory=dict)
    irt_skill: dict[str, float] = field(default_factory=dict)
    form_state: pd.DataFrame = field(default_factory=_frame)
    role_membership: pd.Series = field(default_factory=_series)
    performance: dict[str, Any] = field(default_factory=dict)
    insights: list[Any] = field(default_factory=list)

    def with_updates(self, **updates: Any) -> "PlayerSeason":
        """Return a copy with selected fields replaced."""
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class TeamSeason:
    """One team-season record consumed by match models and UI views."""

    team_id: int
    league: str = ""
    season: str = ""
    bt_strength: float = 0.0
    formation_markov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    squad: list[int] = field(default_factory=list)
    attack_t: pd.Series = field(default_factory=_series)
    defense_t: pd.Series = field(default_factory=_series)
    style_manifold_coord: np.ndarray = field(default_factory=_array)
    mdp_value: float | None = None
    pitch_control_value: float | None = None
    performance: dict[str, Any] = field(default_factory=dict)
    insights: list[Any] = field(default_factory=list)

    def with_updates(self, **updates: Any) -> "TeamSeason":
        """Return a copy with selected fields replaced."""
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class Matchup:
    """Head-to-head intelligence for two entities in one context."""

    entity_i: int
    entity_j: int
    context: str = "overall"
    predicted_distribution: pd.DataFrame = field(default_factory=_frame)
    paired_comparison: dict[str, Any] = field(default_factory=dict)
    network_ranking: dict[str, Any] = field(default_factory=dict)
    tensor_factors: dict[str, Any] = field(default_factory=dict)
    copula: dict[str, Any] = field(default_factory=dict)
    hawkes: dict[str, Any] = field(default_factory=dict)
    explanations: list[Any] = field(default_factory=list)

    def with_updates(self, **updates: Any) -> "Matchup":
        """Return a copy with selected fields replaced."""
        return replace(self, **updates)


def enrich(obj: Any, **updates: Any) -> Any:
    """Generic adapter used by modules that only need to replace fields."""
    if hasattr(obj, "with_updates"):
        return obj.with_updates(**updates)
    return replace(obj, **updates)
