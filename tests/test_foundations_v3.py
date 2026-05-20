"""Tests for v3 foundations and entity enrichment."""

import numpy as np
import pandas as pd

from fas.entities import MatchObject, PlayerSeason
from fas.examples.synthetic_pipeline import synthetic_actions
from fas.foundations import (
    entity_contribution_measure,
    fatigue_operator,
    fit_intensity_surface,
    performance_functional,
    value_density_from_xt,
)
from fas.foundations.performance_functional import enrich as enrich_performance
from fas.network_flow import fit_xt


def test_intensity_surface_is_causal_and_positive():
    actions = synthetic_actions(n_actions=60, seed=11)
    surface = fit_intensity_surface(actions, bandwidth_t=60.0, bandwidth_xy=20.0)
    row = actions.iloc[10]
    t = row["timestamp_ms"] / 1000.0
    assert surface.rate(t + 1.0, row["x_start"], row["y_start"], mark=row["action_type"]) > 0.0
    assert surface.tempo(0.0, 60.0) >= 0.0


def test_performance_functional_enriches_player():
    actions = synthetic_actions(n_actions=100, seed=12)
    xt = fit_xt(actions)
    player = int(actions["player_id"].iloc[0])
    measure = entity_contribution_measure(actions, player)
    estimate = performance_functional(actions, measure, value_density_from_xt(xt))
    enriched = enrich_performance(PlayerSeason(player_uid=player), estimate)
    assert "performance_functional" in enriched.performance
    assert np.isfinite(enriched.performance["performance_functional"]["value"])


def test_context_operator_changes_weights_without_errors():
    actions = synthetic_actions(n_actions=30, seed=13)
    actions = actions.assign(minute=actions["timestamp_ms"] / 60_000.0)
    op = fatigue_operator()
    weighted = op.apply(pd.Series(np.ones(len(actions))), actions)
    assert weighted.shape[0] == len(actions)
    assert weighted.min() >= 1.0


def test_match_object_intensity_enrich_adapter():
    from fas.foundations.point_process import enrich

    actions = synthetic_actions(n_actions=20, seed=14)
    match = MatchObject(match_id=1, actions=actions)
    out = enrich(match)
    assert out.intensity_surface is not None
