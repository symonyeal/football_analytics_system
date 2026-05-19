"""Tests for the network flow module (Part 2)."""

from fas.examples.synthetic_pipeline import synthetic_actions
from fas.network_flow import (
    build_zone_graph,
    buildup_potency,
    fit_xt,
    zone_of,
)


def test_zone_of_range():
    assert 0 <= zone_of(0, 0) < 18
    assert 0 <= zone_of(120, 80) < 18
    assert zone_of(0, 0) != zone_of(119, 79)


def test_xt_surface_shape_and_bounds():
    actions = synthetic_actions(seed=3)
    xt = fit_xt(actions, n_x=16, n_y=12)
    assert xt.grid.shape == (16, 12)
    # xT values are probabilities-of-scoring-flavoured: bounded and finite
    assert xt.grid.min() >= 0.0
    assert xt.grid.max() <= 1.0


def test_buildup_potency_runs():
    actions = synthetic_actions(seed=4)
    xt = fit_xt(actions)
    zg = build_zone_graph(actions, team_id=100, xt_model=xt, min_count=1)
    result = buildup_potency(zg)
    assert result.flow_value >= 0
