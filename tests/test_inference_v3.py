"""Tests for v3 higher-order inference and insight extraction."""

import numpy as np
import pandas as pd

from fas.inference import (
    benjamini_hochberg,
    clean_covariance,
    granger_causality,
    mmd_permutation_test,
    persistence_features,
    render_insight,
    scan_departures,
    sinkhorn_distance,
)
from fas.inference.ot_style import grid_cost


def test_tda_rmt_ot_and_mmd_run():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(8, 2))
    pers = persistence_features(pts)
    assert "compactness" in pers.features

    X = rng.normal(size=(80, 5))
    cov = clean_covariance(X)
    assert cov.shape == (5, 5)

    a = np.array([0.6, 0.4])
    b = np.array([0.2, 0.8])
    assert sinkhorn_distance(a, b, grid_cost(2)) >= 0.0

    test = mmd_permutation_test(rng.normal(size=(20, 2)), rng.normal(loc=0.5, size=(20, 2)), n_perm=20)
    assert 0.0 <= test["p_value"] <= 1.0


def test_causality_and_insights_run():
    t = np.arange(80)
    x = np.sin(t / 5)
    y = np.roll(x, 1) + np.random.default_rng(1).normal(scale=0.05, size=len(t))
    stat = granger_causality(x, y, lag=2)
    assert stat["f_stat"] >= 0.0

    frame = pd.DataFrame({
        "player_id": [1] * 12 + [2] * 12,
        "metric": [1.0] * 12 + [0.0] * 12,
        "expected": [0.0] * 24,
    })
    insights = scan_departures(frame, entity_col="player_id", metric_col="metric", expected_col="expected")
    assert any(i.entity_id == 1 for i in insights)
    assert "method:" in render_insight(insights[0])

    q, reject = benjamini_hochberg(np.array([0.001, 0.2]))
    assert reject[0] and q[0] <= q[1]
