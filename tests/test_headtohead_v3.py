"""Tests for v3 head-to-head models."""

import numpy as np
import pandas as pd

from fas.headtohead import (
    colley_ratings,
    cp_factorize,
    fit_bradley_terry_davidson,
    fit_gaussian_copula,
    fit_hawkes,
    massey_ratings,
    pagerank_results,
    results_graph,
)
from fas.headtohead.copula_outcomes import scoreline_distribution


def test_bradley_terry_davidson_orders_winner():
    rows = []
    for _ in range(20):
        rows.append({"team_i": 1, "team_j": 2, "outcome": "win", "home": 1.0})
        rows.append({"team_i": 2, "team_j": 3, "outcome": "win", "home": 0.0})
    model = fit_bradley_terry_davidson(pd.DataFrame(rows))
    assert model.beta.loc[1] > model.beta.loc[3]
    assert abs(sum(model.probabilities(1, 2).values()) - 1.0) < 1e-8


def test_network_rankings_and_tensor_run():
    matches = pd.DataFrame({
        "home_team": [1, 2, 1],
        "away_team": [2, 3, 3],
        "home_goals": [2, 2, 3],
        "away_goals": [0, 0, 1],
    })
    g = results_graph(matches)
    assert pagerank_results(g).sum() > 0
    assert massey_ratings(matches).loc[1] > massey_ratings(matches).loc[3]
    assert colley_ratings(matches).loc[1] > colley_ratings(matches).loc[3]

    tensor = np.ones((3, 3, 2))
    factors = cp_factorize(tensor, rank=1, n_iter=20)
    assert factors.reconstruction_error < 0.2


def test_copula_and_hawkes_run():
    data = pd.DataFrame({
        "home_xg": np.linspace(0.2, 2.5, 40),
        "away_xg": np.linspace(2.0, 0.4, 40),
        "possession": np.linspace(0.45, 0.65, 40),
    })
    cop = fit_gaussian_copula(data)
    sample = cop.sample(20)
    assert list(sample.columns) == ["home_xg", "away_xg", "possession"]
    scores = scoreline_distribution(cop, n=200, max_goals=5)
    assert abs(scores["prob"].sum() - 1.0) < 1e-9

    hawkes = fit_hawkes(np.array([10.0, 20.0, 21.0, 50.0]), horizon=90.0)
    assert hawkes.branching_ratio >= 0.0
