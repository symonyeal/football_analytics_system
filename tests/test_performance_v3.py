"""Tests for v3 player and team performance models."""

import numpy as np
import pandas as pd

from fas.examples.synthetic_pipeline import synthetic_actions
from fas.performance import (
    fit_dixon_coles,
    fit_hierarchical_skill,
    fit_irt_2pl,
    fit_possession_mdp,
    fit_rapm,
    fit_roles_nmf,
    kalman_form,
    pitch_control_surface,
    team_style_distribution,
)
from fas.performance.style_manifold import fisher_rao_distance


def test_rapm_orders_known_positive_player():
    stints = pd.DataFrame({
        "home_players": [[1, 3], [1, 4], [1, 5], [3, 4]],
        "away_players": [[2, 4], [2, 5], [2, 3], [2, 5]],
        "value_diff": [1.0, 0.8, 1.2, -0.1],
    })
    res = fit_rapm(stints, lambda_=0.1)
    assert res.alpha.loc[1] > res.alpha.loc[2]


def test_skill_irt_form_and_roles_run():
    actions = synthetic_actions(n_actions=180, seed=21)
    posterior = fit_hierarchical_skill(actions, n_samples=30)
    assert "progression" in posterior.mean.columns

    responses = pd.DataFrame({
        "player_id": [1] * 20 + [2] * 20,
        "item_id": ["duel"] * 40,
        "outcome": [1] * 16 + [0] * 4 + [1] * 5 + [0] * 15,
    })
    irt = fit_irt_2pl(responses, max_iter=80)
    assert irt.theta.loc[1] > irt.theta.loc[2]

    obs = pd.DataFrame({"player_id": [1, 1, 1], "t": [1, 2, 3], "performance": [0.1, 0.4, 0.2]})
    form = kalman_form(obs)
    assert len(form.states) == 3

    roles = fit_roles_nmf(actions, n_roles=3)
    assert np.allclose(roles.memberships.sum(axis=1), 1.0)


def test_team_scoring_mdp_pitch_control_and_style():
    matches = pd.DataFrame({
        "home_team": [1, 1, 2, 2, 1, 3],
        "away_team": [2, 3, 1, 3, 3, 2],
        "home_goals": [3, 2, 0, 1, 4, 1],
        "away_goals": [0, 0, 2, 1, 0, 2],
    })
    model = fit_dixon_coles(matches)
    dist = model.score_distribution(1, 2, max_goals=5)
    assert abs(dist["prob"].sum() - 1.0) < 1e-6

    transitions = pd.DataFrame({
        "state": ["build", "build", "final", "final"],
        "next_state": ["final", "turnover", "goal", "turnover"],
    })
    mdp = fit_possession_mdp(transitions)
    assert mdp.values.loc["final"] > mdp.values.loc["build"]

    players = pd.DataFrame({
        "team_id": [1, 1, 2, 2],
        "x": [40.0, 60.0, 45.0, 75.0],
        "y": [30.0, 50.0, 35.0, 55.0],
    })
    surface = pitch_control_surface(players, team_id=1, grid_x=8, grid_y=6)
    assert surface.control.shape == (8, 6)

    actions = synthetic_actions(n_actions=80, seed=22)
    p = team_style_distribution(actions, team_id=100)
    q = team_style_distribution(actions.sample(frac=1.0, random_state=1), team_id=100)
    assert fisher_rao_distance(p, q) < 1e-9
