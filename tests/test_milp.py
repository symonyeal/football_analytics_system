"""Tests for the MILP valuation + squad-selection module (Part 4)."""

import numpy as np
import pandas as pd

from fas.milp.player_valuation import (
    bradley_terry,
    fair_value_regression,
    low_rank_embedding,
    robust_pca,
)
from fas.milp.squad_selection import ROLES, SquadProblem, cosine_compat, solve_squad


def test_bradley_terry_orders_strength():
    # team 1 beats 2 beats 3 consistently
    rows = []
    for _ in range(20):
        rows += [
            {"team_i": 1, "team_j": 2, "i_won": 1.0},
            {"team_i": 2, "team_j": 3, "i_won": 1.0},
            {"team_i": 1, "team_j": 3, "i_won": 1.0},
        ]
    beta = bradley_terry(pd.DataFrame(rows))
    assert beta[1] > beta[2] > beta[3]


def test_robust_pca_reconstructs():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 6)) @ rng.normal(size=(6, 10))  # rank<=6
    L, S = robust_pca(base)
    assert np.linalg.norm(L + S - base) / np.linalg.norm(base) < 1e-2
    assert low_rank_embedding(L).shape[0] == 40


def test_fair_value_regression_predicts():
    pid = list(range(30))
    rng = np.random.default_rng(1)
    pvs = pd.Series(rng.uniform(0, 1, 30), index=pid)
    age = pd.Series(rng.integers(18, 34, 30), index=pid)
    mv = pd.Series(np.exp(2 + 3 * pvs.to_numpy()), index=pid)
    _, predict = fair_value_regression(pvs, age, mv)
    assert predict(0.9, 25) > predict(0.2, 25)


def _squad_problem(n=26, size=18):
    pid = list(range(1, n + 1))
    rng = np.random.default_rng(7)
    eligible = {i: {ROLES[i % len(ROLES)], ROLES[(i + 3) % len(ROLES)],
                    ROLES[(i + 5) % len(ROLES)]} for i in pid}
    feats = pd.DataFrame(rng.normal(size=(n, 6)), index=pid)
    return SquadProblem(
        players=pid,
        pvs=pd.Series(rng.uniform(0.3, 0.95, n), index=pid),
        eligible_roles=eligible,
        wage=pd.Series(rng.uniform(0.2, 2.0, n), index=pid),
        fair_value=pd.Series(rng.uniform(2, 60, n), index=pid),
        age=pd.Series(rng.integers(18, 34, n), index=pid),
        homegrown=pd.Series(rng.random(n) > 0.6, index=pid),
        compat=cosine_compat(feats),
        squad_size=size, min_young=2,
    )


def test_squad_milp_optimal_and_feasible():
    sol = solve_squad(_squad_problem(), time_limit=30)
    assert sol.status in ("Optimal", "Not Solved", "Feasible")
    if sol.status == "Optimal":
        assert len(sol.selected) == 18
        assert len(sol.starters) == 11
        assert sol.formation in ("4-3-3", "4-2-3-1", "3-5-2")
        # every starter has exactly one role
        assert len(sol.role_assignment) == 11
