"""Robust squad selection under uncertainty (Part 4.3).

Player PVS is uncertain; scenarios ``xi^(k)`` carry bootstrapped PVS values.
Minimax-regret objective:

    min_x  max_k [ OPT(xi^k) - Objective(x, xi^k) ]

Reformulated with theta >= r_k:
    min theta  s.t.  r_k = OPT^k - sum_i sum_p w_p PVS_i^k y_ip - gamma*Bonus(x),
                     theta >= r_k for all k,  + all squad constraints.

OPT(xi^k) (the per-scenario optimum) is precomputed by solving the deterministic
squad MILP under each scenario. We then solve the single regret-minimizing MILP
over all scenarios at once — an extensive-form equivalent to the Benders
decomposition in the spec (Benders is the scalable alternative for large K).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from fas.milp.squad_selection import FORMATIONS, ROLES, SquadProblem, solve_squad


@dataclass(slots=True)
class RobustSolution:
    status: str
    max_regret: float
    selected: list[int]
    starters: list[int]
    role_assignment: dict[int, str]
    formation: str


def bootstrap_scenarios(
    pvs_history: pd.DataFrame,
    *,
    n_scenarios: int = 50,
    seed: int = 0,
) -> list[pd.Series]:
    """Bootstrap PVS scenarios from a (player x season) history DataFrame.

    Each scenario resamples, per player, one season's PVS with replacement.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n_scenarios):
        draw = pvs_history.apply(
            lambda row: rng.choice(row.dropna().to_numpy()) if row.notna().any() else 0.0,
            axis=1,
        )
        scenarios.append(draw.rename("PVS"))
    return scenarios


def solve_robust_squad(
    base: SquadProblem,
    scenarios: list[pd.Series],
    *,
    time_limit: int = 300,
) -> RobustSolution:
    """Solve the minimax-regret robust squad MILP (Part 4.3, extensive form)."""
    # 1. Per-scenario optima OPT(xi^k).
    opt_k = []
    for sc in scenarios:
        prob_k = SquadProblem(**{**base.__dict__})
        prob_k.pvs = sc
        opt_k.append(solve_squad(prob_k, time_limit=max(10, time_limit // len(scenarios))).objective)

    # 2. Single regret-minimizing model with shared decision vars.
    P = base.players
    roles = list(ROLES)
    w = {r: base.role_weights.get(r, 1.0) for r in roles}
    m = pulp.LpProblem("robust_squad", pulp.LpMinimize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in P}
    z = {i: pulp.LpVariable(f"z_{i}", cat="Binary") for i in P}
    y = {
        (i, r): pulp.LpVariable(f"y_{i}_{r}", cat="Binary")
        for i in P for r in roles if r in base.eligible_roles.get(i, set())
    }
    delta = {f: pulp.LpVariable(f"d_{f}", cat="Binary") for f in base.allowed_formations}
    theta = pulp.LpVariable("theta", lowBound=0)

    m += theta  # minimize worst-case regret

    for k, sc in enumerate(scenarios):
        obj_k = pulp.lpSum(w[r] * float(sc.get(i, 0.0)) * y[(i, r)] for (i, r) in y)
        m += theta >= opt_k[k] - obj_k, f"regret_{k}"

    _add_structural_constraints(m, base, x, z, y, delta, roles, P)

    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))

    selected = [i for i in P if x[i].value() and x[i].value() > 0.5]
    starters = [i for i in P if z[i].value() and z[i].value() > 0.5]
    ra = {i: r for (i, r) in y if y[(i, r)].value() and y[(i, r)].value() > 0.5}
    formation = next(
        (f for f in base.allowed_formations if delta[f].value() and delta[f].value() > 0.5), ""
    )
    return RobustSolution(
        status=pulp.LpStatus[m.status],
        max_regret=float(theta.value() or 0.0),
        selected=selected, starters=starters, role_assignment=ra, formation=formation,
    )


def _add_structural_constraints(m, base, x, z, y, delta, roles, P):
    """Shared squad constraints (C2-C8) used by every scenario."""
    m += pulp.lpSum(z[i] for i in P) == 11
    m += pulp.lpSum(x[i] for i in P) == base.squad_size
    for i in P:
        m += z[i] <= x[i]
        roles_i = [y[(i, r)] for r in roles if (i, r) in y]
        if roles_i:
            m += pulp.lpSum(roles_i) == z[i]
        else:
            m += z[i] == 0
        for r in roles:
            if (i, r) in y:
                m += y[(i, r)] <= z[i]
    m += pulp.lpSum(delta[f] for f in base.allowed_formations) == 1
    for r in roles:
        assigned = pulp.lpSum(y[(i, r)] for i in P if (i, r) in y)
        required = pulp.lpSum(
            FORMATIONS[f].counts.get(r, 0) * delta[f] for f in base.allowed_formations
        )
        m += assigned == required
    if np.isfinite(base.wage_cap):
        m += pulp.lpSum(float(base.wage.get(i, 0.0)) * x[i] for i in P) <= base.wage_cap
    if np.isfinite(base.transfer_budget):
        m += pulp.lpSum(float(base.fair_value.get(i, 0.0)) * x[i] for i in P) <= base.transfer_budget
    if base.min_young > 0:
        m += pulp.lpSum(x[i] for i in P if base.age.get(i, 0) <= base.young_age) >= base.min_young
    if base.max_old < base.squad_size:
        m += pulp.lpSum(x[i] for i in P if base.age.get(i, 0) >= base.old_age) <= base.max_old
    if base.min_homegrown > 0:
        m += pulp.lpSum(x[i] for i in P if bool(base.homegrown.get(i, False))) >= base.min_homegrown
