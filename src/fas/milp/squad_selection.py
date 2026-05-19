"""Squad selection MILP (Part 4.2).

Maximize squad quality
    sum_i sum_p w_p PVS_i y_ip + gamma * NetworkBonus(x)
subject to positional coverage, starting XI, squad size, budget, age, quota,
formation choice, role flexibility, and the linearized network bonus.

Solved with PuLP + CBC (bundled, open-source). The bilinear products
``x_i x_j`` (NetworkBonus, C11) and ``delta_f n_p(f)`` (formation, C8) are
linearized exactly per the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

ROLES: tuple[str, ...] = (
    "GK", "LCB", "RCB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST",
)


@dataclass(frozen=True, slots=True)
class Formation:
    """A formation as a multiset of roles → required count."""

    name: str
    counts: dict[str, int]

    def total(self) -> int:
        return sum(self.counts.values())


# Standard formations; each sums to 11 starters.
FORMATIONS: dict[str, Formation] = {
    "4-3-3": Formation("4-3-3", {
        "GK": 1, "LCB": 1, "RCB": 1, "LB": 1, "RB": 1,
        "CDM": 1, "CM": 2, "LW": 1, "RW": 1, "ST": 1,
    }),
    "4-2-3-1": Formation("4-2-3-1", {
        "GK": 1, "LCB": 1, "RCB": 1, "LB": 1, "RB": 1,
        "CDM": 2, "CAM": 1, "LW": 1, "RW": 1, "ST": 1,
    }),
    "3-5-2": Formation("3-5-2", {
        "GK": 1, "LCB": 1, "RCB": 1, "CDM": 1, "CM": 2,
        "CAM": 1, "LB": 1, "RB": 1, "ST": 2,
    }),
}


@dataclass(slots=True)
class SquadProblem:
    """Inputs to the squad-selection MILP.

    players : index of candidate player_ids
    pvs : Series PVS_i in (0,1)
    eligible_roles : {player_id: set(roles)} feasibility sets (C10)
    wage, fair_value : Series in chosen currency units
    age : Series of ages
    homegrown : Series[bool]
    compat : (n,n) DataFrame of pairwise style-compatibility (NetworkBonus)
    role_weights : {role: w_p}; defaults to 1.0
    """

    players: list[int]
    pvs: pd.Series
    eligible_roles: dict[int, set[str]]
    wage: pd.Series
    fair_value: pd.Series
    age: pd.Series
    homegrown: pd.Series
    compat: pd.DataFrame | None = None
    role_weights: dict[str, float] = field(default_factory=dict)

    # constraint parameters
    squad_size: int = 23
    wage_cap: float = float("inf")
    transfer_budget: float = float("inf")
    young_age: int = 23
    min_young: int = 0
    old_age: int = 30
    max_old: int = 99
    min_homegrown: int = 0
    allowed_formations: tuple[str, ...] = ("4-3-3", "4-2-3-1", "3-5-2")
    network_gamma: float = 0.0


@dataclass(slots=True)
class SquadSolution:
    status: str
    objective: float
    selected: list[int]
    starters: list[int]
    role_assignment: dict[int, str]
    formation: str
    solve_seconds: float


def solve_squad(prob: SquadProblem, *, time_limit: int = 300) -> SquadSolution:
    """Build and solve the squad-selection MILP (Part 4.2)."""
    P = prob.players
    roles = list(ROLES)
    w = {p: prob.role_weights.get(p, 1.0) for p in roles}

    m = pulp.LpProblem("squad_selection", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in P}
    z = {i: pulp.LpVariable(f"z_{i}", cat="Binary") for i in P}     # starts
    y = {
        (i, r): pulp.LpVariable(f"y_{i}_{r}", cat="Binary")
        for i in P for r in roles
        if r in prob.eligible_roles.get(i, set())                  # C10
    }
    delta = {
        f: pulp.LpVariable(f"d_{f}", cat="Binary") for f in prob.allowed_formations
    }

    # --- objective ---------------------------------------------------------
    quality = pulp.lpSum(
        w[r] * float(prob.pvs.get(i, 0.0)) * y[(i, r)] for (i, r) in y
    )
    obj = quality

    # NetworkBonus(x) with C11 linearization of x_i x_j.
    if prob.network_gamma and prob.compat is not None:
        q = {}
        bonus_terms = []
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                i, j = P[a], P[b]
                cij = float(prob.compat.loc[i, j]) if i in prob.compat.index else 0.0
                if cij == 0:
                    continue
                qij = pulp.LpVariable(f"q_{i}_{j}", cat="Binary")
                q[(i, j)] = qij
                m += qij <= x[i]
                m += qij <= x[j]
                m += qij >= x[i] + x[j] - 1
                bonus_terms.append(cij * qij)
        obj = quality + prob.network_gamma * pulp.lpSum(bonus_terms)

    m += obj

    # --- constraints -------------------------------------------------------
    # C2 starting XI; C3 squad size.
    m += pulp.lpSum(z[i] for i in P) == 11, "C2_starting_xi"
    m += pulp.lpSum(x[i] for i in P) == prob.squad_size, "C3_squad_size"

    # link selection / start / bench: z_i <= x_i.
    for i in P:
        m += z[i] <= x[i], f"link_z_x_{i}"

    # C4 role-selection link: a role assignment requires selection and a start.
    for (i, r) in y:
        m += y[(i, r)] <= x[i]
        m += y[(i, r)] <= z[i]
    # each starter takes exactly one role; each selected non-starter none.
    for i in P:
        roles_i = [y[(i, r)] for r in roles if (i, r) in y]
        if roles_i:
            m += pulp.lpSum(roles_i) == z[i], f"one_role_{i}"
        else:
            m += z[i] == 0, f"no_eligible_role_{i}"

    # C8 formation choice + role-count coupling (big-M-free: equality per role).
    m += pulp.lpSum(delta[f] for f in prob.allowed_formations) == 1, "C8_one_formation"
    for r in roles:
        assigned = pulp.lpSum(y[(i, r)] for i in P if (i, r) in y)
        required = pulp.lpSum(
            FORMATIONS[f].counts.get(r, 0) * delta[f] for f in prob.allowed_formations
        )
        m += assigned == required, f"C8_role_{r}"

    # C5 budget: wage bill + transfer budget on the *selected* squad.
    # Skip caps left at +inf (PuLP rejects inf/NaN RHS).
    if np.isfinite(prob.wage_cap):
        m += pulp.lpSum(float(prob.wage.get(i, 0.0)) * x[i] for i in P) <= prob.wage_cap, "C5_wage"
    if np.isfinite(prob.transfer_budget):
        m += (
            pulp.lpSum(float(prob.fair_value.get(i, 0.0)) * x[i] for i in P)
            <= prob.transfer_budget
        ), "C5_transfer"

    # C6 age distribution.
    if prob.min_young > 0:
        m += (
            pulp.lpSum(x[i] for i in P if prob.age.get(i, 0) <= prob.young_age)
            >= prob.min_young
        ), "C6_min_young"
    if prob.max_old < prob.squad_size:
        m += (
            pulp.lpSum(x[i] for i in P if prob.age.get(i, 0) >= prob.old_age)
            <= prob.max_old
        ), "C6_max_old"

    # C7 homegrown quota.
    if prob.min_homegrown > 0:
        m += (
            pulp.lpSum(x[i] for i in P if bool(prob.homegrown.get(i, False)))
            >= prob.min_homegrown
        ), "C7_homegrown"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    m.solve(solver)

    selected = [i for i in P if x[i].value() and x[i].value() > 0.5]
    starters = [i for i in P if z[i].value() and z[i].value() > 0.5]
    role_assignment = {
        i: r for (i, r) in y if y[(i, r)].value() and y[(i, r)].value() > 0.5
    }
    chosen_formation = next(
        (f for f in prob.allowed_formations if delta[f].value() and delta[f].value() > 0.5),
        "",
    )
    return SquadSolution(
        status=pulp.LpStatus[m.status],
        objective=float(pulp.value(m.objective) or 0.0),
        selected=selected,
        starters=starters,
        role_assignment=role_assignment,
        formation=chosen_formation,
        solve_seconds=float(m.solutionTime or 0.0),
    )


def cosine_compat(features: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine-similarity compatibility matrix from feature vectors."""
    X = features.to_numpy(dtype=float)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Xn = X / norm
    S = Xn @ Xn.T
    return pd.DataFrame(S, index=features.index, columns=features.index)
