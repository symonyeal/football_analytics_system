"""In-match substitution MILP (Part 4.4).

Choose which player to sub off, which substitute to bring on, and at which
minute, to maximize expected EPV improvement over the remaining match:

    max  sum_{t,i,j} P(EPV_improvement | sub j for i at t, state) delta_in[j,t]
    s.t. remaining-subs cap, each player off <=1, each sub on <=1,
         role compatibility, balance of ons and offs.

The improvement probabilities ``P(...)`` are supplied as a precomputed table
(in the spec, a LightGBM model conditioned on minute/scoreline/fatigue/role).
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp


@dataclass(slots=True)
class SubSolution:
    status: str
    expected_gain: float
    moves: list[tuple[int, int, int]]   # (off_player, on_player, minute)


def solve_substitutions(
    on_pitch: list[int],
    bench: list[int],
    minutes: list[int],
    gain: dict[tuple[int, int, int], float],
    *,
    subs_remaining: int = 3,
    compatible: dict[tuple[int, int], bool] | None = None,
) -> SubSolution:
    """Solve the substitution MILP (Part 4.4).

    ``gain[(i, j, t)]`` = expected EPV improvement from replacing on-pitch
    player ``i`` with bench player ``j`` at minute ``t``. ``compatible`` gates
    role-feasible (i, j) pairs (defaults to all-compatible).
    """
    compatible = compatible or {}
    m = pulp.LpProblem("substitutions", pulp.LpMaximize)

    out = {(i, t): pulp.LpVariable(f"out_{i}_{t}", cat="Binary")
           for i in on_pitch for t in minutes}
    inn = {(j, t): pulp.LpVariable(f"in_{j}_{t}", cat="Binary")
           for j in bench for t in minutes}
    swap = {
        (i, j, t): pulp.LpVariable(f"s_{i}_{j}_{t}", cat="Binary")
        for (i, j, t) in gain
        if compatible.get((i, j), True)
    }

    m += pulp.lpSum(gain[k] * swap[k] for k in swap)

    # at most `subs_remaining` substitutions
    m += pulp.lpSum(out[k] for k in out) <= subs_remaining
    # each on-pitch player off at most once; each bench player on at most once
    for i in on_pitch:
        m += pulp.lpSum(out[(i, t)] for t in minutes) <= 1
    for j in bench:
        m += pulp.lpSum(inn[(j, t)] for t in minutes) <= 1
    # couple swap to its off/on indicators
    for (i, j, t), s in swap.items():
        m += s <= out[(i, t)]
        m += s <= inn[(j, t)]
        m += s >= out[(i, t)] + inn[(j, t)] - 1
    # balance: an "on" requires a matching "off" at or before t
    for t_idx, t in enumerate(minutes):
        ins = pulp.lpSum(inn[(j, tt)] for j in bench for tt in minutes[:t_idx + 1])
        offs = pulp.lpSum(out[(i, tt)] for i in on_pitch for tt in minutes[:t_idx + 1])
        m += ins == offs, f"balance_{t}"

    m.solve(pulp.PULP_CBC_CMD(msg=False))

    moves = [k for k, s in swap.items() if s.value() and s.value() > 0.5]
    return SubSolution(
        status=pulp.LpStatus[m.status],
        expected_gain=float(pulp.value(m.objective) or 0.0),
        moves=[(int(i), int(j), int(t)) for (i, j, t) in moves],
    )
