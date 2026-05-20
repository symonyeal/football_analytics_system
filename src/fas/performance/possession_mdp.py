"""Possession as an absorbing Markov decision process (v3 Part C.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.entities import TeamSeason


@dataclass(slots=True)
class PossessionMDP:
    """Absorbing-chain possession model."""

    states: list[str]
    absorbing: list[str]
    transition: pd.DataFrame
    fundamental: pd.DataFrame
    values: pd.Series
    method: str = "absorbing Markov chain"
    math: str = "MDP value function / fundamental matrix"


def fit_possession_mdp(
    transitions: pd.DataFrame,
    *,
    state_col: str = "state",
    next_col: str = "next_state",
    absorbing: tuple[str, ...] = ("goal", "turnover", "foul"),
    rewards: dict[str, float] | None = None,
    alpha: float = 1e-3,
) -> PossessionMDP:
    """Estimate transition matrix and closed-form possession values."""
    rewards = rewards or {"goal": 1.0, "turnover": 0.0, "foul": 0.0}
    all_states = sorted(set(transitions[state_col]) | set(transitions[next_col]) | set(absorbing))
    trans_states = [s for s in all_states if s not in absorbing]
    abs_states = [s for s in all_states if s in absorbing]
    ordered = trans_states + abs_states
    counts = pd.DataFrame(alpha, index=ordered, columns=ordered, dtype=float)
    for s, n in zip(transitions[state_col], transitions[next_col]):
        counts.loc[s, n] += 1.0
    P = counts.div(counts.sum(axis=1), axis=0)
    for s in abs_states:
        P.loc[s, :] = 0.0
        P.loc[s, s] = 1.0
    Q = P.loc[trans_states, trans_states].to_numpy(dtype=float)
    R = P.loc[trans_states, abs_states].to_numpy(dtype=float)
    N = np.linalg.inv(np.eye(len(trans_states)) - Q) if trans_states else np.zeros((0, 0))
    reward_vec = np.array([rewards.get(s, 0.0) for s in abs_states], dtype=float)
    v = N @ R @ reward_vec if trans_states else np.array([])
    value = pd.Series(v, index=trans_states, name="mdp_value")
    return PossessionMDP(
        states=ordered,
        absorbing=abs_states,
        transition=P,
        fundamental=pd.DataFrame(N, index=trans_states, columns=trans_states),
        values=value,
    )


def enrich(team: TeamSeason, mdp: PossessionMDP, *, start_distribution: pd.Series | None = None) -> TeamSeason:
    """Attach team possession value to a :class:`TeamSeason`."""
    if start_distribution is None:
        value = float(mdp.values.mean()) if len(mdp.values) else 0.0
    else:
        aligned = start_distribution.reindex(mdp.values.index).fillna(0.0)
        aligned = aligned / max(aligned.sum(), 1e-12)
        value = float(np.dot(aligned.to_numpy(), mdp.values.to_numpy()))
    perf = dict(team.performance)
    perf["possession_mdp"] = {"value": value, "math": mdp.math}
    return team.with_updates(mdp_value=value, performance=perf)
