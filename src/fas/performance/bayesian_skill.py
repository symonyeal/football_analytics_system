"""Bayesian hierarchical skill model (v3 Part B.2).

This core implementation uses empirical-Bayes partial pooling with conjugate
Beta-Binomial observation models, then reports logit-scale posterior summaries
for multidimensional skills. Full NUTS/VI backends can replace this fitter while
preserving the same :class:`SkillPosterior` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from fas.entities import PlayerSeason


DEFAULT_SKILL_MAP = {
    "shot": "finishing",
    "pass": "progression",
    "carry": "progression",
    "dribble": "retention",
    "pressure": "defending",
    "tackle": "defending",
    "interception": "defending",
    "clearance": "defending",
}


@dataclass(slots=True)
class SkillPosterior:
    """Posterior summaries for multidimensional player ability."""

    mean: pd.DataFrame
    variance: pd.DataFrame
    samples: dict[int, pd.DataFrame]
    prior_strength: float
    method: str = "empirical-Bayes partial pooling"
    math: str = "hierarchical Bayes / exchangeability / shrinkage"


def fit_hierarchical_skill(
    actions: pd.DataFrame,
    *,
    positions: dict[int, str] | None = None,
    skill_map: dict[str, str] | None = None,
    prior_strength: float = 12.0,
    n_samples: int = 200,
    random_state: int = 0,
) -> SkillPosterior:
    """Estimate multidimensional player skill posteriors from action outcomes."""
    if skill_map is None:
        skill_map = DEFAULT_SKILL_MAP
    positions = positions or {}
    rng = np.random.default_rng(random_state)
    df = actions[actions["action_type"].isin(skill_map)].copy()
    df["skill"] = df["action_type"].map(skill_map)
    df["position"] = df["player_id"].map(lambda p: positions.get(int(p), "UNK"))
    skills = sorted(df["skill"].unique())
    players = sorted(int(p) for p in df["player_id"].unique())
    mean = pd.DataFrame(0.0, index=players, columns=skills)
    var = pd.DataFrame(0.0, index=players, columns=skills)
    samples: dict[int, pd.DataFrame] = {}

    if len(df) == 0:
        return SkillPosterior(mean, var, samples, prior_strength)

    global_rate = float(df["outcome"].astype(float).mean())
    global_rate = float(np.clip(global_rate, 0.02, 0.98))

    priors: dict[tuple[str, str], tuple[float, float]] = {}
    for (pos, skill), sub in df.groupby(["position", "skill"]):
        n = len(sub)
        rate = float(sub["outcome"].astype(float).mean()) if n else global_rate
        rate = float(np.clip(rate, 0.02, 0.98))
        priors[(pos, skill)] = (rate * prior_strength, (1.0 - rate) * prior_strength)

    for pid in players:
        player_samples: dict[str, np.ndarray] = {}
        pos = positions.get(pid, "UNK")
        for skill in skills:
            sub = df[(df["player_id"] == pid) & (df["skill"] == skill)]
            s = float(sub["outcome"].astype(float).sum())
            n = float(len(sub))
            a0, b0 = priors.get((pos, skill), (global_rate * prior_strength, (1.0 - global_rate) * prior_strength))
            a, b = a0 + s, b0 + n - s
            draws = rng.beta(a, b, size=n_samples)
            clipped = np.clip(draws, 1e-5, 1.0 - 1e-5)
            logit_draws = logit(clipped)
            mean.loc[pid, skill] = float(logit_draws.mean())
            var.loc[pid, skill] = float(logit_draws.var(ddof=1))
            player_samples[skill] = logit_draws
        samples[pid] = pd.DataFrame(player_samples)
    return SkillPosterior(mean=mean, variance=var, samples=samples, prior_strength=prior_strength)


def posterior_probability_better(
    posterior: SkillPosterior,
    player_a: int,
    player_b: int,
    skill: str,
) -> float:
    """Return ``P(theta_a > theta_b | data)`` from posterior draws."""
    a = posterior.samples[int(player_a)][skill].to_numpy()
    b = posterior.samples[int(player_b)][skill].to_numpy()
    n = min(len(a), len(b))
    return float(np.mean(a[:n] > b[:n]))


def enrich(player: PlayerSeason, posterior: SkillPosterior) -> PlayerSeason:
    """Attach skill posterior summaries to a player entity."""
    if player.player_uid not in posterior.mean.index:
        return player
    payload = {
        "mean": posterior.mean.loc[player.player_uid].to_dict(),
        "variance": posterior.variance.loc[player.player_uid].to_dict(),
        "method": posterior.method,
        "math": posterior.math,
    }
    perf = dict(player.performance)
    perf["skill_posterior"] = payload
    return player.with_updates(skill_posterior=payload, performance=perf)


def skill_probability_table(posterior: SkillPosterior) -> pd.DataFrame:
    """Convert logit-scale means to success-probability summaries."""
    return posterior.mean.applymap(lambda x: float(expit(x)))
