"""Dynamic Dixon-Coles style team scoring model (v3 Part C.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from fas.entities import TeamSeason


@dataclass(slots=True)
class TeamScoringModel:
    """Time-weighted Poisson attack/defense model."""

    teams: list[int]
    attack: pd.Series
    defense: pd.Series
    home_advantage: float
    rho: float
    log_likelihood: float
    method: str = "Dixon-Coles weighted Poisson"
    math: str = "Poisson GLM / time-weighted maximum likelihood"

    def expected_goals(self, home_team: int, away_team: int) -> tuple[float, float]:
        ah = float(self.attack.get(home_team, 0.0))
        aa = float(self.attack.get(away_team, 0.0))
        dh = float(self.defense.get(home_team, 0.0))
        da = float(self.defense.get(away_team, 0.0))
        return float(np.exp(ah - da + self.home_advantage)), float(np.exp(aa - dh))

    def score_distribution(self, home_team: int, away_team: int, *, max_goals: int = 8) -> pd.DataFrame:
        lam_h, lam_a = self.expected_goals(home_team, away_team)
        goals = np.arange(max_goals + 1)
        ph = np.exp(goals * np.log(lam_h + 1e-12) - lam_h - gammaln(goals + 1))
        pa = np.exp(goals * np.log(lam_a + 1e-12) - lam_a - gammaln(goals + 1))
        P = np.outer(ph, pa)
        P = _dixon_coles_low_score(P, lam_h, lam_a, self.rho)
        P = P / P.sum()
        rows = []
        for i, hg in enumerate(goals):
            for j, ag in enumerate(goals):
                rows.append({"home_goals": int(hg), "away_goals": int(ag), "prob": float(P[i, j])})
        return pd.DataFrame(rows)

    def outcome_probabilities(self, home_team: int, away_team: int, *, max_goals: int = 8) -> dict[str, float]:
        dist = self.score_distribution(home_team, away_team, max_goals=max_goals)
        h = dist["home_goals"].to_numpy()
        a = dist["away_goals"].to_numpy()
        p = dist["prob"].to_numpy()
        return {
            "home": float(p[h > a].sum()),
            "draw": float(p[h == a].sum()),
            "away": float(p[h < a].sum()),
        }


def fit_dixon_coles(
    matches: pd.DataFrame,
    *,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
    time_col: str | None = None,
    decay: float = 0.0,
    ridge: float = 0.02,
    rho: float = -0.05,
) -> TeamScoringModel:
    """Fit a time-weighted Poisson scoring model with Dixon-Coles correction."""
    teams = sorted(set(matches[home_team_col]) | set(matches[away_team_col]))
    idx = {t: i for i, t in enumerate(teams)}
    h = matches[home_team_col].map(idx).to_numpy()
    a = matches[away_team_col].map(idx).to_numpy()
    x = matches[home_goals_col].to_numpy(dtype=float)
    y = matches[away_goals_col].to_numpy(dtype=float)
    w = _time_weights(matches, time_col, decay)
    n = len(teams)

    def unpack(par: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        attack = par[:n] - par[:n].mean()
        defense = par[n:2 * n] - par[n:2 * n].mean()
        home = par[-1]
        return attack, defense, home

    def objective(par: np.ndarray) -> float:
        attack, defense, home = unpack(par)
        lam_h = np.exp(np.clip(attack[h] - defense[a] + home, -8, 8))
        lam_a = np.exp(np.clip(attack[a] - defense[h], -8, 8))
        ll = w * (
            x * np.log(lam_h + 1e-12) - lam_h - gammaln(x + 1)
            + y * np.log(lam_a + 1e-12) - lam_a - gammaln(y + 1)
        )
        pen = ridge * float(np.sum(attack**2) + np.sum(defense**2) + home * home)
        return float(-ll.sum() + pen)

    start = np.zeros(2 * n + 1)
    res = minimize(objective, start, method="L-BFGS-B")
    attack, defense, home = unpack(res.x)
    return TeamScoringModel(
        teams=teams,
        attack=pd.Series(attack, index=teams, name="attack"),
        defense=pd.Series(defense, index=teams, name="defense"),
        home_advantage=float(home),
        rho=float(rho),
        log_likelihood=float(-objective(res.x)),
    )


def enrich(team: TeamSeason, model: TeamScoringModel) -> TeamSeason:
    """Attach attack/defense strengths to a :class:`TeamSeason`."""
    attack = pd.Series({"current": float(model.attack.get(team.team_id, 0.0))})
    defense = pd.Series({"current": float(model.defense.get(team.team_id, 0.0))})
    perf = dict(team.performance)
    perf["scoring_model"] = {"method": model.method, "math": model.math}
    return team.with_updates(attack_t=attack, defense_t=defense, performance=perf)


def _time_weights(matches: pd.DataFrame, time_col: str | None, decay: float) -> np.ndarray:
    if time_col is None or time_col not in matches or decay <= 0:
        return np.ones(len(matches))
    t = pd.to_numeric(matches[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    age = t.max() - t
    return np.exp(-decay * age)


def _dixon_coles_low_score(P: np.ndarray, lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    out = P.copy()
    if out.shape[0] > 1 and out.shape[1] > 1:
        out[0, 0] *= max(1.0 - lam_h * lam_a * rho, 1e-6)
        out[0, 1] *= max(1.0 + lam_h * rho, 1e-6)
        out[1, 0] *= max(1.0 + lam_a * rho, 1e-6)
        out[1, 1] *= max(1.0 - rho, 1e-6)
    return out
