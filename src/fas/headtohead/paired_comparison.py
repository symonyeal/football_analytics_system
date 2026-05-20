"""Generalized paired comparison with ties and covariates (v3 Part D.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fas.entities import Matchup


@dataclass(slots=True)
class PairedComparisonModel:
    """Bradley-Terry-Davidson model with home advantage and covariates."""

    teams: list[int]
    beta: pd.Series
    home_advantage: float
    coef: pd.Series
    tie: float
    method: str = "Bradley-Terry-Davidson"
    math: str = "generalized linear paired comparison / penalized MLE"

    def probabilities(self, team_i: int, team_j: int, *, home: float = 0.0, covariates=None) -> dict[str, float]:
        x = np.zeros(len(self.coef)) if covariates is None else np.asarray(covariates, dtype=float)
        eta_i = float(self.beta.get(team_i, 0.0)) + self.home_advantage * home + float(x @ self.coef.to_numpy())
        eta_j = float(self.beta.get(team_j, 0.0))
        ai = np.exp(eta_i)
        aj = np.exp(eta_j)
        tie_mass = np.exp(self.tie) * np.sqrt(ai * aj)
        den = ai + aj + tie_mass
        return {"win": float(ai / den), "tie": float(tie_mass / den), "loss": float(aj / den)}


def fit_bradley_terry_davidson(
    results: pd.DataFrame,
    *,
    team_i_col: str = "team_i",
    team_j_col: str = "team_j",
    outcome_col: str = "outcome",
    home_col: str | None = "home",
    covariate_cols: list[str] | None = None,
    ridge: float = 1e-3,
) -> PairedComparisonModel:
    """Fit penalized Bradley-Terry-Davidson paired comparison."""
    covariate_cols = covariate_cols or []
    teams = sorted(set(results[team_i_col]) | set(results[team_j_col]))
    idx = {t: i for i, t in enumerate(teams)}
    i = results[team_i_col].map(idx).to_numpy()
    j = results[team_j_col].map(idx).to_numpy()
    y = _outcome_codes(results[outcome_col])
    home = np.zeros(len(results)) if home_col is None or home_col not in results else results[home_col].to_numpy(dtype=float)
    X = results[covariate_cols].to_numpy(dtype=float) if covariate_cols else np.zeros((len(results), 0))
    n, p = len(teams), len(covariate_cols)

    def unpack(par: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
        beta = par[:n] - par[:n].mean()
        home_adv = par[n]
        coef = par[n + 1:n + 1 + p]
        tie = par[-1]
        return beta, home_adv, coef, tie

    def objective(par: np.ndarray) -> float:
        beta, home_adv, coef, tie = unpack(par)
        eta_i = beta[i] + home_adv * home + X @ coef
        eta_j = beta[j]
        ai = np.exp(np.clip(eta_i, -20, 20))
        aj = np.exp(np.clip(eta_j, -20, 20))
        tm = np.exp(np.clip(tie, -20, 20)) * np.sqrt(ai * aj)
        den = ai + aj + tm
        probs = np.column_stack([aj / den, tm / den, ai / den])
        # y is 0 loss, 1 tie, 2 win for team_i.
        ll = np.log(probs[np.arange(len(y)), y] + 1e-12).sum()
        pen = ridge * float(np.sum(beta**2) + home_adv**2 + np.sum(coef**2) + tie**2)
        return float(-ll + pen)

    res = minimize(objective, np.zeros(n + 2 + p), method="L-BFGS-B")
    beta, home_adv, coef, tie = unpack(res.x)
    return PairedComparisonModel(
        teams=teams,
        beta=pd.Series(beta, index=teams, name="beta"),
        home_advantage=float(home_adv),
        coef=pd.Series(coef, index=covariate_cols, name="coef"),
        tie=float(tie),
    )


def bootstrap_intervals(
    results: pd.DataFrame,
    *,
    n_boot: int = 100,
    random_state: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for paired-comparison strengths."""
    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(n_boot):
        sample = results.iloc[rng.integers(0, len(results), len(results))]
        draws.append(fit_bradley_terry_davidson(sample, **kwargs).beta)
    B = pd.concat(draws, axis=1).T
    return pd.DataFrame({"lo": B.quantile(0.025), "hi": B.quantile(0.975), "mean": B.mean()})


def enrich(matchup: Matchup, model: PairedComparisonModel, *, home: float = 0.0, covariates=None) -> Matchup:
    """Attach paired-comparison probabilities to a matchup."""
    probs = model.probabilities(matchup.entity_i, matchup.entity_j, home=home, covariates=covariates)
    payload = {"probabilities": probs, "math": model.math}
    return matchup.with_updates(paired_comparison=payload)


def _outcome_codes(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        arr = s.to_numpy(dtype=float)
        return np.where(arr > 0.5, 2, np.where(arr < 0.5, 0, 1))
    mapping = {"loss": 0, "away": 0, "tie": 1, "draw": 1, "win": 2, "home": 2}
    return s.astype(str).str.lower().map(mapping).fillna(1).to_numpy(dtype=int)
