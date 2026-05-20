"""Joint outcome modeling with Gaussian copulas (v3 Part D.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from fas.entities import Matchup


@dataclass(slots=True)
class GaussianCopula:
    """Gaussian copula with empirical marginals."""

    columns: list[str]
    corr: np.ndarray
    marginals: dict[str, np.ndarray]
    method: str = "Gaussian copula with empirical marginals"
    math: str = "copula theory / Sklar theorem / Monte Carlo"

    def sample(self, n: int, *, random_state: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        z = rng.multivariate_normal(np.zeros(len(self.columns)), self.corr, size=n)
        u = norm.cdf(z)
        out = {}
        for k, col in enumerate(self.columns):
            vals = self.marginals[col]
            q = np.clip(u[:, k], 0.0, 1.0)
            out[col] = np.quantile(vals, q)
        return pd.DataFrame(out)


def fit_gaussian_copula(data: pd.DataFrame, *, columns: list[str] | None = None) -> GaussianCopula:
    """Fit a copula by rank-normalizing each marginal."""
    if columns is None:
        columns = list(data.columns)
    X = data[columns].astype(float)
    Z = np.zeros_like(X.to_numpy(dtype=float))
    for j, col in enumerate(columns):
        ranks = X[col].rank(method="average").to_numpy()
        u = (ranks - 0.5) / len(X)
        Z[:, j] = norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    marginals = {col: X[col].to_numpy(dtype=float) for col in columns}
    return GaussianCopula(columns=columns, corr=corr, marginals=marginals)


def scoreline_distribution(
    copula: GaussianCopula,
    *,
    home_xg_col: str = "home_xg",
    away_xg_col: str = "away_xg",
    n: int = 5000,
    max_goals: int = 8,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample copula xG marginals and convert to scoreline probabilities."""
    rng = np.random.default_rng(random_state)
    draws = copula.sample(n, random_state=random_state)
    lam_h = np.clip(draws[home_xg_col].to_numpy(dtype=float), 1e-6, max_goals)
    lam_a = np.clip(draws[away_xg_col].to_numpy(dtype=float), 1e-6, max_goals)
    hg = np.minimum(rng.poisson(lam_h), max_goals)
    ag = np.minimum(rng.poisson(lam_a), max_goals)
    out = pd.DataFrame({"home_goals": hg, "away_goals": ag}).value_counts(normalize=True)
    return out.rename("prob").reset_index()


def enrich(matchup: Matchup, copula: GaussianCopula, distribution: pd.DataFrame | None = None) -> Matchup:
    """Attach copula metadata and sampled predictive distribution."""
    payload = {"columns": copula.columns, "corr": copula.corr.tolist(), "math": copula.math}
    updates = {"copula": payload}
    if distribution is not None:
        updates["predicted_distribution"] = distribution
    return matchup.with_updates(**updates)
