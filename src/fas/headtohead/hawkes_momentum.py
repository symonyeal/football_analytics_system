"""Momentum and goal clustering as a Hawkes process (v3 Part D.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from fas.entities import MatchObject, Matchup


@dataclass(slots=True)
class HawkesResult:
    """Univariate exponential Hawkes fit."""

    mu: float
    alpha: float
    beta: float
    log_likelihood: float
    method: str = "exponential Hawkes MLE"
    math: str = "self-exciting point process / branching ratio"

    @property
    def branching_ratio(self) -> float:
        return float(self.alpha / self.beta) if self.beta > 0 else float("inf")

    def intensity(self, t: float, events: np.ndarray) -> float:
        hist = events[events < t]
        return float(self.mu + np.sum(self.alpha * np.exp(-self.beta * (t - hist))))


def fit_hawkes(
    event_times: np.ndarray,
    *,
    horizon: float | None = None,
    bounds: tuple[float, float] = (1e-6, 10.0),
) -> HawkesResult:
    """Fit a univariate exponential Hawkes process by maximum likelihood."""
    t = np.sort(np.asarray(event_times, dtype=float))
    if horizon is None:
        horizon = float(t.max(initial=0.0) + 1.0)
    if len(t) == 0:
        return HawkesResult(mu=1e-6, alpha=1e-6, beta=1.0, log_likelihood=0.0)

    def nll(log_par: np.ndarray) -> float:
        mu, alpha, beta = np.exp(log_par)
        ll = 0.0
        excitation = 0.0
        prev = 0.0
        for tk in t:
            excitation *= np.exp(-beta * (tk - prev))
            lam = mu + alpha * excitation
            ll += np.log(lam + 1e-12)
            excitation += 1.0
            prev = tk
        integral = mu * horizon + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (horizon - t)))
        return float(-(ll - integral))

    start = np.log([len(t) / max(horizon, 1.0), 0.1, 1.0])
    log_bounds = [(np.log(bounds[0]), np.log(bounds[1]))] * 3
    res = minimize(nll, start, method="L-BFGS-B", bounds=log_bounds)
    mu, alpha, beta = np.exp(res.x)
    return HawkesResult(float(mu), float(alpha), float(beta), float(-res.fun))


def enrich_match(match: MatchObject, result: HawkesResult) -> MatchObject:
    """Attach Hawkes momentum diagnostics to a match."""
    momentum = dict(match.momentum)
    momentum["hawkes"] = {
        "mu": result.mu,
        "alpha": result.alpha,
        "beta": result.beta,
        "branching_ratio": result.branching_ratio,
        "math": result.math,
    }
    return match.with_updates(momentum=momentum)


def enrich(matchup: Matchup, result: HawkesResult) -> Matchup:
    """Attach Hawkes momentum diagnostics to a matchup."""
    payload = {
        "mu": result.mu,
        "alpha": result.alpha,
        "beta": result.beta,
        "branching_ratio": result.branching_ratio,
        "math": result.math,
    }
    return matchup.with_updates(hawkes=payload)
