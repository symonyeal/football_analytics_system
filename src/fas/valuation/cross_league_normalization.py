"""Three-layer cross-league normalization (Part 6.2).

Layer 1  within-league percentile -> inverse-normal z-score per metric.
Layer 2  Bradley-Terry league adjustment, feature-specific shrinkage alpha_k
         (attacking ~0.8, defensive ~0.6).
Layer 3  age-adjusted projection to career peak (see development_curves).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def within_league_percentile(
    features: pd.DataFrame,
    player_league: dict[int, str],
) -> pd.DataFrame:
    """Layer 1: inverse-normal transform of within-league ranks (Part 6.2)."""
    leagues = pd.Series({pid: player_league.get(pid, "?") for pid in features.index})
    out = features.copy().astype(float)
    for lg, idx in leagues.groupby(leagues).groups.items():
        sub = features.loc[idx]
        ranks = sub.rank(axis=0, method="average")
        n = len(sub)
        out.loc[idx] = norm.ppf(ranks / (n + 1)).clip(-4, 4)
    return out


def three_layer_normalize(
    features: pd.DataFrame,
    player_league: dict[int, str],
    league_factor: dict[str, float],
    *,
    reference_league: str,
    alpha: dict[str, float] | float = 0.7,
) -> pd.DataFrame:
    """Apply Layers 1-2 (Layer 3 handled by development_curves).

    ``alpha`` may be a scalar or a per-column dict (feature-specific shrinkage).
    """
    z = within_league_percentile(features, player_league)
    lam_ref = league_factor[reference_league]
    leagues = pd.Series({pid: player_league.get(pid, reference_league) for pid in z.index})

    out = z.copy()
    for col in z.columns:
        a = alpha[col] if isinstance(alpha, dict) else alpha
        factor = leagues.map(
            lambda lg: (league_factor.get(lg, lam_ref) / lam_ref) ** a
        )
        out[col] = z[col] * factor.to_numpy()
    return out
