"""Head-to-head and matchup models for v3."""

from fas.headtohead.copula_outcomes import GaussianCopula, fit_gaussian_copula
from fas.headtohead.hawkes_momentum import HawkesResult, fit_hawkes
from fas.headtohead.matchup_tensor import TensorFactorization, cp_factorize
from fas.headtohead.network_ranking import (
    colley_ratings,
    competitiveness_spectral_gap,
    massey_ratings,
    pagerank_results,
    results_graph,
)
from fas.headtohead.paired_comparison import PairedComparisonModel, fit_bradley_terry_davidson

__all__ = [
    "GaussianCopula",
    "fit_gaussian_copula",
    "HawkesResult",
    "fit_hawkes",
    "TensorFactorization",
    "cp_factorize",
    "colley_ratings",
    "competitiveness_spectral_gap",
    "massey_ratings",
    "pagerank_results",
    "results_graph",
    "PairedComparisonModel",
    "fit_bradley_terry_davidson",
]
