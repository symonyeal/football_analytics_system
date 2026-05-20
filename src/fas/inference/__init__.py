"""Higher-order inference and insight extraction for v3."""

from fas.inference.causality import granger_causality, influence_network, transfer_entropy_discrete
from fas.inference.insight_engine import Insight, benjamini_hochberg, render_insight, scan_departures
from fas.inference.kernel_mmd import mmd_permutation_test, mmd2_unbiased, rbf_kernel
from fas.inference.ot_style import sinkhorn_barycenter, sinkhorn_distance
from fas.inference.rmt_clean import clean_covariance, marchenko_pastur_bounds
from fas.inference.tda_shape import PersistenceSummary, persistence_features, persistence_image

__all__ = [
    "granger_causality",
    "influence_network",
    "transfer_entropy_discrete",
    "Insight",
    "benjamini_hochberg",
    "render_insight",
    "scan_departures",
    "mmd_permutation_test",
    "mmd2_unbiased",
    "rbf_kernel",
    "sinkhorn_barycenter",
    "sinkhorn_distance",
    "clean_covariance",
    "marchenko_pastur_bounds",
    "PersistenceSummary",
    "persistence_features",
    "persistence_image",
]
