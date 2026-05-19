"""Evaluation metrics (Part 7.1).

Targets from the spec:
    EPV calibration ECE <= 0.02; next-goal AUC >= 0.72.
    Formation clustering Adjusted Rand Index >= 0.70.
    Valuation log-MarketValue RMSE <= 0.35; Spearman(PVS, minutes) >= 0.55.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: |accuracy - confidence| averaged over equal-width probability bins."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def adjusted_rand_index(true_labels, pred_labels) -> float:
    """ARI between manual and model formation clusters (Part 7.1)."""
    return float(adjusted_rand_score(true_labels, pred_labels))


def rmse_log_value(pred_eur: np.ndarray, true_eur: np.ndarray) -> float:
    """RMSE on log market value (Part 7.1 valuation target)."""
    return float(np.sqrt(np.mean((np.log(pred_eur) - np.log(true_eur)) ** 2)))
