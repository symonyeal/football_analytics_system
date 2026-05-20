"""Insight extraction with multiplicity control (v3 Part F)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(slots=True)
class Insight:
    """A statistically validated insight."""

    entity_id: int | str
    metric: str
    effect: float
    ci: tuple[float, float]
    p_value: float
    q_value: float
    method: str
    math: str
    context: str = "overall"


def bootstrap_ci(x: np.ndarray, *, n_boot: int = 500, alpha: float = 0.05, random_state: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a mean."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (0.0, 0.0)
    draws = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1.0 - alpha / 2))


def benjamini_hochberg(p_values: np.ndarray, *, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control; returns q-values and reject mask."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q, q <= alpha


def scan_departures(
    frame: pd.DataFrame,
    *,
    entity_col: str,
    metric_col: str,
    expected_col: str | None = None,
    context_col: str | None = None,
    fdr_alpha: float = 0.1,
    min_n: int = 3,
    n_boot: int = 300,
) -> list[Insight]:
    """Scan entity-metric-context cells and surface FDR-controlled departures."""
    rows = []
    group_cols = [entity_col] + ([context_col] if context_col else [])
    for key, grp in frame.groupby(group_cols):
        if len(grp) < min_n:
            continue
        entity = key[0] if isinstance(key, tuple) else key
        context = key[1] if isinstance(key, tuple) and len(key) > 1 else "overall"
        values = grp[metric_col].to_numpy(dtype=float)
        expected = 0.0 if expected_col is None else float(grp[expected_col].mean())
        diff = values - expected
        effect = float(diff.mean())
        _, p = stats.ttest_1samp(diff, 0.0)
        if not np.isfinite(p):
            p = 1.0
        ci = bootstrap_ci(diff, n_boot=n_boot)
        rows.append((entity, context, effect, ci, float(p)))
    if not rows:
        return []
    q, reject = benjamini_hochberg(np.array([r[4] for r in rows]), alpha=fdr_alpha)
    insights = []
    for keep, qv, row in zip(reject, q, rows):
        if not keep:
            continue
        entity, context, effect, ci, p = row
        insights.append(Insight(
            entity_id=entity,
            metric=metric_col,
            effect=effect,
            ci=ci,
            p_value=p,
            q_value=float(qv),
            method="bootstrap CI + one-sample t-test + Benjamini-Hochberg FDR",
            math="multiple testing / FDR / bootstrap",
            context=str(context),
        ))
    return insights


def shapley_values(contributions: dict[str, float]) -> dict[str, float]:
    """Shapley attribution for an additive action-value game."""
    return {k: float(v) for k, v in contributions.items()}


def render_insight(insight: Insight, *, percentile: float | None = None) -> str:
    """Render a validated finding as UI-ready natural language."""
    band = f"[{insight.ci[0]:.3f}, {insight.ci[1]:.3f}]"
    pct = "" if percentile is None else f" ({percentile:.0f}th percentile)"
    return (
        f"{insight.entity_id}'s {insight.metric} is {insight.effect:.3f}{pct} "
        f"in {insight.context} (95% CI {band}, q={insight.q_value:.3f}; "
        f"method: {insight.method})."
    )
