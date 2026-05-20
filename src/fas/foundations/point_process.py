"""Marked spatio-temporal point-process foundations (v3 Part A.1).

A match is treated as a realization of a marked point process on
``T x Omega x M``. The conditional intensity is the primitive object. This
module provides a dependency-light kernel estimator over canonical actions:

    lambda(t, x, m | H_t) ~= sum_k K_t(t - t_k) K_x(x - x_k) 1[m_k = m].

It is intentionally simple and deterministic so it can run on the core install;
neural intensity models can expose the same :class:`IntensitySurface` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.data.schema import PITCH_LENGTH, PITCH_WIDTH
from fas.entities import MatchObject


@dataclass(slots=True)
class IntensitySurface:
    """Kernel estimate of ``lambda(t, x, y, mark | history)``."""

    event_times: np.ndarray
    xy: np.ndarray
    marks: np.ndarray
    mark_values: tuple[str, ...]
    bandwidth_t: float
    bandwidth_xy: float
    t_max: float

    def rate(self, t: float, x: float, y: float, mark: str | None = None) -> float:
        """Evaluate the conditional intensity at seconds ``t`` and coordinate.

        The estimator uses only events with ``t_k <= t`` so it is causal for
        live-match use. Units are events per second per spatial kernel mass.
        """
        if len(self.event_times) == 0:
            return 0.0
        t = float(t)
        dx = self.xy[:, 0] - float(x)
        dy = self.xy[:, 1] - float(y)
        history = self.event_times <= t
        if mark is not None:
            history &= self.marks == mark
        if not history.any():
            return 0.0
        kt = np.exp(-0.5 * ((t - self.event_times[history]) / self.bandwidth_t) ** 2)
        ks = np.exp(-0.5 * ((dx[history] ** 2 + dy[history] ** 2) / self.bandwidth_xy**2))
        norm = 2.0 * np.pi * self.bandwidth_xy**2 * self.bandwidth_t
        return float((kt * ks).sum() / max(norm, 1e-12))

    def tempo(self, t_lo: float, t_hi: float, mark: str | None = None) -> float:
        """Expected event rate in a time window, estimated by event counts."""
        mask = (self.event_times >= t_lo) & (self.event_times < t_hi)
        if mark is not None:
            mask &= self.marks == mark
        duration = max(float(t_hi - t_lo), 1e-12)
        return float(mask.sum() / duration)

    def grid(self, t: float, *, n_x: int = 16, n_y: int = 12, mark: str | None = None) -> np.ndarray:
        """Evaluate the surface on a regular pitch grid."""
        xs = np.linspace(0.0, PITCH_LENGTH, n_x)
        ys = np.linspace(0.0, PITCH_WIDTH, n_y)
        out = np.zeros((n_x, n_y))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                out[i, j] = self.rate(t, x, y, mark=mark)
        return out


def fit_intensity_surface(
    actions: pd.DataFrame,
    *,
    mark_col: str = "action_type",
    bandwidth_t: float = 45.0,
    bandwidth_xy: float = 10.0,
) -> IntensitySurface:
    """Fit the kernel intensity estimator from canonical actions."""
    if len(actions) == 0:
        return IntensitySurface(
            event_times=np.array([], dtype=float),
            xy=np.zeros((0, 2), dtype=float),
            marks=np.array([], dtype=object),
            mark_values=(),
            bandwidth_t=bandwidth_t,
            bandwidth_xy=bandwidth_xy,
            t_max=0.0,
        )
    df = actions.sort_values(["period", "timestamp_ms"]).copy()
    event_times = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    # Periods restart clocks in most feeds; offset them into match seconds.
    if "period" in df:
        event_times += (df["period"].to_numpy(dtype=float) - 1.0) * 45.0 * 60.0
    xy = df[["x_start", "y_start"]].to_numpy(dtype=float)
    marks = df[mark_col].astype(str).to_numpy()
    return IntensitySurface(
        event_times=event_times,
        xy=xy,
        marks=marks,
        mark_values=tuple(sorted(set(marks))),
        bandwidth_t=float(bandwidth_t),
        bandwidth_xy=float(bandwidth_xy),
        t_max=float(event_times.max(initial=0.0)),
    )


def enrich(match: MatchObject, *, bandwidth_t: float = 45.0, bandwidth_xy: float = 10.0) -> MatchObject:
    """Attach an :class:`IntensitySurface` to a :class:`MatchObject`."""
    surface = fit_intensity_surface(
        match.actions,
        bandwidth_t=bandwidth_t,
        bandwidth_xy=bandwidth_xy,
    )
    return match.with_updates(intensity_surface=surface)
