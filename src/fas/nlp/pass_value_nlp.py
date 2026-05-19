"""Pass value optimization: reward-risk decomposition (Part 3.2).

For a carrier at (x0, y0) considering a pass to (xT, yT):

    R(xT,yT) = EPV(ball at target) - EPV(s0)              (reward if completed)
    K(xT,yT) = P(intercept) * (EPV(s0) - EPV(turnover))   (risk if intercepted)
    V(xT,yT) = R - K

Optimal target solves
    max_{(xT,yT) in Omega} V    s.t. ||target-origin|| <= d_max,
                                     P(complete) >= p_min,
                                     target not in blocked ellipsoids.

We optimize with SLSQP (scipy) from a grid of restarts (a practical,
dependency-light stand-in for the SQP + CMA-ES scheme in the spec). The EPV,
completion- and interception-probability models are injected as callables, so
this works against the xT surface today and the U-Net EPV later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from fas.data.schema import PITCH_LENGTH, PITCH_WIDTH

EPVFn = Callable[[float, float], float]
ProbFn = Callable[[float, float, float, float], float]


@dataclass(slots=True)
class Defender:
    x: float
    y: float
    rx: float = 4.0   # ellipse semi-axis along x
    ry: float = 3.0   # ellipse semi-axis along y

    def inside(self, x: float, y: float) -> float:
        """Negative when (x,y) is inside the ellipse (constraint g>=0 => outside)."""
        return ((x - self.x) / self.rx) ** 2 + ((y - self.y) / self.ry) ** 2 - 1.0


@dataclass(slots=True)
class PassOptResult:
    target: tuple[float, float]
    value: float
    reward: float
    risk: float


def optimize_pass(
    origin: tuple[float, float],
    epv: EPVFn,
    p_complete: ProbFn,
    p_intercept: ProbFn,
    epv_turnover: EPVFn,
    *,
    defenders: list[Defender] | None = None,
    d_max: float = 40.0,
    p_min: float = 0.4,
    n_restarts: int = 25,
) -> PassOptResult:
    """Find the best legal pass target by multi-start SLSQP (Part 3.2)."""
    x0, y0 = origin
    epv0 = epv(x0, y0)
    defenders = defenders or []

    def neg_value(t):
        xT, yT = t
        reward = epv(xT, yT) - epv0
        risk = p_intercept(x0, y0, xT, yT) * (epv0 - epv_turnover(xT, yT))
        return -(reward - risk)

    cons = [
        {"type": "ineq", "fun": lambda t: d_max ** 2 - ((t[0]-x0)**2 + (t[1]-y0)**2)},
        {"type": "ineq", "fun": lambda t: p_complete(x0, y0, t[0], t[1]) - p_min},
    ]
    for dfn in defenders:
        cons.append({"type": "ineq", "fun": (lambda t, d=dfn: d.inside(t[0], t[1]))})

    bounds = [(0, PITCH_LENGTH), (0, PITCH_WIDTH)]
    best: PassOptResult | None = None
    rng = np.random.default_rng(0)
    for _ in range(n_restarts):
        guess = (
            np.clip(x0 + rng.uniform(-d_max, d_max), 0, PITCH_LENGTH),
            np.clip(y0 + rng.uniform(-d_max, d_max), 0, PITCH_WIDTH),
        )
        res = minimize(neg_value, guess, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            continue
        xT, yT = res.x
        reward = epv(xT, yT) - epv0
        risk = p_intercept(x0, y0, xT, yT) * (epv0 - epv_turnover(xT, yT))
        val = reward - risk
        if best is None or val > best.value:
            best = PassOptResult((float(xT), float(yT)), float(val), float(reward), float(risk))
    if best is None:
        return PassOptResult(origin, 0.0, 0.0, 0.0)
    return best
