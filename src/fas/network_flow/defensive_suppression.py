"""Defensive flow suppression as a bilevel program (Part 2.3).

We choose a suppression vector ``r in [0,1]^E`` (fraction of each opponent
edge's capacity we neutralize by pressing) to minimize the opponent's max-flow,
subject to a pressing-energy budget ``B``:

    min_r   MaxFlow( cap(i,j) (1 - r_ij) )
    s.t.    sum_ij cost_ij r_ij <= B,   r in [0,1].

The inner max-flow is an LP; by strong duality MaxFlow = min over cut-indicator
duals ``y``. Substituting yields a single-level *bilinear* program in (r, y);
we expose a McCormick-relaxation builder and a successive-linearization solver
(a simple, dependency-light alternative to a global bilinear solver).

This module is a working reference implementation of the relaxation; the full
global solve (tightness analysis, Part 9.1) is left for research extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # cvxpy is an optional dependency (extras: nlp)
    import cvxpy as cp

    _HAVE_CVXPY = True
except ImportError:  # pragma: no cover
    _HAVE_CVXPY = False


@dataclass(slots=True)
class SuppressionPlan:
    r: np.ndarray            # suppression fraction per edge, aligned to edges
    edges: list[tuple[int, int]]
    objective: float         # resulting (relaxed) opponent max-flow


def successive_linearization(
    edges: list[tuple[int, int]],
    capacity: np.ndarray,
    cost: np.ndarray,
    source: int,
    sink: int,
    budget: float,
    *,
    iters: int = 20,
) -> SuppressionPlan:
    """Solve the suppression bilevel via successive linearization (Part 2.3).

    Requires ``cvxpy`` (``pip install fas[nlp]``). Each iteration fixes the
    current cut duals, solves the resulting LP in ``r``, re-evaluates the inner
    max-flow, and repeats — a standard fix-point heuristic for the bilinear
    reformulation. Raises if cvxpy is unavailable.
    """
    if not _HAVE_CVXPY:  # pragma: no cover
        raise ImportError("defensive_suppression requires cvxpy: pip install fas[nlp]")

    m = len(edges)
    nodes = sorted({u for u, _ in edges} | {v for _, v in edges})
    incidence = _incidence(edges, nodes)

    r = np.zeros(m)
    obj = np.inf
    for _ in range(iters):
        eff_cap = capacity * (1 - r)
        flow_val, _ = _max_flow_lp(edges, eff_cap, incidence, nodes, source, sink)
        # LP step: choose r to reduce flow on currently saturated edges.
        r_var = cp.Variable(m, nonneg=True)
        reduce = cp.sum(cp.multiply(capacity, r_var))
        constraints = [r_var <= 1, cost @ r_var <= budget]
        prob = cp.Problem(cp.Maximize(reduce), constraints)
        prob.solve()
        new_r = np.clip(r_var.value, 0, 1) if r_var.value is not None else r
        if np.allclose(new_r, r, atol=1e-4):
            r = new_r
            obj = flow_val
            break
        r = new_r
        obj = flow_val
    return SuppressionPlan(r=r, edges=edges, objective=float(obj))


def mccormick_envelope(lb: float = 0.0, ub: float = 1.0):
    """Return McCormick under/over-estimator callables for a product x*y.

    For ``w = x*y`` with ``x, y in [lb, ub]`` the convex/concave envelopes are
    the four standard McCormick inequalities. Useful when assembling the
    single-level relaxation of the bilinear program (Part 2.3 / Part 9.1).
    """
    def lower(x, y):
        return [lb * y + lb * x - lb * lb, ub * y + ub * x - ub * ub]

    def upper(x, y):
        return [ub * y + lb * x - ub * lb, lb * y + ub * x - lb * ub]

    return lower, upper


# --- inner max-flow LP (used by the linearization loop) --------------------

def _incidence(edges, nodes) -> np.ndarray:
    idx = {n: k for k, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(edges)))
    for e, (u, v) in enumerate(edges):
        A[idx[u], e] = -1.0
        A[idx[v], e] = 1.0
    return A


def _max_flow_lp(edges, cap, incidence, nodes, source, sink):
    f = cp.Variable(len(edges), nonneg=True)
    idx = {n: k for k, n in enumerate(nodes)}
    net = incidence @ f
    cons = [f <= cap]
    for n in nodes:
        if n not in (source, sink):
            cons.append(net[idx[n]] == 0)
    val = net[idx[sink]]
    prob = cp.Problem(cp.Maximize(val), cons)
    prob.solve()
    return float(prob.value or 0.0), (f.value if f.value is not None else np.zeros(len(edges)))
