"""Formation switch detection via the Boolean lattice (Part 5.2).

Formations are elements of the Boolean lattice ``(2^R, subseteq)`` over the set
of roles R. ``F <= F'`` iff ``F subseteq F'``. A transition ``F_t -> F_{t+d}``
is a Hasse-diagram edge iff exactly one role differs. We fit a discrete Markov
chain over observed formation clusters and return its stationary distribution
(a tactical-flexibility fingerprint).
"""

from __future__ import annotations

import numpy as np


def formation_set(roles: list[str]) -> frozenset[str]:
    """A formation as an immutable subset of roles (a lattice element)."""
    return frozenset(roles)


def is_subformation(f: frozenset[str], g: frozenset[str]) -> bool:
    """``f <= g`` in the lattice iff f is a subset of g (fewer commitments)."""
    return f <= g


def is_hasse_edge(f: frozenset[str], g: frozenset[str]) -> bool:
    """True iff f -> g is a covering relation: symmetric difference of size 1."""
    return len(f ^ g) == 1


def transition_markov_chain(sequence: list[int], n_states: int) -> np.ndarray:
    """Row-stochastic transition matrix from an observed cluster-label sequence.

    ``sequence`` are formation-cluster ids (e.g. from Part 1.2 k-means) over
    consecutive phases/matches; returns the MLE transition matrix.
    """
    T = np.zeros((n_states, n_states))
    for a, b in zip(sequence[:-1], sequence[1:]):
        T[a, b] += 1.0
    row = T.sum(axis=1, keepdims=True)
    return np.where(row > 0, T / row, 1.0 / n_states)


def stationary_distribution(T: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Stationary distribution pi = pi T via power iteration."""
    n = T.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(10_000):
        new = pi @ T
        if np.abs(new - pi).sum() < tol:
            pi = new
            break
        pi = new
    return pi / pi.sum()
