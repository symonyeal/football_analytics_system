"""Tests for the Boolean & dual functions module (Part 5)."""

import numpy as np

from fas.boolean import (
    dual_function,
    is_hasse_edge,
    is_subformation,
    learn_decision_list,
    stationary_distribution,
    transition_markov_chain,
)
from fas.boolean.formation_lattice import formation_set


def test_decision_list_learns_simple_rule():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(400, 5))
    y = (X[:, 0] & X[:, 1]).astype(int)  # AND of first two features
    dl = learn_decision_list(X, y)
    acc = np.mean([dl.predict(x) == yi for x, yi in zip(X, y)])
    assert acc > 0.85


def test_dual_is_involution():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(200, 4))
    y = (X[:, 0] | X[:, 2]).astype(int)
    dl = learn_decision_list(X, y)
    f = dl.predict
    fdd = dual_function(dual_function(f, 4), 4)
    assert all(fdd(x) == f(x) for x in X[:50])


def test_lattice_relations():
    f = formation_set(["GK", "CB", "CM"])
    g = formation_set(["GK", "CB", "CM", "ST"])
    assert is_subformation(f, g)
    assert is_hasse_edge(f, g)  # differ by exactly one role


def test_markov_stationary_sums_to_one():
    T = transition_markov_chain([0, 1, 0, 1, 2, 1, 0], n_states=3)
    pi = stationary_distribution(T)
    assert abs(pi.sum() - 1.0) < 1e-6
