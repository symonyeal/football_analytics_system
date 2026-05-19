"""Tactical pattern recognition via Boolean functions (Part 5.1).

Learns a decision list f: {0,1}^d -> {0,1} (Rivest 1987) predicting a binary
tactical event (e.g. turnover within 3 actions), and exposes the *dual*
function f^d(b) = NOT f(NOT b), which encodes the complementary (escape)
conditions for coaching staff.

A lightweight greedy decision-list learner is provided so the module runs
without ``wittgenstein``/RIPPER installed; swap in RIPPER for production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Rule:
    """A conjunction of (feature_index, required_value) literals -> label."""

    literals: tuple[tuple[int, int], ...]
    label: int

    def covers(self, b: np.ndarray) -> bool:
        return all(b[i] == v for i, v in self.literals)


@dataclass(slots=True)
class DecisionList:
    """Ordered list of rules with a default label (fires first match)."""

    rules: list[Rule]
    default: int

    def predict(self, b: np.ndarray) -> int:
        for rule in self.rules:
            if rule.covers(b):
                return rule.label
        return self.default


def learn_decision_list(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_rules: int = 12,
    min_coverage: int = 5,
) -> DecisionList:
    """Greedy single-literal decision-list learner (RIPPER-style, simplified).

    At each step pick the literal (feature == value) maximizing class purity on
    the remaining examples, emit it as a rule, and remove the examples it
    covers. The default is the majority class of whatever remains.
    """
    n, d = X.shape
    remaining = np.ones(n, dtype=bool)
    rules: list[Rule] = []

    for _ in range(max_rules):
        if remaining.sum() < min_coverage:
            break
        best = _best_literal(X, y, remaining, min_coverage)
        if best is None:
            break
        (feat, val, label) = best
        mask = remaining & (X[:, feat] == val)
        rules.append(Rule(((feat, val),), label))
        remaining &= ~mask

    default = int(round(y[remaining].mean())) if remaining.any() else int(round(y.mean()))
    return DecisionList(rules=rules, default=default)


def _best_literal(X, y, remaining, min_coverage):
    n, d = X.shape
    best = None
    best_score = -1.0
    for feat in range(d):
        for val in (0, 1):
            mask = remaining & (X[:, feat] == val)
            cov = mask.sum()
            if cov < min_coverage:
                continue
            pos = y[mask].mean()
            purity = max(pos, 1 - pos)
            score = purity * np.log1p(cov)  # purity weighted by coverage
            if score > best_score:
                best_score = score
                best = (feat, val, int(pos >= 0.5))
    return best


def dual_function(f, d: int):
    """Return the dual Boolean function f^d(b) = NOT f(NOT b) (Part 5.1).

    ``f`` is any callable ``{0,1}^d -> {0,1}`` (e.g. ``DecisionList.predict``).
    If f encodes "pressing trap triggered", f^d encodes the opponent's escape
    conditions. Pure construction — no retraining needed.
    """
    def f_dual(b: np.ndarray) -> int:
        return 1 - int(f(1 - np.asarray(b)))

    return f_dual
