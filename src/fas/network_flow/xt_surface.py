"""Expected Threat (xT) surface via value iteration (Part 2.1).

Karun Singh's xT decomposes the value of having the ball at a grid cell into a
shoot branch and a move branch:

    xT(c) = s(c) * g(c) + m(c) * sum_{c'} T(c -> c') * xT(c')

where for cell ``c``
    s(c) = P(shoot | ball at c)         g(c) = P(goal | shoot from c)
    m(c) = P(move  | ball at c)         T    = move transition matrix.

We estimate s, m, g, T empirically from a canonical action frame and solve by
value iteration to tolerance ``eps`` in the sup-norm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.data.schema import PITCH_LENGTH, PITCH_WIDTH


@dataclass(slots=True)
class XTModel:
    """Fitted xT surface on an ``(n_x, n_y)`` grid.

    ``grid[i, j]`` is the xT value of the cell whose x-band is ``i`` and
    y-band is ``j``. Use :meth:`value` to look up a pitch coordinate.
    """

    grid: np.ndarray
    n_x: int
    n_y: int

    def value(self, x: float, y: float) -> float:
        i, j = self._cell(x, y)
        return float(self.grid[i, j])

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        i = min(int(x / PITCH_LENGTH * self.n_x), self.n_x - 1)
        j = min(int(y / PITCH_WIDTH * self.n_y), self.n_y - 1)
        return max(i, 0), max(j, 0)


def xt_value(model: XTModel, x: float, y: float) -> float:
    """Functional accessor mirroring :meth:`XTModel.value`."""
    return model.value(x, y)


def fit_xt(
    actions: pd.DataFrame,
    *,
    n_x: int = 16,
    n_y: int = 12,
    eps: float = 1e-6,
    max_iter: int = 10_000,
) -> XTModel:
    """Estimate and solve the xT surface from a canonical action frame.

    Shots define the shoot branch; completed passes/carries define the move
    branch and the move-transition matrix. Goals are read from ``shot`` actions
    with ``outcome == True`` (our convention from the StatsBomb loader).
    """
    n_cells = n_x * n_y
    df = actions.copy()
    df["ci"] = (df["x_start"] / PITCH_LENGTH * n_x).clip(0, n_x - 1).astype(int)
    df["cj"] = (df["y_start"] / PITCH_WIDTH * n_y).clip(0, n_y - 1).astype(int)
    df["cell"] = df["ci"] * n_y + df["cj"]

    shots = df["action_type"] == "shot"
    moves = df["action_type"].isin(["pass", "carry"]) & df["outcome"].astype(bool)

    n_shot = _cell_counts(df[shots], n_cells)
    n_goal = _cell_counts(df[shots & df["outcome"].astype(bool)], n_cells)
    n_move = _cell_counts(df[moves], n_cells)
    n_total = n_shot + n_move
    n_total_safe = np.where(n_total > 0, n_total, 1)

    s = n_shot / n_total_safe        # P(shoot | cell)
    m = n_move / n_total_safe        # P(move  | cell)
    g = np.where(n_shot > 0, n_goal / np.where(n_shot > 0, n_shot, 1), 0.0)

    # Move-transition matrix T[c, c'] from completed move end-cells.
    mv = df[moves].dropna(subset=["x_end", "y_end"]).copy()
    mv["ei"] = (mv["x_end"] / PITCH_LENGTH * n_x).clip(0, n_x - 1).astype(int)
    mv["ej"] = (mv["y_end"] / PITCH_WIDTH * n_y).clip(0, n_y - 1).astype(int)
    mv["from"] = mv["ci"] * n_y + mv["cj"]
    mv["to"] = mv["ei"] * n_y + mv["ej"]
    T = np.zeros((n_cells, n_cells))
    for src, dst in zip(mv["from"].to_numpy(), mv["to"].to_numpy()):
        T[src, dst] += 1.0
    row = T.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(row > 0, T / row, 0.0)

    # Value iteration: xT^{k+1} = s*g + m * (T @ xT^k).
    xt = np.zeros(n_cells)
    sg = s * g
    for _ in range(max_iter):
        nxt = sg + m * (T @ xt)
        if np.max(np.abs(nxt - xt)) < eps:
            xt = nxt
            break
        xt = nxt

    return XTModel(grid=xt.reshape(n_x, n_y), n_x=n_x, n_y=n_y)


def _cell_counts(sub: pd.DataFrame, n_cells: int) -> np.ndarray:
    counts = np.zeros(n_cells)
    if len(sub):
        vc = sub["cell"].value_counts()
        counts[vc.index.to_numpy()] = vc.to_numpy()
    return counts


def xt_added(model: XTModel, actions: pd.DataFrame) -> pd.Series:
    """Per-action xT delta = xT(end) - xT(start) for completed moves (Part 2.1)."""
    mv = actions[
        actions["action_type"].isin(["pass", "carry"])
        & actions["outcome"].astype(bool)
    ].dropna(subset=["x_end", "y_end"])
    start = mv.apply(lambda r: model.value(r["x_start"], r["y_start"]), axis=1)
    end = mv.apply(lambda r: model.value(r["x_end"], r["y_end"]), axis=1)
    return (end - start).rename("xt_added")
