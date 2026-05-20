"""Role discovery via nonnegative matrix factorization (v3 Part B.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from fas.entities import PlayerSeason


@dataclass(slots=True)
class RoleModel:
    """Latent role components and player memberships."""

    memberships: pd.DataFrame
    components: pd.DataFrame
    reconstruction_error: float
    method: str = "NMF role discovery"
    math: str = "nonnegative matrix factorization / convex role memberships"


def player_action_matrix(
    actions: pd.DataFrame,
    *,
    player_col: str = "player_id",
    action_col: str = "action_type",
    normalize: bool = True,
) -> pd.DataFrame:
    """Build player by action-type frequency matrix."""
    X = pd.crosstab(actions[player_col], actions[action_col]).astype(float)
    if normalize:
        row = X.sum(axis=1).replace(0.0, 1.0)
        X = X.div(row, axis=0)
    return X


def fit_roles_nmf(
    actions_or_matrix: pd.DataFrame,
    *,
    n_roles: int = 4,
    random_state: int = 0,
    max_iter: int = 1000,
) -> RoleModel:
    """Decompose player action profiles into soft role memberships."""
    if {"player_id", "action_type"}.issubset(actions_or_matrix.columns):
        X = player_action_matrix(actions_or_matrix)
    else:
        X = actions_or_matrix.astype(float)
    n_roles = max(1, min(n_roles, min(X.shape)))
    model = NMF(n_components=n_roles, init="nndsvda", random_state=random_state, max_iter=max_iter)
    W = model.fit_transform(np.maximum(X.to_numpy(dtype=float), 0.0))
    H = model.components_
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    role_names = [f"role_{i}" for i in range(n_roles)]
    return RoleModel(
        memberships=pd.DataFrame(W, index=X.index, columns=role_names),
        components=pd.DataFrame(H, index=role_names, columns=X.columns),
        reconstruction_error=float(model.reconstruction_err_),
    )


def eligible_roles(model: RoleModel, *, threshold: float = 0.2) -> dict[int, set[str]]:
    """Convert soft memberships into MILP-compatible role eligibility sets."""
    out: dict[int, set[str]] = {}
    for pid, row in model.memberships.iterrows():
        roles = set(row[row >= threshold].index)
        if not roles:
            roles = {str(row.idxmax())}
        out[int(pid)] = roles
    return out


def enrich(player: PlayerSeason, model: RoleModel) -> PlayerSeason:
    """Attach role membership vector to a player entity."""
    if player.player_uid not in model.memberships.index:
        return player
    membership = model.memberships.loc[player.player_uid]
    perf = dict(player.performance)
    perf["role_membership"] = {"values": membership.to_dict(), "math": model.math}
    return player.with_updates(role_membership=membership, performance=perf)
