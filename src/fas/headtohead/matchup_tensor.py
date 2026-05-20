"""Style-matchup tensor factorization (v3 Part D.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fas.entities import Matchup


@dataclass(slots=True)
class TensorFactorization:
    """CANDECOMP/PARAFAC factors for a matchup tensor."""

    factors: tuple[np.ndarray, np.ndarray, np.ndarray]
    weights: np.ndarray
    reconstruction_error: float
    method: str = "CP tensor factorization"
    math: str = "CANDECOMP/PARAFAC latent interaction factors"

    def reconstruct(self) -> np.ndarray:
        A, B, C = self.factors
        out = np.zeros((A.shape[0], B.shape[0], C.shape[0]))
        for r, w in enumerate(self.weights):
            out += w * np.einsum("i,j,k->ijk", A[:, r], B[:, r], C[:, r])
        return out

    def predict(self, i: int, j: int, c: int) -> float:
        return float(self.reconstruct()[i, j, c])


def cp_factorize(
    tensor: np.ndarray,
    *,
    rank: int = 2,
    n_iter: int = 200,
    ridge: float = 1e-6,
    random_state: int = 0,
) -> TensorFactorization:
    """Factor a style or duel matchup tensor with ALS."""
    X = np.asarray(tensor, dtype=float)
    rng = np.random.default_rng(random_state)
    I, J, K = X.shape
    rank = max(1, min(rank, I, J, K))
    A = rng.normal(size=(I, rank))
    B = rng.normal(size=(J, rank))
    C = rng.normal(size=(K, rank))
    for _ in range(n_iter):
        A = _als_update(_unfold(X, 0), C, B, ridge)
        B = _als_update(_unfold(X, 1), C, A, ridge)
        C = _als_update(_unfold(X, 2), B, A, ridge)
        A, B, C, weights = _normalize(A, B, C)
    factors = TensorFactorization((A, B, C), weights, 0.0)
    recon = factors.reconstruct()
    factors.reconstruction_error = float(np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-12))
    return factors


def latent_matchup_edges(factors: TensorFactorization, *, component: int = 0) -> np.ndarray:
    """Expose pairwise style edges for one latent component."""
    A, B, _ = factors.factors
    return factors.weights[component] * np.outer(A[:, component], B[:, component])


def enrich(matchup: Matchup, factors: TensorFactorization) -> Matchup:
    """Attach tensor-factor metadata to a matchup entity."""
    payload = {
        "rank": int(len(factors.weights)),
        "reconstruction_error": factors.reconstruction_error,
        "math": factors.math,
    }
    return matchup.with_updates(tensor_factors=payload)


def _khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.column_stack([np.kron(A[:, r], B[:, r]) for r in range(A.shape[1])])


def _unfold(X: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))


def _als_update(unfolded: np.ndarray, F1: np.ndarray, F2: np.ndarray, ridge: float) -> np.ndarray:
    KR = _khatri_rao(F1, F2)
    gram = (F1.T @ F1) * (F2.T @ F2) + ridge * np.eye(F1.shape[1])
    return unfolded @ KR @ np.linalg.inv(gram)


def _normalize(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    rank = A.shape[1]
    weights = np.ones(rank)
    mats = [A, B, C]
    for r in range(rank):
        for m, M in enumerate(mats):
            norm = np.linalg.norm(M[:, r])
            if norm > 0:
                M[:, r] /= norm
                weights[r] *= norm
            mats[m] = M
    return mats[0], mats[1], mats[2], weights
