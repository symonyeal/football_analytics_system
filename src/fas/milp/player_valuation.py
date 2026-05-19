"""Player Value Score pipeline (Part 4.1) and cross-league adjustment.

Pipeline:
    1. per-90 feature vector f_i                       (caller supplies)
    2. Bradley-Terry league strength -> adjust f_i      (this module)
    3. Robust PCA  F = L + S  via ADMM -> embedding z_i  (this module)
    4. positional percentile PVS_i = Phi((z_i-mu_P)/sig_P)
    5. fair-value regression on log(MarketValue)

Implemented with numpy + scikit-learn only (no heavy deps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# --- Step 2: Bradley-Terry league strength --------------------------------

def bradley_terry(
    matches: pd.DataFrame,
    *,
    team_col_i: str = "team_i",
    team_col_j: str = "team_j",
    outcome_col: str = "i_won",
    ridge: float = 1e-3,
    iters: int = 500,
    lr: float = 0.1,
) -> dict[int, float]:
    """Fit Bradley-Terry strengths ``beta`` by penalized MLE (Part 4.1 Step 2).

        P(i beats j) = sigmoid(beta_i - beta_j)

    Returns ``{team_id: beta}``. Uses gradient ascent on the ridge-penalized
    log-likelihood (stable, no design-matrix blow-up for many teams). Draws
    (``outcome`` in (0,1)) are handled as fractional wins.
    """
    teams = sorted(set(matches[team_col_i]) | set(matches[team_col_j]))
    idx = {t: k for k, t in enumerate(teams)}
    beta = np.zeros(len(teams))

    ti = matches[team_col_i].map(idx).to_numpy()
    tj = matches[team_col_j].map(idx).to_numpy()
    y = matches[outcome_col].to_numpy(dtype=float)

    for _ in range(iters):
        diff = beta[ti] - beta[tj]
        p = 1.0 / (1.0 + np.exp(-diff))
        resid = y - p
        grad = np.zeros_like(beta)
        np.add.at(grad, ti, resid)
        np.add.at(grad, tj, -resid)
        grad -= ridge * beta
        beta += lr * grad / len(matches)
    beta -= beta.mean()  # identifiability: center
    return {t: float(beta[idx[t]]) for t in teams}


def league_strength(
    beta: dict[int, float],
    team_league: dict[int, str],
) -> dict[str, float]:
    """League factor lambda_L = mean_{i in L} exp(beta_i) (Part 4.1 Step 2)."""
    rows: dict[str, list[float]] = {}
    for team, b in beta.items():
        rows.setdefault(team_league.get(team, "?"), []).append(np.exp(b))
    return {lg: float(np.mean(v)) for lg, v in rows.items()}


def adjust_features(
    features: pd.DataFrame,
    player_league: dict[int, str],
    lam: dict[str, float],
    *,
    reference_league: str,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """Scale per-90 features by ``(lambda_L / lambda_ref) ** alpha`` (Step 2)."""
    lam_ref = lam[reference_league]
    factor = features.index.to_series().map(
        lambda pid: (lam.get(player_league.get(pid, reference_league), lam_ref) / lam_ref)
        ** alpha
    )
    return features.mul(factor, axis=0)


# --- Step 3: Robust PCA via ADMM ------------------------------------------

def robust_pca(
    F: np.ndarray,
    *,
    lam: float | None = None,
    mu: float | None = None,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust PCA decomposition ``F = L + S`` via ADMM (Part 4.1 Step 3).

        min_{L,S}  ||L||_* + lam ||S||_1   s.t.  L + S = F

    Principal Component Pursuit (Candes et al. 2011). Returns ``(L, S)``: the
    low-rank signal and sparse outlier matrices. ``lam`` defaults to
    ``1/sqrt(max(m,n))``.
    """
    m, n = F.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if mu is None:
        mu = m * n / (4.0 * np.abs(F).sum() + 1e-12)
    mu_inv = 1.0 / mu

    L = np.zeros_like(F)
    S = np.zeros_like(F)
    Y = np.zeros_like(F)
    norm_F = np.linalg.norm(F, "fro")

    for _ in range(max_iter):
        L = _svd_threshold(F - S + mu_inv * Y, mu_inv)
        S = _soft_threshold(F - L + mu_inv * Y, lam * mu_inv)
        Z = F - L - S
        Y = Y + mu * Z
        if np.linalg.norm(Z, "fro") / (norm_F + 1e-12) < tol:
            break
    return L, S


def _svd_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return (U * s) @ Vt


def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def low_rank_embedding(L: np.ndarray, *, kaiser: bool = True, k: int | None = None) -> np.ndarray:
    """Project the low-rank matrix onto its top components (Kaiser: eig>1)."""
    Lc = L - L.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Lc, full_matrices=False)
    eig = (s ** 2) / max(L.shape[0] - 1, 1)
    if k is None:
        k = int((eig > 1).sum()) if kaiser else len(s)
        k = max(k, 1)
    return U[:, :k] * s[:k]


# --- Step 4: positional PVS -----------------------------------------------

def player_value_scores(
    z: np.ndarray,
    player_ids: list[int],
    positions: dict[int, str],
) -> pd.Series:
    """PVS_i = Phi((score_i - mu_P)/sig_P) within position peer group (Step 4).

    The scalar player score is the L2 norm of the embedding (overall signal
    magnitude); percentiles are computed within each position group P via the
    standard-normal CDF, yielding ``PVS in (0,1)``.
    """
    raw = np.linalg.norm(z, axis=1)
    df = pd.DataFrame({"player_id": player_ids, "raw": raw})
    df["position"] = df["player_id"].map(positions).fillna("UNK")
    out = np.zeros(len(df))
    for _, grp in df.groupby("position"):
        mu, sig = grp["raw"].mean(), grp["raw"].std(ddof=0) or 1.0
        out[grp.index] = norm.cdf((grp["raw"] - mu) / sig)
    return pd.Series(out, index=df["player_id"], name="PVS")


# --- Step 5: fair-value regression ----------------------------------------

def fair_value_regression(
    pvs: pd.Series,
    age: pd.Series,
    market_value: pd.Series,
    *,
    weights: pd.Series | None = None,
) -> tuple[np.ndarray, "callable"]:
    """WLS fit of log(MarketValue) on PVS, Age, Age^2, PVS*Age (Step 5).

    Returns ``(theta, predict_fn)`` where ``predict_fn(pvs, age) -> EUR``.
    Players whose observed value sits far below prediction are undervalued.
    """
    pid = pvs.index
    X = np.column_stack([
        np.ones(len(pid)),
        pvs.loc[pid].to_numpy(),
        age.loc[pid].to_numpy(),
        age.loc[pid].to_numpy() ** 2,
        (pvs.loc[pid].to_numpy() * age.loc[pid].to_numpy()),
    ])
    y = np.log(market_value.loc[pid].to_numpy())
    w = np.ones(len(pid)) if weights is None else weights.loc[pid].to_numpy()
    W = np.diag(w)
    theta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]

    def predict(p: float, a: float) -> float:
        x = np.array([1.0, p, a, a * a, p * a])
        return float(np.exp(x @ theta))

    return theta, predict
