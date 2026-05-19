"""Offline synthetic end-to-end pipeline (Part 7.2, no-network variant).

Generates a small synthetic action stream, then runs the *real* core modules:
pass-network centrality -> xT surface -> zone-flow build-up -> PVS valuation ->
squad-selection MILP. Used by ``fas demo`` and by the integration test so the
whole stack is exercised without downloading StatsBomb data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fas.data.schema import ACTION_TYPES, PITCH_LENGTH, PITCH_WIDTH, validate_actions
from fas.graph import build_pass_network, centrality_table, network_entropy
from fas.milp import FORMATIONS
from fas.milp.player_valuation import (
    low_rank_embedding,
    player_value_scores,
    robust_pca,
)
from fas.milp.squad_selection import ROLES, SquadProblem, cosine_compat, solve_squad
from fas.network_flow import build_zone_graph, buildup_potency, fit_xt


def synthetic_actions(n_players: int = 14, n_actions: int = 1200, seed: int = 0) -> pd.DataFrame:
    """A plausible single-team action stream on the standardized pitch."""
    rng = np.random.default_rng(seed)
    players = list(range(1, n_players + 1))
    rows = []
    t = 0
    for _ in range(n_actions):
        t += int(rng.integers(500, 3000))
        typ = rng.choice(["pass", "pass", "pass", "carry", "shot"], p=[0.4, 0.2, 0.15, 0.2, 0.05])
        x0 = float(rng.uniform(0, PITCH_LENGTH))
        y0 = float(rng.uniform(0, PITCH_WIDTH))
        dx = float(np.clip(x0 + rng.normal(8, 12), 0, PITCH_LENGTH))
        dy = float(np.clip(y0 + rng.normal(0, 12), 0, PITCH_WIDTH))
        rows.append({
            "match_id": 1, "period": 1 if t < 2_700_000 else 2,
            "timestamp_ms": t % 2_700_000,
            "player_id": int(rng.choice(players)), "team_id": 100,
            "action_type": typ if typ in ACTION_TYPES else "pass",
            "x_start": x0, "y_start": y0,
            "x_end": dx, "y_end": dy,
            "outcome": bool(rng.random() > 0.25),
        })
    return validate_actions(pd.DataFrame(rows))


def run_demo() -> dict:
    """Run the synthetic pipeline and print a summary; returns key artifacts."""
    actions = synthetic_actions()

    # 1. Graph theory.
    net = build_pass_network(actions, team_id=100)
    cents = centrality_table(net)
    H = network_entropy(net)

    # 2. Network flow.
    xt = fit_xt(actions)
    zg = build_zone_graph(actions, team_id=100, xt_model=xt)
    flow = buildup_potency(zg)

    # 3. Valuation: build an 18-feature matrix, rPCA -> PVS.
    rng = np.random.default_rng(1)
    players = net.players
    F = np.abs(rng.normal(size=(len(players), 18)))
    # inject the three graph features from Part 1 into the matrix
    F[:, -3] = cents["betweenness"].to_numpy()
    F[:, -2] = cents["pagerank"].to_numpy()
    F[:, -1] = cents["closeness"].to_numpy()
    L, _ = robust_pca(F)
    z = low_rank_embedding(L)
    positions = {p: ROLES[i % len(ROLES)] for i, p in enumerate(players)}
    pvs = player_value_scores(z, players, positions)

    # 4. Squad MILP on an expanded synthetic candidate pool.
    pool = list(range(1, 31))
    rng2 = np.random.default_rng(2)
    pvs_pool = pd.Series(rng2.uniform(0.3, 0.95, size=len(pool)), index=pool)
    eligible = {i: {ROLES[i % len(ROLES)], ROLES[(i + 1) % len(ROLES)]} for i in pool}
    feats = pd.DataFrame(rng2.normal(size=(len(pool), 6)), index=pool)
    prob = SquadProblem(
        players=pool, pvs=pvs_pool, eligible_roles=eligible,
        wage=pd.Series(rng2.uniform(0.2, 2.0, len(pool)), index=pool),
        fair_value=pd.Series(rng2.uniform(2, 60, len(pool)), index=pool),
        age=pd.Series(rng2.integers(18, 34, len(pool)), index=pool),
        homegrown=pd.Series(rng2.random(len(pool)) > 0.7, index=pool),
        compat=cosine_compat(feats),
        squad_size=18, min_young=3, network_gamma=0.1,
    )
    sol = solve_squad(prob, time_limit=30)

    print("=== fas synthetic end-to-end pipeline ===")
    print(f"pass-network players: {net.n}, entropy H(G) = {H:.3f}")
    print(f"top centrality player: {cents['pagerank'].idxmax()} "
          f"(PageRank {cents['pagerank'].max():.3f})")
    print(f"build-up max-flow value: {flow.flow_value}, xT reward: {flow.total_value:.3f}")
    print(f"squad MILP status: {sol.status}, objective {sol.objective:.3f}, "
          f"formation {sol.formation}, starters {len(sol.starters)}")
    print(f"available formations: {list(FORMATIONS)}")
    return {"net": net, "centrality": cents, "xt": xt, "flow": flow, "squad": sol}


if __name__ == "__main__":  # pragma: no cover
    run_demo()
