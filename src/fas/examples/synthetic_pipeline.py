"""Offline end-to-end pipeline (Part 7.2, local-data-first variant).

The demo looks for a canonical actions file under ``data/``. If none is found,
it generates a small synthetic action stream. Either way, the same core modules
run: graph -> xT -> flow -> valuation -> v3 performance/head-to-head/inference
-> squad MILP.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fas.entities import MatchMeta, MatchObject, Matchup, PlayerSeason, TeamSeason
from fas.data.schema import ACTION_TYPES, PITCH_LENGTH, PITCH_WIDTH, validate_actions
from fas.foundations import fit_intensity_surface, functorial_statement
from fas.graph import build_pass_network, centrality_table, network_entropy
from fas.headtohead import (
    colley_ratings,
    cp_factorize,
    fit_bradley_terry_davidson,
    fit_gaussian_copula,
    fit_hawkes,
    massey_ratings,
    results_graph,
)
from fas.headtohead.copula_outcomes import scoreline_distribution
from fas.inference import (
    clean_covariance,
    granger_causality,
    mmd_permutation_test,
    persistence_features,
    render_insight,
    scan_departures,
    sinkhorn_distance,
)
from fas.inference.ot_style import grid_cost
from fas.milp import FORMATIONS
from fas.milp.player_valuation import (
    low_rank_embedding,
    player_value_scores,
    robust_pca,
)
from fas.milp.squad_selection import ROLES, SquadProblem, cosine_compat, solve_squad
from fas.network_flow import build_zone_graph, buildup_potency, fit_xt, zone_of
from fas.network_flow.xt_surface import xt_added
from fas.performance import (
    fit_dixon_coles,
    fit_hierarchical_skill,
    fit_irt_2pl,
    fit_possession_mdp,
    fit_rapm,
    fit_roles_nmf,
    kalman_form,
    pitch_control_surface,
    team_style_distribution,
)
from fas.performance.pitch_control import integrate_control_xt
from fas.performance.style_manifold import fisher_rao_distance


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


def discover_actions_file(root: str | Path = "data") -> Path | None:
    """Find a local canonical actions file, if one has been placed under data/."""
    root = Path(root)
    preferred = [
        root / "processed" / "actions.parquet",
        root / "processed" / "actions.csv",
        root / "processed" / "actions.json",
        root / "raw" / "actions.parquet",
        root / "raw" / "actions.csv",
        root / "raw" / "actions.json",
    ]
    for path in preferred:
        if path.exists() and path.stat().st_size > 0:
            return path
    for pattern in ("*actions*.parquet", "*actions*.csv", "*actions*.json"):
        for path in root.rglob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                return path
    return None


def load_actions(path: str | Path) -> pd.DataFrame:
    """Load a canonical actions file from parquet, CSV, or JSON."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"unsupported actions file type: {path}")
    return validate_actions(df)


def run_demo(
    *,
    data_path: str | Path | None = None,
    write_summary: bool = False,
) -> dict:
    """Run the local-data or synthetic pipeline and return key artifacts."""
    source_path = Path(data_path) if data_path is not None else discover_actions_file()
    if source_path is not None:
        actions = load_actions(source_path)
        data_source = str(source_path)
    else:
        actions = synthetic_actions()
        data_source = "synthetic fallback"

    team_id = int(actions["team_id"].mode().iloc[0])

    # 1. Graph theory.
    net = build_pass_network(actions, team_id=team_id)
    cents = centrality_table(net)
    H = network_entropy(net)

    # 2. Network flow.
    xt = fit_xt(actions)
    zg = build_zone_graph(actions, team_id=team_id, xt_model=xt)
    flow = buildup_potency(zg)
    action_values = _action_values(actions, xt)

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

    # 5. Entity spine and v3 models.
    match = MatchObject(
        match_id=int(actions["match_id"].iloc[0]),
        actions=actions,
        pass_networks={team_id: net},
        centrality={team_id: cents},
        xt_added=xt_added(xt, actions),
        zone_flow={team_id: flow},
        intensity_surface=fit_intensity_surface(actions),
        meta=MatchMeta(competition="demo", season="local", home_team_id=team_id),
    )
    player_entities = _player_entities(players, positions, pvs, action_values, cents, team_id)
    team_entity = TeamSeason(team_id=team_id, league="demo", season="local", squad=players)

    rapm = _demo_rapm(actions, action_values)
    skill = fit_hierarchical_skill(actions, positions=positions, n_samples=50)
    irt = fit_irt_2pl(_irt_responses(actions), max_iter=120)
    form = kalman_form(_form_observations(actions, action_values))
    roles = fit_roles_nmf(actions, n_roles=min(4, max(1, len(players))))
    mdp = fit_possession_mdp(_possession_transitions(actions))
    style_all = team_style_distribution(actions, team_id=team_id)
    style_first, style_second = _split_style(actions, team_id)
    pitch_players = _pitch_players(actions, team_id)
    pitch = pitch_control_surface(pitch_players, team_id=team_id, grid_x=12, grid_y=8)
    pitch_value = integrate_control_xt(pitch, xt)

    opponent_id, third_team_id = _demo_opponents(team_id)
    results = _demo_results(team_id, opponent_id, third_team_id)
    scoring = fit_dixon_coles(results)
    score_dist = scoring.score_distribution(team_id, opponent_id, max_goals=5)
    result_graph = results_graph(results)
    massey = massey_ratings(results)
    colley = colley_ratings(results)
    paired = fit_bradley_terry_davidson(_paired_results(results))
    copula = fit_gaussian_copula(_copula_frame(results))
    copula_scores = scoreline_distribution(copula, n=500, max_goals=5)
    hawkes = fit_hawkes(_event_seconds(actions, action_values), horizon=95.0 * 60.0)
    tensor = cp_factorize(_matchup_tensor(results), rank=2, n_iter=40)
    matchup = Matchup(
        entity_i=team_id,
        entity_j=opponent_id,
        predicted_distribution=score_dist,
        paired_comparison={"probabilities": paired.probabilities(team_id, opponent_id)},
        network_ranking={
            "massey": float(massey.get(team_id, 0.0)),
            "colley": float(colley.get(team_id, 0.0)),
        },
        tensor_factors={"rank": int(len(tensor.weights)), "error": tensor.reconstruction_error},
        copula={"scoreline_rows": int(len(copula_scores))},
        hawkes={"branching_ratio": hawkes.branching_ratio},
    )

    shape = persistence_features(pitch_players[pitch_players["team_id"] == team_id][["x", "y"]].to_numpy())
    cov_clean = clean_covariance(F)
    style_distance = fisher_rao_distance(style_first, style_second)
    ot_distance = sinkhorn_distance(
        style_first.to_numpy(),
        style_second.reindex(style_first.index).fillna(0.0).to_numpy(),
        grid_cost(len(style_first)),
    )
    mmd = _safe_mmd(F)
    causality = _safe_granger(actions, action_values)
    insights = scan_departures(
        pd.DataFrame({
            "player_id": actions["player_id"],
            "metric": action_values.to_numpy(),
            "expected": float(action_values.mean()),
        }),
        entity_col="player_id",
        metric_col="metric",
        expected_col="expected",
        min_n=5,
    )

    print("=== fas synthetic end-to-end pipeline ===")
    print(f"data source: {data_source}")
    print(f"pass-network players: {net.n}, entropy H(G) = {H:.3f}")
    print(f"top centrality player: {cents['pagerank'].idxmax()} "
          f"(PageRank {cents['pagerank'].max():.3f})")
    print(f"build-up max-flow value: {flow.flow_value}, xT reward: {flow.total_value:.3f}")
    print(f"v3 RAPM leader: {rapm.alpha.idxmax()} ({rapm.alpha.max():.3f})")
    print(f"v3 skill dimensions: {list(skill.mean.columns)}")
    print(f"v3 top learned role: {roles.memberships.mean().idxmax()}")
    print(f"v3 possession MDP mean value: {mdp.values.mean():.3f}")
    print(f"v3 pitch-control xT value: {pitch_value:.3f}")
    print(f"v3 score P(home/draw/away): {scoring.outcome_probabilities(team_id, opponent_id, max_goals=5)}")
    print(f"v3 Hawkes branching ratio: {hawkes.branching_ratio:.3f}")
    print(f"v3 style drift Fisher-Rao: {style_distance:.3f}, OT: {ot_distance:.3f}")
    print(f"v3 shape compactness: {shape.features['compactness']:.3f}")
    if insights:
        print(f"v3 insight: {render_insight(insights[0])}")
    else:
        print("v3 insight: no FDR-controlled departures in this demo sample")
    print(f"squad MILP status: {sol.status}, objective {sol.objective:.3f}, "
          f"formation {sol.formation}, starters {len(sol.starters)}")
    print(f"available formations: {list(FORMATIONS)}")

    summary = {
        "data_source": data_source,
        "team_id": team_id,
        "players": int(net.n),
        "network_entropy": float(H),
        "flow_value": int(flow.flow_value),
        "flow_xt_reward": float(flow.total_value),
        "rapm_leader": int(rapm.alpha.idxmax()),
        "rapm_value": float(rapm.alpha.max()),
        "role_leader": str(roles.memberships.mean().idxmax()),
        "mdp_mean_value": float(mdp.values.mean()),
        "pitch_control_xt": float(pitch_value),
        "hawkes_branching_ratio": float(hawkes.branching_ratio),
        "style_fisher_rao": float(style_distance),
        "style_ot": float(ot_distance),
        "shape_compactness": float(shape.features["compactness"]),
        "mmd_p_value": float(mmd["p_value"]),
        "granger_p_value": float(causality["p_value"]),
        "insights": [render_insight(i) for i in insights[:3]],
        "coherence": functorial_statement(),
    }
    if write_summary:
        _write_summary(summary)

    return {
        "actions": actions,
        "match": match,
        "players": player_entities,
        "team": team_entity,
        "matchup": matchup,
        "net": net,
        "centrality": cents,
        "xt": xt,
        "flow": flow,
        "squad": sol,
        "rapm": rapm,
        "skill": skill,
        "irt": irt,
        "form": form,
        "roles": roles,
        "mdp": mdp,
        "scoring": scoring,
        "hawkes": hawkes,
        "summary": summary,
    }


def _action_values(actions: pd.DataFrame, xt_model) -> pd.Series:
    values = np.zeros(len(actions), dtype=float)
    for k, row in enumerate(actions.itertuples()):
        if row.action_type in ("pass", "carry") and row.outcome and not pd.isna(row.x_end):
            values[k] = xt_model.value(row.x_end, row.y_end) - xt_model.value(row.x_start, row.y_start)
        elif row.action_type == "shot":
            values[k] = 0.30 if row.outcome else 0.03
        elif row.action_type in ("pressure", "tackle", "interception") and row.outcome:
            values[k] = 0.02
    return pd.Series(values, index=actions.index, name="action_value")


def _player_entities(
    players: list[int],
    positions: dict[int, str],
    pvs: pd.Series,
    action_values: pd.Series,
    centrality: pd.DataFrame,
    team_id: int,
) -> list[PlayerSeason]:
    out = []
    for pid in players:
        idx = centrality.index
        graph_features = centrality.loc[pid] if pid in idx else pd.Series(dtype=float)
        out.append(PlayerSeason(
            player_uid=int(pid),
            league="demo",
            season="local",
            team_id=team_id,
            position=positions.get(pid),
            minutes=90,
            graph_features=graph_features,
            epv_added_90=float(action_values.mean()),
            pvs=float(pvs.get(pid, 0.0)),
            pvs_distribution=np.array([float(pvs.get(pid, 0.0))]),
        ))
    return out


def _demo_rapm(actions: pd.DataFrame, action_values: pd.Series):
    rows = []
    players = sorted(int(p) for p in actions["player_id"].unique())
    for k, (_, row) in enumerate(actions.iterrows()):
        home = [int(row["player_id"])]
        away = [players[(players.index(int(row["player_id"])) + 1) % len(players)]]
        rows.append({"home_players": home, "away_players": away, "value_diff": float(action_values.iloc[k])})
    return fit_rapm(pd.DataFrame(rows), player_ids=players, lambda_=0.5)


def _irt_responses(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    out["item_id"] = out.apply(
        lambda r: f"{r['action_type']}_{zone_of(r['x_start'], r['y_start'])}",
        axis=1,
    )
    return out[["player_id", "item_id", "outcome"]]


def _form_observations(actions: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    df = actions.assign(value=values.to_numpy())
    df["t"] = (df["period"].astype(int) - 1) * 4 + pd.cut(
        df["timestamp_ms"],
        bins=4,
        labels=False,
        duplicates="drop",
    ).fillna(0).astype(int)
    obs = df.groupby(["player_id", "t"], as_index=False)["value"].mean()
    return obs.rename(columns={"value": "performance"})


def _possession_transitions(actions: pd.DataFrame) -> pd.DataFrame:
    df = actions.sort_values(["period", "timestamp_ms"]).copy()
    df["state"] = df.apply(lambda r: f"{r['action_type']}_{zone_of(r['x_start'], r['y_start'])}", axis=1)
    rows = []
    states = df["state"].to_list()
    for i, row in enumerate(df.itertuples()):
        if row.action_type == "shot" and row.outcome:
            nxt = "goal"
        elif not row.outcome:
            nxt = "turnover"
        elif i + 1 < len(states):
            nxt = states[i + 1]
        else:
            nxt = "turnover"
        rows.append({"state": states[i], "next_state": nxt})
    return pd.DataFrame(rows)


def _split_style(actions: pd.DataFrame, team_id: int):
    ordered = actions.sort_values(["period", "timestamp_ms"])
    mid = len(ordered) // 2
    first = team_style_distribution(ordered.iloc[:mid], team_id=team_id)
    second = team_style_distribution(ordered.iloc[mid:], team_id=team_id)
    idx = first.index.union(second.index)
    return first.reindex(idx).fillna(0.0), second.reindex(idx).fillna(0.0)


def _pitch_players(actions: pd.DataFrame, team_id: int) -> pd.DataFrame:
    last = actions.sort_values(["period", "timestamp_ms"]).groupby("player_id").tail(1)
    players = last[["player_id", "team_id", "x_start", "y_start"]].rename(
        columns={"x_start": "x", "y_start": "y"}
    )
    if players["team_id"].nunique() == 1:
        mirrored = players.copy()
        mirrored["player_id"] = mirrored["player_id"] + 10_000
        mirrored["team_id"] = 200 if team_id != 200 else 201
        mirrored["x"] = PITCH_LENGTH - mirrored["x"]
        mirrored["y"] = PITCH_WIDTH - mirrored["y"]
        players = pd.concat([players, mirrored], ignore_index=True)
    return players


def _demo_opponents(team_id: int) -> tuple[int, int]:
    candidates = [200, 300, 400, 500, team_id + 1000, team_id + 2000]
    usable = [c for c in candidates if c != team_id]
    return usable[0], usable[1]


def _demo_results(team_id: int, opponent_id: int, third_team_id: int) -> pd.DataFrame:
    return pd.DataFrame({
        "home_team": [team_id, opponent_id, team_id, third_team_id,
                      opponent_id, third_team_id, team_id, opponent_id],
        "away_team": [opponent_id, team_id, third_team_id, team_id,
                      third_team_id, opponent_id, opponent_id, third_team_id],
        "home_goals": [2, 1, 3, 0, 1, 2, 1, 2],
        "away_goals": [1, 1, 0, 2, 1, 1, 0, 0],
    })


def _paired_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in results.itertuples(index=False):
        if row.home_goals > row.away_goals:
            outcome = "win"
        elif row.home_goals < row.away_goals:
            outcome = "loss"
        else:
            outcome = "tie"
        rows.append({"team_i": row.home_team, "team_j": row.away_team, "outcome": outcome, "home": 1.0})
    return pd.DataFrame(rows)


def _copula_frame(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "home_xg": results["home_goals"] + 0.25,
        "away_xg": results["away_goals"] + 0.25,
        "possession": np.linspace(0.45, 0.60, len(results)),
        "momentum": np.linspace(0.2, 0.8, len(results)),
    })


def _event_seconds(actions: pd.DataFrame, values: pd.Series) -> np.ndarray:
    df = actions.assign(value=values.to_numpy())
    mask = (df["action_type"] == "shot") | (df["value"] >= df["value"].quantile(0.9))
    seconds = df.loc[mask, "timestamp_ms"].to_numpy(dtype=float) / 1000.0
    seconds += (df.loc[mask, "period"].to_numpy(dtype=float) - 1.0) * 45.0 * 60.0
    return seconds


def _matchup_tensor(results: pd.DataFrame) -> np.ndarray:
    teams = sorted(set(results["home_team"]) | set(results["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    tensor = np.zeros((len(teams), len(teams), 2))
    for row in results.itertuples(index=False):
        i, j = idx[row.home_team], idx[row.away_team]
        margin = float(row.home_goals - row.away_goals)
        tensor[i, j, 0] += margin
        tensor[j, i, 1] -= margin
    return tensor + 0.1


def _safe_mmd(F: np.ndarray) -> dict[str, float]:
    if len(F) < 4:
        return {"mmd2": 0.0, "p_value": 1.0}
    mid = len(F) // 2
    return mmd_permutation_test(F[:mid], F[mid:], n_perm=30)


def _safe_granger(actions: pd.DataFrame, values: pd.Series) -> dict[str, float]:
    players = sorted(actions["player_id"].unique())
    if len(players) < 2:
        return {"f_stat": 0.0, "p_value": 1.0}
    df = actions.assign(value=values.to_numpy())
    df["bin"] = np.arange(len(df)) // max(len(df) // 40, 1)
    pivot = df.pivot_table(index="bin", columns="player_id", values="value", aggfunc="sum").fillna(0.0)
    if len(pivot) < 6:
        return {"f_stat": 0.0, "p_value": 1.0}
    return granger_causality(pivot[players[0]].to_numpy(), pivot[players[1]].to_numpy(), lag=2)


def _write_summary(summary: dict) -> None:
    path = Path("data") / "processed" / "demo_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    run_demo()
