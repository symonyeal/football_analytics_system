"""Six-layer artifact materializer.

Takes a normalized :class:`~fas.product.ingest.DataSpine` and runs the ``fas``
engine across all six product layers, returning a dictionary of artifact
tables (DataFrames) plus a summary dict. The orchestrator in
:mod:`fas.product.build` writes these to ``data/processed/``.

Every long-format ``*_artifacts`` table carries the full context contract
(match/team/opponent/competition/season/phase/minute window/model/sample
size/data source/synthetic flag/limitations) so the UI can filter and label
without guessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fas.graph import build_pass_network, centrality_table, network_entropy
from fas.headtohead import fit_bradley_terry_davidson
from fas.inference.insight_engine import bootstrap_ci, scan_departures
from fas.milp.player_valuation import low_rank_embedding, player_value_scores, robust_pca
from fas.milp.squad_selection import ROLES, SquadProblem, solve_squad
from fas.network_flow import build_zone_graph, buildup_potency, fit_xt, zone_of
from fas.performance import (
    fit_dixon_coles,
    kalman_form,
    team_style_distribution,
)
from fas.performance.style_manifold import fisher_rao_distance
from fas.product.centralisation import centralisation_by_phase, centralisation_from_network
from fas.product.clustering import cluster_passes
from fas.product.formation import phase_formation_from_actions, starting_formation_from_lineup
from fas.product.ingest import DataSpine
from fas.valuation.development_curves import fit_curve, project_to_peak

PHASE_LEN = 15  # minutes
N_PHASES = 6


# --------------------------------------------------------------------------- #
# Layer 1 — event & possession context: enrich the canonical action stream.
# --------------------------------------------------------------------------- #

def enrich_actions(spine: DataSpine):
    """Add xT, action value, minute, phase, zone, and opponent context."""
    actions = spine.actions.copy()
    xt = fit_xt(actions)

    minute = actions["timestamp_ms"].to_numpy(float) / 60_000.0
    actions["minute"] = minute
    phase_idx = np.clip((minute // PHASE_LEN).astype(int), 0, N_PHASES - 1)
    actions["phase_index"] = phase_idx
    actions["phase"] = [f"{p * PHASE_LEN}-{(p + 1) * PHASE_LEN}" for p in phase_idx]

    actions["zone_start"] = [zone_of(x, y) for x, y in
                             zip(actions["x_start"], actions["y_start"])]
    end_ok = actions["x_end"].notna() & actions["y_end"].notna()
    actions["zone_end"] = [
        zone_of(x, y) if ok else -1
        for x, y, ok in zip(actions["x_end"].fillna(0), actions["y_end"].fillna(0), end_ok)
    ]

    xt_start = np.array([xt.value(x, y) for x, y in zip(actions["x_start"], actions["y_start"])])
    xt_end = np.array([xt.value(x, y) if ok else np.nan
                       for x, y, ok in zip(actions["x_end"].fillna(0),
                                           actions["y_end"].fillna(0), end_ok)])
    actions["xt_start"] = xt_start
    actions["xt_end"] = xt_end

    is_move = actions["action_type"].isin(["pass", "carry"]) & actions["outcome"].astype(bool) & end_ok
    xt_added = np.where(is_move, xt_end - xt_start, 0.0)
    actions["xt_added"] = np.nan_to_num(xt_added)

    av = np.zeros(len(actions))
    is_shot = actions["action_type"] == "shot"
    av[is_shot.to_numpy()] = np.where(actions.loc[is_shot, "outcome"], 0.30, 0.03)
    av[is_move.to_numpy()] = actions.loc[is_move, "xt_added"]
    defen = actions["action_type"].isin(["pressure", "tackle", "interception", "recovery"]) \
        & actions["outcome"].astype(bool)
    av[defen.to_numpy() & ~is_move.to_numpy()] = 0.02
    actions["action_value"] = av

    # Opponent and friendly metadata per match.
    opp = {}
    for mid, grp in actions.groupby("match_id"):
        teams = list(grp["team_id"].unique())
        if len(teams) >= 2:
            opp[(mid, teams[0])] = teams[1]
            opp[(mid, teams[1])] = teams[0]
        else:
            opp[(mid, teams[0])] = teams[0]
    actions["opponent_id"] = [opp.get((m, t), t) for m, t in
                              zip(actions["match_id"], actions["team_id"])]

    names = dict(zip(spine.players["player_id"], spine.players["player_name"]))
    actions["player_name"] = actions["player_id"].map(names).fillna(
        actions["player_id"].astype(str))
    tnames = dict(zip(spine.teams["team_id"], spine.teams["team_name"]))
    actions["team_name"] = actions["team_id"].map(tnames).fillna(actions["team_id"].astype(str))

    meta = spine.manifest
    actions["competition"] = meta.get("competition")
    actions["season"] = meta.get("season")
    actions["data_source"] = meta.get("data_mode")
    actions["is_synthetic"] = meta.get("is_synthetic")
    return actions, xt


# --------------------------------------------------------------------------- #
# Helpers for the contract-compliant long tables.
# --------------------------------------------------------------------------- #

def _ctx(spine: DataSpine, **kw) -> dict:
    """Common context columns for every long-format artifact row."""
    m = spine.manifest
    base = {
        "competition": m.get("competition"),
        "season": m.get("season"),
        "data_source": m.get("data_mode"),
        "is_synthetic": m.get("is_synthetic"),
        "limitations": "; ".join(m.get("limitations", [])) or "none",
    }
    base.update(kw)
    return base


def _avg_positions(actions: pd.DataFrame, match_id: int, team_id: int) -> dict[int, tuple]:
    df = actions[(actions["match_id"] == match_id) & (actions["team_id"] == team_id)]
    if df.empty:
        return {}
    g = df.groupby("player_id")[["x_start", "y_start"]].mean()
    return {int(p): (float(r.x_start), float(r.y_start)) for p, r in g.iterrows()}


# --------------------------------------------------------------------------- #
# Layers 1–3 — per match, per team: networks, centralisation, zones, clusters.
# --------------------------------------------------------------------------- #

def build_match_layer(spine: DataSpine, actions: pd.DataFrame, xt):
    edges_rows, central_rows, formation_rows = [], [], []
    cluster_rows, zone_rows, match_rows = [], [], []

    keeper_by_team = {}
    for tid, grp in spine.players.groupby("team_id"):
        gks = grp[grp["position"].astype(str).str.contains("GK|Goalkeeper", case=False, na=False)]
        if not gks.empty:
            keeper_by_team[int(tid)] = int(gks.iloc[0]["player_id"])
    team_formation = (dict(zip(spine.teams["team_id"], spine.teams["formation"]))
                      if "formation" in spine.teams else {})

    for mid in sorted(actions["match_id"].unique()):
        macts = actions[actions["match_id"] == mid]
        for tid in sorted(macts["team_id"].unique()):
            tid = int(tid)
            tacts = macts[macts["team_id"] == tid]
            opp = int(tacts["opponent_id"].iloc[0])
            positions = _avg_positions(actions, mid, tid)

            # --- pass networks (full + per half) -> edges ---
            for half_label, period in (("all", None), ("H1", 1), ("H2", 2)):
                sub = macts if period is None else macts[macts["period"] == period]
                net = build_pass_network(sub, team_id=tid)
                for i, u in enumerate(net.players):
                    for j, v in enumerate(net.players):
                        w = net.W[i, j]
                        if w > 0:
                            ux, uy = positions.get(int(u), (np.nan, np.nan))
                            vx, vy = positions.get(int(v), (np.nan, np.nan))
                            edges_rows.append(_ctx(
                                spine, match_id=mid, team_id=tid, opponent_id=opp,
                                window=half_label, passer=int(u), receiver=int(v),
                                weight=float(w), passer_x=ux, passer_y=uy,
                                receiver_x=vx, receiver_y=vy))

            # --- centralisation per phase + match summary ---
            cb = centralisation_by_phase(macts, tid, exclude_keeper_id=keeper_by_team.get(tid))
            for r in cb.itertuples(index=False):
                central_rows.append(_ctx(
                    spine, match_id=mid, team_id=tid, opponent_id=opp,
                    phase=r.phase, phase_index=int(r.phase_index),
                    minute_start=int(r.minute_start), minute_end=int(r.minute_end),
                    centralisation=float(r.centralisation),
                    hub_player=None if r.hub_player is None else int(r.hub_player),
                    hub_share=float(r.hub_share), entropy=float(r.entropy),
                    n_players=int(r.n_players), n_passes=int(r.n_passes)))

            net_all = build_pass_network(macts, team_id=tid)
            cres = centralisation_from_network(net_all, exclude={keeper_by_team.get(tid, -1)})
            H = network_entropy(net_all)

            # --- formation (lineup high-confidence + phase low-confidence) ---
            lineup = spine.players[(spine.players["team_id"] == tid) &
                                   (spine.players.get("is_starter", True))]
            f_line = starting_formation_from_lineup(lineup) if "position" in lineup else None
            line_formation = None
            if f_line is not None and f_line.formation != "unknown":
                line_formation = f_line.formation
            elif team_formation.get(tid, "unknown") not in (None, "unknown"):
                line_formation = team_formation[tid]  # real StatsBomb formation
            f_phase = phase_formation_from_actions(macts, tid,
                                                   keeper_id=keeper_by_team.get(tid))
            if line_formation is not None:
                formation_rows.append(_ctx(spine, match_id=mid, team_id=tid, opponent_id=opp,
                                           confidence="lineup", formation=line_formation,
                                           lines=str(getattr(f_line, "lines", []))))
            formation_rows.append(_ctx(spine, match_id=mid, team_id=tid, opponent_id=opp,
                                       confidence="phase", formation=f_phase.formation,
                                       lines=str(f_phase.lines)))

            # --- pass clusters ---
            pc = cluster_passes(macts, team_id=tid)
            for r in pc.summary.itertuples(index=False):
                # mean xT added of passes near this cluster centroid.
                cluster_rows.append(_ctx(
                    spine, match_id=mid, team_id=tid, opponent_id=opp,
                    cluster_id=int(r.cluster_id), label=r.label, size=int(r.size),
                    x_start=float(r.x_start), y_start=float(r.y_start),
                    x_end=float(r.x_end), y_end=float(r.y_end),
                    mean_length=float(r.mean_length), mean_angle_deg=float(r.mean_angle_deg),
                    method=pc.method, n_noise=int(pc.n_noise)))

            # --- zone flow / buildup corridors ---
            zg = build_zone_graph(macts, team_id=tid, xt_model=xt, min_count=3)
            flow = buildup_potency(zg)
            for i in range(zg.capacity.shape[0]):
                for j in range(zg.capacity.shape[1]):
                    if zg.capacity[i, j] > 0:
                        zone_rows.append(_ctx(
                            spine, match_id=mid, team_id=tid, opponent_id=opp,
                            z0=i, z1=j, capacity=float(zg.capacity[i, j]),
                            value=float(zg.value[i, j]),
                            flow_used=float(flow.corridors.get((i, j), 0))))

            # --- match-level metric rows (long, contract-compliant) ---
            n_pass = int(((tacts["action_type"] == "pass")).sum())
            n_shot = int((tacts["action_type"] == "shot").sum())
            xg = float(tacts.loc[tacts["action_type"] == "shot", "action_value"].sum())
            goals = int(((tacts["action_type"] == "shot") & tacts["outcome"]).sum())
            for name, value, model, n in [
                ("network_entropy", H, "pass_network/shannon", net_all.n),
                ("centralisation", cres.index, "pagerank/freeman", net_all.n),
                ("hub_share", cres.hub_share, "pagerank/freeman", net_all.n),
                ("buildup_flow_value", float(flow.flow_value), "min_cost_max_flow", n_pass),
                ("buildup_xt_reward", float(flow.total_value), "min_cost_max_flow", n_pass),
                ("passes", float(n_pass), "count", n_pass),
                ("shots", float(n_shot), "count", n_shot),
                ("xg_proxy", xg, "xt_shot_proxy", n_shot),
                ("goals", float(goals), "count", n_shot),
                ("xt_added_total", float(tacts["xt_added"].sum()), "xt_value_iteration", n_pass),
            ]:
                match_rows.append(_ctx(
                    spine, match_id=mid, team_id=tid, opponent_id=opp, phase="all",
                    period=0, minute_start=0, minute_end=90, metric_name=name,
                    metric_value=float(value), model_name=model, model_version="fas-0.2",
                    sample_size=int(n), hub_player=cres.hub_player))

    return {
        "pass_network_edges": pd.DataFrame(edges_rows),
        "centralisation": pd.DataFrame(central_rows),
        "formations": pd.DataFrame(formation_rows),
        "pass_clusters": pd.DataFrame(cluster_rows),
        "zone_flow": pd.DataFrame(zone_rows),
        "match_artifacts": pd.DataFrame(match_rows),
    }


# --------------------------------------------------------------------------- #
# Layer 4/5 — players, roles, valuation, form, development.
# --------------------------------------------------------------------------- #

def build_player_layer(spine: DataSpine, actions: pd.DataFrame):
    players = spine.players.set_index("player_id")
    rng = np.random.default_rng(11)

    # per-player season aggregates from the enriched action stream
    agg = actions.groupby("player_id").agg(
        actions_total=("action_type", "size"),
        xt_added=("xt_added", "sum"),
        action_value=("action_value", "sum"),
        matches=("match_id", "nunique"),
    )
    counts = actions.pivot_table(index="player_id", columns="action_type",
                                 values="minute", aggfunc="size", fill_value=0)
    feat = agg.join(counts, how="left").fillna(0.0)
    pid_index = [int(p) for p in feat.index]
    feat["minutes"] = feat["matches"] * 90.0
    per90 = feat["xt_added"] / feat["minutes"].replace(0, np.nan) * 90.0
    feat["xt_added_90"] = per90.fillna(0.0)

    # PVS via robust PCA -> low-rank embedding (Layer 5).
    cent = centrality_features(actions)
    F = feat.drop(columns=["matches"]).to_numpy(dtype=float)
    F = np.column_stack([F, cent.reindex(feat.index).fillna(0.0).to_numpy()])
    F = np.nan_to_num(F)
    L, _ = robust_pca(F)
    z = low_rank_embedding(L)
    positions = {int(p): players.loc[p, "position"] if p in players.index else "UNK"
                 for p in pid_index}
    pos_for_pvs = {p: (pos if pos in ROLES else _nearest_role(pos)) for p, pos in positions.items()}
    pvs = player_value_scores(z, pid_index, pos_for_pvs)

    # form trend (Kalman) per player across phases.
    form_obs = _form_observations(actions)
    form = kalman_form(form_obs) if len(form_obs) else None
    form_latest = _form_latest(form) if form is not None else {}

    # synthetic ages, market values; fair value from PVS + development curve.
    rows = []
    role_groups = players["role_group"].to_dict() if "role_group" in players else {}
    for pid in pid_index:
        pos = positions[pid]
        rg = role_groups.get(pid, "Midfielder")
        age = int(rng.integers(19, 33))
        pv = float(pvs.get(pid, 0.0))
        fair_value = round(2.0 + 60.0 * pv ** 1.5, 2)
        market_value = round(max(0.5, fair_value * float(rng.uniform(0.6, 1.5))), 2)
        # development projection
        ages = np.array([age - 2, age, age + 2], dtype=float)
        vals = np.array([pv * 0.8, pv, pv * 0.95])
        params = fit_curve(ages, vals)
        peak = float(project_to_peak(age, pv, params))
        rows.append({
            "player_id": pid,
            "player_name": players.loc[pid, "player_name"] if pid in players.index else str(pid),
            "team_id": int(players.loc[pid, "team_id"]) if pid in players.index else -1,
            "position": pos, "role_group": rg,
            "minutes": float(feat.loc[pid, "minutes"]),
            "matches": int(feat.loc[pid, "matches"]),
            "actions_total": int(feat.loc[pid, "actions_total"]),
            "xt_added": float(feat.loc[pid, "xt_added"]),
            "xt_added_90": float(feat.loc[pid, "xt_added_90"]),
            "action_value": float(feat.loc[pid, "action_value"]),
            "pvs": pv, "fair_value_eur_m": fair_value, "market_value_eur_m": market_value,
            "value_gap_eur_m": round(fair_value - market_value, 2),
            "age": age, "projected_peak_pvs": round(peak, 4),
            "form_latest": float(form_latest.get(pid, np.nan)),
            "centrality_pagerank": float(cent.get("pagerank", pd.Series()).get(pid, 0.0))
            if isinstance(cent, pd.DataFrame) else 0.0,
        })
    players_df = pd.DataFrame(rows)

    # role-group percentiles (never a naked ranking).
    for col in ("pvs", "xt_added_90", "action_value"):
        players_df[f"{col}_pct"] = players_df.groupby("role_group")[col].rank(pct=True) * 100.0

    # long-format player_artifacts with full context.
    long_rows = []
    for r in players_df.itertuples(index=False):
        for metric, value, model in [
            ("pvs", r.pvs, "robust_pca/low_rank_embedding/PVS"),
            ("xt_added_90", r.xt_added_90, "xt_value_iteration"),
            ("fair_value_eur_m", r.fair_value_eur_m, "fair_value_regression_proxy"),
            ("market_value_eur_m", r.market_value_eur_m, "synthetic_market"),
            ("projected_peak_pvs", r.projected_peak_pvs, "beta_career_curve"),
            ("form_latest", r.form_latest, "kalman_form"),
        ]:
            long_rows.append(_ctx(
                spine, player_id=r.player_id, player_name=r.player_name,
                team_id=r.team_id, position=r.position, role_group=r.role_group,
                metric_name=metric, metric_value=float(value) if pd.notna(value) else None,
                model_name=model, model_version="fas-0.2",
                sample_size=int(r.matches * 90), minutes=float(r.minutes),
                minutes_threshold=90))
    return players_df, pd.DataFrame(long_rows), pvs, pos_for_pvs


def centrality_features(actions: pd.DataFrame) -> pd.DataFrame:
    """PageRank centrality per player, pooled across that player's matches."""
    frames = []
    for (mid, tid), grp in actions.groupby(["match_id", "team_id"]):
        net = build_pass_network(grp, team_id=int(tid))
        if net.n:
            frames.append(centrality_table(net))
    if not frames:
        return pd.DataFrame()
    allc = pd.concat(frames)
    return allc.groupby(level=0).mean()


def _nearest_role(pos: str) -> str:
    group_default = {"Goalkeeper": "GK", "Defender": "LCB", "Midfielder": "CM", "Forward": "ST"}
    return group_default.get(pos, "CM")


def _form_observations(actions: pd.DataFrame) -> pd.DataFrame:
    df = actions.assign(t=actions["phase_index"].astype(int)
                        + (actions["match_id"].rank(method="dense").astype(int) - 1) * N_PHASES)
    obs = df.groupby(["player_id", "t"], as_index=False)["action_value"].mean()
    return obs.rename(columns={"action_value": "performance"})


def _form_latest(form) -> dict:
    try:
        states = form.states  # DataFrame player_id, t, mean
        last = states.sort_values("t").groupby("player_id").tail(1)
        return dict(zip(last["player_id"], last["mean"]))
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Layer 6 — matchups, forecasts, decision support.
# --------------------------------------------------------------------------- #

def build_matchup_layer(spine: DataSpine, actions: pd.DataFrame, players_df: pd.DataFrame):
    results = spine.results
    matchup_rows, scoreline_rows = [], []
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    scoring = fit_dixon_coles(results)
    paired = _fit_paired(results)
    style = {int(t): team_style_distribution(actions[actions["team_id"] == t], team_id=int(t))
             for t in spine.teams["team_id"]}

    # Only compute matchups between teams that share a competition+season cohort,
    # so the corpus can span many leagues/eras without an O(T^2) cross product or
    # meaningless cross-league pairings.
    pairs: set[tuple[int, int]] = set()
    if {"competition", "season"}.issubset(spine.teams.columns):
        for _, grp in spine.teams.groupby(["competition", "season"]):
            cohort = sorted(int(t) for t in grp["team_id"])
            for i in cohort:
                for j in cohort:
                    if i != j:
                        pairs.add((i, j))
    else:
        ts = sorted(int(t) for t in spine.teams["team_id"])
        pairs = {(i, j) for i in ts for j in ts if i != j}
    # safety cap for very large corpora
    pairs_list = sorted(pairs)[:4000]

    for i, j in pairs_list:
        if i in style and j in style:
            probs = scoring.outcome_probabilities(i, j, max_goals=5)
            eg = scoring.expected_goals(i, j)
            bt = _paired_prob(paired, i, j)
            sd = _style_distance(style.get(i), style.get(j))
            # pressing min-cut targets: hub of opponent's network.
            targets = _pressing_targets(actions, j)
            matchup_rows.append(_ctx(
                spine, entity_i=i, entity_j=j,
                p_home_win=probs["home"], p_draw=probs["draw"], p_away_win=probs["away"],
                expected_goals_i=float(eg[0]), expected_goals_j=float(eg[1]),
                bt_win_prob=float(bt), style_distance=float(sd),
                pressing_targets=",".join(str(t) for t in targets),
                model_name="dixon_coles + bradley_terry_davidson + fisher_rao + min_cut",
                model_version="fas-0.2", sample_size=int(len(results))))
            dist = scoring.score_distribution(i, j, max_goals=5)
            for r in dist.itertuples(index=False):
                scoreline_rows.append(_ctx(
                    spine, entity_i=i, entity_j=j,
                    home_goals=int(r.home_goals), away_goals=int(r.away_goals),
                    prob=float(r.prob)))
    return pd.DataFrame(matchup_rows), pd.DataFrame(scoreline_rows)


def _fit_paired(results: pd.DataFrame):
    rows = []
    for r in results.itertuples(index=False):
        outcome = "win" if r.home_goals > r.away_goals else (
            "loss" if r.home_goals < r.away_goals else "tie")
        rows.append({"team_i": r.home_team, "team_j": r.away_team, "outcome": outcome, "home": 1.0})
    try:
        return fit_bradley_terry_davidson(pd.DataFrame(rows))
    except Exception:
        return None


def _paired_prob(model, i, j) -> float:
    if model is None:
        return 0.5
    try:
        p = model.probabilities(i, j)
        return float(p.get("win", p) if isinstance(p, dict) else p)
    except Exception:
        return 0.5


def _style_distance(a, b) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    idx = a.index.union(b.index)
    return float(fisher_rao_distance(a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)))


def _pressing_targets(actions: pd.DataFrame, team_id: int, k: int = 3) -> list[int]:
    net = build_pass_network(actions[actions["team_id"] == team_id], team_id=team_id)
    if net.n == 0:
        return []
    cents = centrality_table(net)
    return [int(p) for p in cents["pagerank"].nlargest(k).index]


# --------------------------------------------------------------------------- #
# Insight engine — validated + exploratory cards with full context.
# --------------------------------------------------------------------------- #

def build_insights(spine: DataSpine, actions: pd.DataFrame,
                   central_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    cards: list[dict] = []
    tnames = dict(zip(spine.teams["team_id"], spine.teams["team_name"]))
    pnames = dict(zip(players_df["player_id"], players_df["player_name"])) if not players_df.empty else {}

    # 1. FDR-controlled player xT-added departures (validated).
    move = actions[actions["action_type"].isin(["pass", "carry"])].copy()
    if not move.empty:
        baseline = float(move["xt_added"].mean())
        frame = pd.DataFrame({
            "player_id": move["player_id"],
            "metric": move["xt_added"].to_numpy(),
            "expected": baseline,
        })
        for ins in scan_departures(frame, entity_col="player_id", metric_col="metric",
                                   expected_col="expected", min_n=20, fdr_alpha=0.1):
            name = pnames.get(ins.entity_id, str(ins.entity_id))
            direction = "above" if ins.effect > 0 else "below"
            cards.append({
                "title": f"{name}: xT per action runs {direction} league baseline",
                "entity": name, "entity_id": int(ins.entity_id),
                "scope": "player", "context": ins.context,
                "claim": f"{name}'s mean xT added is {ins.effect:+.4f} vs the "
                         f"{baseline:+.4f} league baseline.",
                "evidence": f"95% CI [{ins.ci[0]:+.4f}, {ins.ci[1]:+.4f}], "
                            f"q={ins.q_value:.3f}, p={ins.p_value:.3f}.",
                "method": ins.method, "metric": "xt_added",
                "effect": float(ins.effect), "ci_low": float(ins.ci[0]),
                "ci_high": float(ins.ci[1]), "p_value": float(ins.p_value),
                "q_value": float(ins.q_value),
                "sample_size": int((move["player_id"] == ins.entity_id).sum()),
                "baseline": baseline, "validation_status": "validated (FDR q<=0.1)",
                "caveats": "event-only data; no freeze-frame pressure context.",
                "next_look": "inspect this player's pass clusters and zone-flow corridors.",
            })

    # 2. Centralisation hub drift per team (exploratory, descriptive).
    if not central_df.empty:
        for tid, grp in central_df.groupby("team_id"):
            grp = grp.sort_values("phase_index")
            if grp["centralisation"].notna().sum() < 2:
                continue
            early = grp.head(2)["centralisation"].mean()
            late = grp.tail(2)["centralisation"].mean()
            hub_early = grp.head(2)["hub_player"].dropna()
            hub_late = grp.tail(2)["hub_player"].dropna()
            drift = (not hub_early.empty and not hub_late.empty
                     and hub_early.iloc[0] != hub_late.iloc[-1])
            tname = tnames.get(int(tid), str(tid))
            change = late - early
            samp = int(grp["n_passes"].sum())
            ci = bootstrap_ci(grp["centralisation"].dropna().to_numpy())
            cards.append({
                "title": f"{tname}: control {'shifted' if drift else 'stayed'} "
                         f"as the match progressed",
                "entity": tname, "entity_id": int(tid),
                "scope": "team", "context": "phase trend",
                "claim": f"centralisation moved {change:+.3f} from the opening to the "
                         f"closing phases" + (", and the dominant hub changed players"
                                              if drift else ", with a stable hub."),
                "evidence": f"phase centralisation range "
                            f"[{grp['centralisation'].min():.3f}, "
                            f"{grp['centralisation'].max():.3f}], 95% CI "
                            f"[{ci[0]:.3f}, {ci[1]:.3f}].",
                "method": "PageRank Freeman centralisation per 15-min phase + bootstrap CI",
                "metric": "centralisation", "effect": float(change),
                "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                "p_value": np.nan, "q_value": np.nan, "sample_size": samp,
                "baseline": float(early), "validation_status": "exploratory (not FDR-controlled)",
                "caveats": "phase windows conflate possessions; small per-phase samples.",
                "next_look": "compare phase pass networks H1 vs H2 for this team.",
            })

    return pd.DataFrame(cards)


# --------------------------------------------------------------------------- #
# Orchestration helpers used by build.py
# --------------------------------------------------------------------------- #

def build_squad(spine: DataSpine, players_df: pd.DataFrame, pvs: pd.Series,
                pos_for_pvs: dict) -> dict[str, Any]:
    """Run the squad-selection MILP over the strongest pool (Layer 5)."""
    if players_df.empty:
        return {}
    pool_df = players_df.sort_values("pvs", ascending=False).head(30)
    pool = [int(p) for p in pool_df["player_id"]]
    pvs_pool = pd.Series({p: float(pvs.get(p, 0.3)) for p in pool})
    eligible = {p: {pos_for_pvs.get(p, "CM"),
                    ROLES[(ROLES.index(pos_for_pvs.get(p, "CM")) + 1) % len(ROLES)]}
                for p in pool}
    rng = np.random.default_rng(3)
    prob = SquadProblem(
        players=pool, pvs=pvs_pool, eligible_roles=eligible,
        wage=pd.Series({p: float(v) for p, v in zip(pool, rng.uniform(0.2, 2.0, len(pool)))}),
        fair_value=pd.Series(dict(zip(pool, pool_df["fair_value_eur_m"].to_numpy()))),
        age=pd.Series(dict(zip(pool, pool_df["age"].to_numpy()))),
        homegrown=pd.Series({p: bool(b) for p, b in zip(pool, rng.random(len(pool)) > 0.7)}),
        squad_size=18, min_young=3,
    )
    try:
        sol = solve_squad(prob, time_limit=20)
        return {"status": sol.status, "objective": float(sol.objective),
                "formation": sol.formation, "starters": [int(s) for s in sol.starters]}
    except Exception as exc:  # pragma: no cover
        return {"status": f"error: {exc}", "objective": 0.0, "formation": None, "starters": []}
