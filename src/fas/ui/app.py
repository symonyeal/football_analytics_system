"""Streamlit analytics workspace for ``fas``.

Run via ``python -m fas.cli ui`` (recommended) or directly:

    streamlit run src/fas/ui/app.py

Reads pre-built artifacts from ``data/processed`` and never re-runs models on
interaction. Six views mirror an internal club analysis tool. Global filters
(competition / season / team) let the workspace span the whole real-data corpus
— every team, league, and era StatsBomb Open Data exposes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fas.ui import charts, theme  # noqa: E402
from fas.product.loader import ensure_artifacts, load_product  # noqa: E402

DATA_ROOT = os.environ.get("FAS_DATA_ROOT", "data")
ALL = "— All —"

st.set_page_config(page_title="FAS — Football Intelligence", layout="wide",
                   initial_sidebar_state="expanded", page_icon="⚽")


@st.cache_data(show_spinner="Loading product artifacts ...")
def _load(data_root: str):
    ensure_artifacts(data_root, allow_download=False, verbose=False)
    p = load_product(data_root)
    return p.tables, p.summary, p.manifest


def _names(teams, players):
    tn = dict(zip(teams["team_id"], teams["team_name"]))
    pn = dict(zip(players["player_id"], players["player_name"])) if not players.empty else {}
    return tn, pn


def main() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    tables, summary, manifest = _load(DATA_ROOT)
    teams, players, matches = tables["teams"], tables["players"], tables["matches"]
    tn, pn = _names(teams, players)
    mode = manifest.get("data_mode", "local")

    span = manifest.get("season", "")
    comps = manifest.get("competitions") or ([manifest.get("competition")]
                                             if manifest.get("competition") else [])
    subtitle = (f"{summary.get('n_matches')} matches · {summary.get('n_teams')} teams · "
                f"{summary.get('n_players')} players · {len(comps)} competition(s) · "
                f"seasons {span} · source: {manifest.get('source')}")
    st.markdown(theme.hero("FAS — Football Intelligence", subtitle, mode),
                unsafe_allow_html=True)

    # ---- sidebar: navigation + global scope filters ----
    st.sidebar.title("⚽ FAS")
    view = st.sidebar.radio("View", [
        "Match Workspace", "Team Style", "Player Intelligence",
        "Matchup Lab", "Recruitment & Squad", "Data Quality"])
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scope")
    comp_opts = [ALL] + (sorted(teams["competition"].dropna().unique())
                         if "competition" in teams else [])
    sel_comp = st.sidebar.selectbox("Competition", comp_opts)
    season_opts = [ALL]
    if "season" in matches:
        if sel_comp != ALL:
            season_opts += sorted(matches[matches["competition"] == sel_comp]
                                  ["season"].dropna().unique())
        else:
            season_opts += sorted(matches["season"].dropna().unique())
    sel_season = st.sidebar.selectbox("Season", season_opts)

    scope = _apply_scope(tables, sel_comp, sel_season)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"In scope: {len(scope['matches'])} matches · "
                       f"{len(scope['teams'])} teams · {len(scope['players'])} players")

    if view == "Match Workspace":
        _match_workspace(tables, scope, tn, pn, manifest)
    elif view == "Team Style":
        _team_style(tables, scope, tn, pn, manifest)
    elif view == "Player Intelligence":
        _player_intel(tables, scope, tn, manifest)
    elif view == "Matchup Lab":
        _matchup_lab(tables, scope, tn, manifest)
    elif view == "Recruitment & Squad":
        _recruitment(tables, scope, summary, manifest)
    else:
        _data_quality(tables, summary, manifest)


def _apply_scope(tables, comp, season):
    matches, teams, players = tables["matches"].copy(), tables["teams"].copy(), tables["players"].copy()
    if comp != ALL and "competition" in teams:
        teams = teams[teams["competition"] == comp]
        if "competition" in matches:
            matches = matches[matches["competition"] == comp]
    if season != ALL and "season" in matches:
        matches = matches[matches["season"] == season]
    team_ids = set(teams["team_id"])
    if "team_id" in players:
        players = players[players["team_id"].isin(team_ids)]
    # keep only teams that actually appear in the scoped matches
    if not matches.empty:
        live = set(matches["home_team_id"]) | set(matches["away_team_id"])
        teams = teams[teams["team_id"].isin(live)]
        players = players[players["team_id"].isin(live)]
    return {"matches": matches, "teams": teams, "players": players}


# --------------------------------------------------------------------------- #
# View 1 — Match Workspace
# --------------------------------------------------------------------------- #

def _match_workspace(tables, scope, tn, pn, manifest):
    st.markdown('<div class="fas-section">Match Workspace</div>', unsafe_allow_html=True)
    matches = scope["matches"]
    actions = tables["actions"]
    if matches.empty:
        st.info("No matches in scope. Widen the competition/season filters.")
        return
    matches = matches.sort_values("date", ascending=False)
    labels = {int(r.match_id): f"{r.season} · {r.home_team} {r.home_goals}-{r.away_goals} "
                               f"{r.away_team}" for r in matches.itertuples(index=False)}
    mid = st.selectbox("Match", list(labels), format_func=lambda m: labels[m])
    row = matches[matches["match_id"] == mid].iloc[0]
    team_opts = [int(row["home_team_id"]), int(row["away_team_id"])]
    team = st.radio("Team", team_opts, format_func=lambda t: tn.get(t, str(t)), horizontal=True)
    opp = team_opts[1] if team == team_opts[0] else team_opts[0]

    macts = actions[actions["match_id"] == mid]
    tacts = macts[macts["team_id"] == team]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score", f"{row['home_goals']}-{row['away_goals']}")
    c2.metric(f"{tn.get(team)} passes", int((tacts["action_type"] == "pass").sum()))
    c3.metric("Shots", int((tacts["action_type"] == "shot").sum()))
    c4.metric("xT added", f"{tacts['xt_added'].sum():.2f}")
    c5.metric("Competition", row.get("competition", "—"))
    st.caption(f"{row.get('competition')} {row.get('season')} · {row.get('date')} · "
               f"sample {len(tacts)} actions · 360 freeze frames: "
               f"{'yes' if manifest.get('has_360') else 'no (event-only)'}")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(charts.shot_map(macts, team_id=team,
                        title=f"Shot map — {tn.get(team)}"), use_container_width=True)
        edges = tables["pass_network_edges"]
        ed = edges[(edges["match_id"] == mid) & (edges["team_id"] == team)
                   & (edges["window"] == "all")]
        st.plotly_chart(charts.pass_network(ed, names=pn,
                        title=f"Pass network — {tn.get(team)}"), use_container_width=True)
    with b:
        st.plotly_chart(charts.xt_timeline(macts, team_id=team, opponent_id=opp,
                        team_name=tn.get(team), opp_name=tn.get(opp),
                        title="Cumulative xT added"), use_container_width=True)
        zf = tables["zone_flow"]
        zf = zf[(zf["match_id"] == mid) & (zf["team_id"] == team)]
        st.plotly_chart(charts.zone_flow_map(zf,
                        title=f"Build-up corridors — {tn.get(team)}"), use_container_width=True)

    st.markdown('<div class="fas-section">Centralisation through the match</div>',
                unsafe_allow_html=True)
    cen = tables["centralisation"]
    cen = cen[(cen["match_id"] == mid) & (cen["team_id"] == team)]
    st.plotly_chart(charts.centralisation_trend(cen, names=pn,
                    title=f"Influence concentration by phase — {tn.get(team)}"),
                    use_container_width=True)
    _insight_cards(tables["insights"], scope_entity=team, title="Top validated insights")


# --------------------------------------------------------------------------- #
# View 2 — Team Style
# --------------------------------------------------------------------------- #

def _team_style(tables, scope, tn, pn, manifest):
    st.markdown('<div class="fas-section">Team Style</div>', unsafe_allow_html=True)
    teams = scope["teams"]
    if teams.empty:
        st.info("No teams in scope.")
        return
    team = st.selectbox("Team", list(teams["team_id"]),
                        format_func=lambda t: tn.get(t, str(t)))

    forms = tables["formations"]
    tf = forms[forms["team_id"] == team]
    line = tf[tf["confidence"] == "lineup"]["formation"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Formation (lineup)", line.mode().iloc[0] if not line.empty else "—")
    phase = tf[tf["confidence"] == "phase"]["formation"]
    c2.metric("Phase shape", phase.mode().iloc[0] if not phase.empty else "—",
              help="Lower-confidence: from average action locations, not the lineup.")
    trow = teams[teams["team_id"] == team].iloc[0]
    c3.metric("Competition", trow.get("competition", "—"))

    a, b = st.columns(2)
    with a:
        cen = tables["centralisation"]
        cen = cen[cen["team_id"] == team].groupby("phase_index", as_index=False).agg(
            phase=("phase", "first"), centralisation=("centralisation", "mean"),
            entropy=("entropy", "mean"), hub_player=("hub_player", "first"))
        st.plotly_chart(charts.centralisation_trend(cen, names=pn,
                        title=f"Centralisation & entropy — {tn.get(team)}"),
                        use_container_width=True)
    with b:
        mm = tables["matchup_artifacts"]
        cohort_ids = set(scope["teams"]["team_id"])
        mm = mm[mm["entity_i"].isin(cohort_ids) & mm["entity_j"].isin(cohort_ids)]
        st.plotly_chart(charts.style_distance_matrix(mm, scope["teams"],
                        title="Style distance matrix (Fisher-Rao)"), use_container_width=True)

    st.markdown('<div class="fas-section">Pass cluster atlas</div>', unsafe_allow_html=True)
    pc = tables["pass_clusters"]
    pc = pc[pc["team_id"] == team]
    pc_agg = (pc.groupby("label").agg(
        size=("size", "sum"), x_start=("x_start", "mean"), y_start=("y_start", "mean"),
        x_end=("x_end", "mean"), y_end=("y_end", "mean"),
        mean_length=("mean_length", "mean")).reset_index())
    st.plotly_chart(charts.pass_cluster_atlas(pc_agg,
                    title=f"Pass clusters — {tn.get(team)} (spatial style)"),
                    use_container_width=True)
    st.caption(f"Clustering: {pc['method'].iloc[0] if not pc.empty else 'n/a'}; "
               "noise/unclustered passes excluded honestly.")
    _interpretation_card(tables, team, tn)


# --------------------------------------------------------------------------- #
# View 3 — Player Intelligence
# --------------------------------------------------------------------------- #

def _player_intel(tables, scope, tn, manifest):
    st.markdown('<div class="fas-section">Player Intelligence</div>', unsafe_allow_html=True)
    players = scope["players"]
    if players.empty:
        st.info("No players in scope.")
        return
    c0, c1 = st.columns(2)
    rg = c0.selectbox("Role group", ["All"] + sorted(players["role_group"].unique()))
    pool = players if rg == "All" else players[players["role_group"] == rg]
    pid = c1.selectbox("Player", list(pool["player_id"]),
                       format_func=lambda p: pool.set_index("player_id").loc[p, "player_name"])
    row = players[players["player_id"] == pid].iloc[0]

    st.markdown(f"#### {row['player_name']} · {row['position']} ({row['role_group']}) "
                f"· {tn.get(int(row['team_id']), '')}")
    c = st.columns(5)
    c[0].metric("Minutes", int(row["minutes"]))
    c[1].metric("PVS", f"{row['pvs']:.3f}", f"{row['pvs_pct']:.0f}th pct in role")
    c[2].metric("xT added /90", f"{row['xt_added_90']:.4f}")
    c[3].metric("Fair value", f"€{row['fair_value_eur_m']:.1f}M")
    c[4].metric("Value gap", f"€{row['value_gap_eur_m']:.1f}M")
    st.caption(f"Percentiles within role group at a 90-minute threshold · "
               f"context: {manifest.get('competition')} {manifest.get('season')} · "
               "scores are model-derived (event-only).")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(charts.role_profile(players, row,
                        title="Role-adjusted percentile profile"), use_container_width=True)
    with b:
        st.markdown('<div class="fas-card"><b>Development projection</b><br>'
                    f"Age {int(row['age'])} · projected peak PVS "
                    f"<b>{row['projected_peak_pvs']:.3f}</b> (Beta career curve)."
                    "</div>", unsafe_allow_html=True)
        fl = row["form_latest"]
        st.markdown('<div class="fas-card"><b>Form</b><br>latest Kalman state: '
                    f"<b>{fl:.4f}</b></div>" if pd.notna(fl) else
                    '<div class="fas-card"><b>Form</b><br>n/a</div>',
                    unsafe_allow_html=True)
    _insight_cards(tables["insights"], scope_entity=int(pid),
                   title="Contextual insight cards for this player")


# --------------------------------------------------------------------------- #
# View 4 — Matchup Lab
# --------------------------------------------------------------------------- #

def _matchup_lab(tables, scope, tn, manifest):
    st.markdown('<div class="fas-section">Matchup Lab</div>', unsafe_allow_html=True)
    mm = tables["matchup_artifacts"]
    cohort = sorted(set(scope["teams"]["team_id"]))
    mm = mm[mm["entity_i"].isin(cohort) & mm["entity_j"].isin(cohort)]
    if mm.empty:
        st.info("No precomputed matchups in scope (matchups exist within a "
                "competition+season cohort). Narrow the scope to one league/season.")
        return
    ids = sorted(set(mm["entity_i"]))
    c1, c2 = st.columns(2)
    i = c1.selectbox("Team", ids, format_func=lambda t: tn.get(t), key="mi")
    j_opts = sorted(set(mm[mm["entity_i"] == i]["entity_j"]))
    j = c2.selectbox("Opponent", j_opts, format_func=lambda t: tn.get(t), key="mj")
    row = mm[(mm["entity_i"] == i) & (mm["entity_j"] == j)]
    if row.empty:
        st.info("No matchup record for this pair.")
        return
    row = row.iloc[0]
    probs = {"home": row["p_home_win"], "draw": row["p_draw"], "away": row["p_away_win"]}

    a, b = st.columns(2)
    with a:
        st.plotly_chart(charts.outcome_bars(probs, home=tn.get(i), away=tn.get(j),
                        title="Win / draw / loss (Dixon-Coles)"), use_container_width=True)
    with b:
        sc = tables["scorelines"]
        sc = sc[(sc["entity_i"] == i) & (sc["entity_j"] == j)]
        st.plotly_chart(charts.scoreline_grid(sc,
                        title=f"Scoreline — {tn.get(i)} vs {tn.get(j)}"),
                        use_container_width=True)

    st.markdown('<div class="fas-section">Tactical levers</div>', unsafe_allow_html=True)
    c = st.columns(3)
    c[0].metric("Expected goals", f"{row['expected_goals_i']:.2f} – {row['expected_goals_j']:.2f}")
    c[1].metric("Style distance", f"{row['style_distance']:.3f}")
    c[2].metric("Bradley-Terry edge", f"{row['bt_win_prob']:.0%}")
    targets = [t for t in str(row["pressing_targets"]).split(",") if t]
    pn = dict(zip(tables["players"]["player_id"], tables["players"]["player_name"]))
    named = ", ".join(pn.get(int(t), t) for t in targets) if targets else "—"
    st.markdown(f'<div class="fas-card"><b>Pressing min-cut targets (opponent hubs):</b> '
                f'{named}</div>', unsafe_allow_html=True)
    st.caption("Forecast: Dixon-Coles weighted Poisson; paired comparison: "
               "Bradley-Terry-Davidson; style: Fisher-Rao; pressing: PageRank hubs. "
               f"Sample: {int(row['sample_size'])} results. Event-only; ratings assumed "
               "stable over the sample.")


# --------------------------------------------------------------------------- #
# View 5 — Recruitment & Squad
# --------------------------------------------------------------------------- #

def _recruitment(tables, scope, summary, manifest):
    st.markdown('<div class="fas-section">Recruitment &amp; Squad</div>', unsafe_allow_html=True)
    players = scope["players"]
    if players.empty:
        st.info("No players in scope.")
        return
    c0, c1 = st.columns([2, 1])
    rg = c0.multiselect("Role groups", sorted(players["role_group"].unique()),
                        default=sorted(players["role_group"].unique()))
    max_min = int(players["minutes"].max()) if not players.empty else 90
    min_minutes = c1.slider("Minimum minutes", 0, max_min, 0, 90)
    pool = players[players["role_group"].isin(rg) & (players["minutes"] >= min_minutes)]

    st.plotly_chart(charts.value_vs_market(pool,
                    title="Model fair value vs market value (bubble = PVS)"),
                    use_container_width=True)

    st.markdown('<div class="fas-section">Shortlist (role-contextual)</div>',
                unsafe_allow_html=True)
    cols = ["player_name", "position", "role_group", "minutes", "pvs", "pvs_pct",
            "xt_added_90", "fair_value_eur_m", "market_value_eur_m", "value_gap_eur_m", "age"]
    st.dataframe(pool.sort_values("value_gap_eur_m", ascending=False)[cols]
                 .reset_index(drop=True), use_container_width=True, height=330)
    st.caption("PVS via robust PCA → low-rank embedding; percentiles within role group at "
               "a 90-minute threshold. Cross-league normalization applies when leagues mix.")

    sq = summary.get("squad_optimization", {})
    if sq:
        st.markdown('<div class="fas-section">Squad optimizer (MILP)</div>',
                    unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Status", sq.get("status", "—"))
        c[1].metric("Formation", sq.get("formation", "—"))
        c[2].metric("Objective", f"{sq.get('objective', 0):.2f}")
        names = dict(zip(tables["players"]["player_id"], tables["players"]["player_name"]))
        st.markdown('<div class="fas-card"><b>Selected XI:</b> '
                    + ", ".join(names.get(int(s), str(s)) for s in sq.get("starters", []))
                    + "</div>", unsafe_allow_html=True)
        st.caption("Maximizes squad PVS subject to positional coverage, squad size, and "
                   "youth-quota constraints (PuLP/CBC).")


# --------------------------------------------------------------------------- #
# View 6 — Data Quality
# --------------------------------------------------------------------------- #

def _data_quality(tables, summary, manifest):
    st.markdown('<div class="fas-section">Data Quality</div>', unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Data mode", manifest.get("data_mode"))
    c[1].metric("Matches", summary.get("n_matches"))
    c[2].metric("Teams", summary.get("n_teams"))
    c[3].metric("Players", summary.get("n_players"))
    c[4].metric("360 frames", "yes" if manifest.get("has_360") else "no")

    if manifest.get("competitions"):
        st.markdown('<div class="fas-section">Coverage</div>', unsafe_allow_html=True)
        st.write("**Competitions:** " + ", ".join(manifest["competitions"]))
        st.write("**Seasons:** " + ", ".join(manifest.get("seasons", [])))

    st.markdown('<div class="fas-section">Row counts by table</div>', unsafe_allow_html=True)
    counts = pd.DataFrame([{"table": k, "rows": len(v)} for k, v in tables.items()])
    st.dataframe(counts, use_container_width=True, height=300)

    st.markdown('<div class="fas-section">Limitations</div>', unsafe_allow_html=True)
    for lim in summary.get("limitations", []):
        st.markdown(f"- {lim}")

    st.markdown('<div class="fas-section">Missing data by field (actions)</div>',
                unsafe_allow_html=True)
    miss = (tables["actions"].isna().mean() * 100).round(2)
    miss = miss[miss > 0]
    if miss.empty:
        st.success("No missing values in core action fields.")
    else:
        st.dataframe(miss.rename("percent_missing").reset_index(), use_container_width=True)

    with st.expander("Full manifest"):
        st.json(manifest)


# --------------------------------------------------------------------------- #
# Shared rendering
# --------------------------------------------------------------------------- #

def _insight_cards(insights: pd.DataFrame, *, scope_entity=None, title="Insights"):
    st.markdown(f'<div class="fas-section">{title}</div>', unsafe_allow_html=True)
    if insights.empty:
        st.info("No insights generated.")
        return
    df = insights
    if scope_entity is not None:
        df = df[df["entity_id"] == scope_entity]
        if df.empty:
            st.caption("No entity-specific insights for this selection.")
            return
    for _, ins in df.head(5).iterrows():
        validated = str(ins["validation_status"]).startswith("validated")
        icon = "✅" if validated else "🔎"
        with st.expander(f"{icon} {ins['title']}"):
            st.markdown(f"**Claim.** {ins['claim']}")
            st.markdown(f"**Evidence.** {ins['evidence']}")
            st.markdown(f"**Method.** {ins['method']}")
            st.markdown(f"**Sample size.** {ins['sample_size']} · "
                        f"**Baseline.** {float(ins['baseline']):.4f} · "
                        f"**Status.** {ins['validation_status']}")
            st.markdown(f"**Caveats.** {ins['caveats']}")
            st.markdown(f"**Next look.** {ins['next_look']}")


def _interpretation_card(tables, team, tn):
    cen = tables["centralisation"]
    cen = cen[cen["team_id"] == team].sort_values("phase_index")
    if cen.empty:
        return
    early = cen.head(2)["centralisation"].mean()
    late = cen.tail(2)["centralisation"].mean()
    hub_early = cen.head(2)["hub_player"].dropna()
    hub_late = cen.tail(2)["hub_player"].dropna()
    drift = (not hub_early.empty and not hub_late.empty
             and hub_early.iloc[0] != hub_late.iloc[-1])
    msg = (f"<b>{tn.get(team)}</b> centralisation moved {late - early:+.3f} from the opening "
           "to the closing phases. ")
    msg += ("The dominant hub <b>changed players</b> — shape held but control shifted."
            if drift else "The dominant hub stayed stable.")
    st.markdown(f'<div class="fas-card fas-insight">{msg}</div>', unsafe_allow_html=True)


if __name__ == "__main__":  # pragma: no cover
    main()
