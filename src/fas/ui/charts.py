"""Football-native Plotly charts shared by the Streamlit app and HTML report.

Every builder takes already-loaded artifact frames (no model runs) and returns
a Plotly figure. The visual encoding contract is consistent throughout:

- pitch coordinates are StatsBomb ``[0,120] x [0,80]``, attacking toward +x;
- the selected team uses :data:`TEAM_COLOR`, the opponent :data:`OPP_COLOR`;
- xT / value use a single sequential scale (``Plasma``);
- failed actions are drawn at lower opacity;
- sample size and data mode are surfaced in titles / subtitles by callers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Dark, professional default for every figure (Streamlit + HTML report).
pio.templates.default = "plotly_dark"
_TRANSPARENT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

TEAM_COLOR = "#1f77b4"
OPP_COLOR = "#9e9e9e"
ACCENT = "#d62728"
SEQ = "Plasma"
PITCH_L, PITCH_W = 120.0, 80.0


# --------------------------------------------------------------------------- #
# Pitch scaffold
# --------------------------------------------------------------------------- #

def _pitch_shapes() -> list[dict]:
    line = dict(color="rgba(120,120,120,0.6)", width=1)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=PITCH_L, y1=PITCH_W, line=line),
        dict(type="line", x0=60, y0=0, x1=60, y1=PITCH_W, line=line),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=line),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=line),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=line),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=line),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=line),
    ]
    return shapes


def _pitch_fig(height: int = 470) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        shapes=_pitch_shapes(),
        xaxis=dict(range=[-2, 122], visible=False, constrain="domain"),
        yaxis=dict(range=[-2, 82], visible=False, scaleanchor="x", scaleratio=1),
        height=height, margin=dict(l=10, r=10, t=46, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,28,38,0.35)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return fig


def _empty(msg: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#888"))
    fig.update_layout(height=height, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Layer 1 — events
# --------------------------------------------------------------------------- #

def shot_map(actions: pd.DataFrame, *, team_id: int, title: str) -> go.Figure:
    shots = actions[(actions["action_type"] == "shot") & (actions["team_id"] == team_id)]
    fig = _pitch_fig()
    if shots.empty:
        return _empty("No shots in this selection.")
    goals = shots[shots["outcome"]]
    miss = shots[~shots["outcome"]]
    for sub, name, sym, op in ((miss, "Shot", "circle", 0.5), (goals, "Goal", "star", 1.0)):
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["x_start"], y=sub["y_start"], mode="markers", name=name,
            marker=dict(size=13 if name == "Goal" else 10, symbol=sym,
                        color=ACCENT if name == "Goal" else TEAM_COLOR, opacity=op,
                        line=dict(width=1, color="white")),
            customdata=np.c_[sub["minute"].round(1), sub["action_value"].round(3)],
            hovertemplate="min %{customdata[0]}<br>xG proxy %{customdata[1]}<extra>"
                          + name + "</extra>"))
    fig.update_layout(title=title)
    return fig


def xt_timeline(actions: pd.DataFrame, *, team_id: int, opponent_id: int,
                team_name: str, opp_name: str, title: str) -> go.Figure:
    fig = go.Figure()
    df = actions[actions["match_id"].notna()]
    for tid, name, color in ((team_id, team_name, TEAM_COLOR), (opponent_id, opp_name, OPP_COLOR)):
        sub = df[df["team_id"] == tid].sort_values("minute")
        if sub.empty:
            continue
        cum = sub["xt_added"].cumsum()
        fig.add_trace(go.Scatter(x=sub["minute"], y=cum, mode="lines", name=name,
                                 line=dict(color=color, width=2)))
    fig.update_layout(title=title, height=320, xaxis_title="minute",
                      yaxis_title="cumulative xT added",
                      margin=dict(l=10, r=10, t=46, b=10),
                      legend=dict(orientation="h", y=1.02))
    return fig


# --------------------------------------------------------------------------- #
# Layer 2/3 — space, networks, control
# --------------------------------------------------------------------------- #

def heatmap_actions(actions: pd.DataFrame, *, team_id: int, title: str) -> go.Figure:
    df = actions[actions["team_id"] == team_id]
    fig = _pitch_fig()
    if df.empty:
        return _empty("No actions in this selection.")
    fig.add_trace(go.Histogram2dContour(
        x=df["x_start"], y=df["y_start"], colorscale=SEQ, showscale=True,
        ncontours=14, opacity=0.85, colorbar=dict(title="density")))
    fig.update_layout(title=title, showlegend=False)
    return fig


def pass_network(edges: pd.DataFrame, *, title: str, names: dict | None = None) -> go.Figure:
    fig = _pitch_fig()
    if edges.empty:
        return _empty("No pass-network edges for this window.")
    wmax = edges["weight"].max()
    for r in edges.itertuples(index=False):
        if not np.isfinite(r.passer_x) or not np.isfinite(r.receiver_x):
            continue
        fig.add_trace(go.Scatter(
            x=[r.passer_x, r.receiver_x], y=[r.passer_y, r.receiver_y],
            mode="lines", line=dict(width=0.5 + 4 * r.weight / wmax,
                                    color="rgba(31,119,180,0.35)"),
            hoverinfo="skip", showlegend=False))
    nodes = _network_nodes(edges)
    fig.add_trace(go.Scatter(
        x=nodes["x"], y=nodes["y"], mode="markers+text",
        marker=dict(size=8 + 26 * nodes["involvement"] / nodes["involvement"].max(),
                    color=nodes["involvement"], colorscale=SEQ, showscale=True,
                    colorbar=dict(title="touches"), line=dict(width=1, color="white")),
        text=[(names or {}).get(int(p), str(int(p))).split()[-1] for p in nodes["player"]],
        textposition="top center", textfont=dict(size=9),
        hovertemplate="%{text}<br>involvement %{marker.color}<extra></extra>",
        showlegend=False))
    fig.update_layout(title=title)
    return fig


def _network_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges[["passer", "passer_x", "passer_y", "weight"]].rename(
        columns={"passer": "player", "passer_x": "x", "passer_y": "y"})
    inc = edges[["receiver", "receiver_x", "receiver_y", "weight"]].rename(
        columns={"receiver": "player", "receiver_x": "x", "receiver_y": "y"})
    both = pd.concat([out, inc])
    nodes = both.groupby("player").agg(x=("x", "mean"), y=("y", "mean"),
                                       involvement=("weight", "sum")).reset_index()
    return nodes


def zone_flow_map(zone_flow: pd.DataFrame, *, title: str) -> go.Figure:
    fig = _pitch_fig()
    if zone_flow.empty:
        return _empty("No zone-flow corridors for this team.")
    n_rows = 3
    n_bands = 6

    def zc(z):
        band, row = z // n_rows, z % n_rows
        return (band + 0.5) / n_bands * PITCH_L, (row + 0.5) / n_rows * PITCH_W

    fmax = max(zone_flow["flow_used"].max(), 1)
    for r in zone_flow.itertuples(index=False):
        x0, y0 = zc(int(r.z0))
        x1, y1 = zc(int(r.z1))
        used = r.flow_used > 0
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines+markers",
            line=dict(width=1 + 6 * r.flow_used / fmax if used else 1,
                      color=ACCENT if used else "rgba(120,120,120,0.3)"),
            marker=dict(size=4), hoverinfo="skip", showlegend=False))
    fig.update_layout(title=title)
    return fig


def xt_surface_heatmap(xt_surface: pd.DataFrame, *, title: str) -> go.Figure:
    fig = _pitch_fig()
    if xt_surface.empty:
        return _empty("xT surface unavailable.")
    n_x = int(xt_surface["n_x"].iloc[0])
    n_y = int(xt_surface["n_y"].iloc[0])
    grid = np.zeros((n_y, n_x))
    for r in xt_surface.itertuples(index=False):
        grid[int(r.cell_y), int(r.cell_x)] = r.xt
    fig.add_trace(go.Heatmap(
        z=grid, x=np.linspace(0, PITCH_L, n_x), y=np.linspace(0, PITCH_W, n_y),
        colorscale=SEQ, opacity=0.8, colorbar=dict(title="xT")))
    fig.update_layout(title=title, showlegend=False)
    return fig


def pass_cluster_atlas(clusters: pd.DataFrame, *, title: str) -> go.Figure:
    fig = _pitch_fig()
    if clusters.empty:
        return _empty("No pass clusters discovered.")
    smax = clusters["size"].max()
    for r in clusters.itertuples(index=False):
        fig.add_trace(go.Scatter(
            x=[r.x_start, r.x_end], y=[r.y_start, r.y_end], mode="lines+markers",
            line=dict(width=1 + 6 * r.size / smax),
            marker=dict(size=[6, 12]),
            name=f"{r.label} (n={int(r.size)})",
            hovertemplate=f"{r.label}<br>size {int(r.size)}<br>"
                          f"len {r.mean_length:.1f}<extra></extra>"))
    fig.update_layout(title=title)
    return fig


# --------------------------------------------------------------------------- #
# Trends and matrices
# --------------------------------------------------------------------------- #

def centralisation_trend(central: pd.DataFrame, *, title: str,
                         names: dict | None = None) -> go.Figure:
    fig = go.Figure()
    if central.empty:
        return _empty("No centralisation data.")
    c = central.sort_values("phase_index")
    fig.add_trace(go.Scatter(x=c["phase"], y=c["centralisation"], mode="lines+markers",
                             name="centralisation", line=dict(color=TEAM_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=c["phase"], y=c["entropy"] / max(c["entropy"].max(), 1e-9),
                             mode="lines", name="entropy (norm)",
                             line=dict(color=OPP_COLOR, dash="dot")))
    # annotate hub identity changes
    for r in c.itertuples(index=False):
        if r.hub_player is not None and np.isfinite(r.hub_player):
            label = (names or {}).get(int(r.hub_player), str(int(r.hub_player)))
            fig.add_annotation(x=r.phase, y=r.centralisation, text=label.split()[-1],
                               showarrow=False, yshift=12, font=dict(size=8, color="#555"))
    fig.update_layout(title=title, height=340, yaxis_title="index",
                      xaxis_title="phase", legend=dict(orientation="h", y=1.02),
                      margin=dict(l=10, r=10, t=46, b=10))
    return fig


def style_distance_matrix(matchups: pd.DataFrame, teams: pd.DataFrame, *, title: str) -> go.Figure:
    if matchups.empty:
        return _empty("No style distances.")
    ids = sorted(set(matchups["entity_i"]) | set(matchups["entity_j"]))
    name = dict(zip(teams["team_id"], teams["team_name"]))
    mat = np.zeros((len(ids), len(ids)))
    idx = {t: k for k, t in enumerate(ids)}
    for r in matchups.itertuples(index=False):
        mat[idx[r.entity_i], idx[r.entity_j]] = r.style_distance
    labels = [name.get(t, str(t)) for t in ids]
    fig = go.Figure(go.Heatmap(z=mat, x=labels, y=labels, colorscale=SEQ,
                               colorbar=dict(title="Fisher-Rao")))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=46, b=10))
    return fig


def outcome_bars(probs: dict, *, home: str, away: str, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[f"{home} win", "draw", f"{away} win"],
        y=[probs["home"], probs["draw"], probs["away"]],
        marker_color=[TEAM_COLOR, OPP_COLOR, ACCENT],
        text=[f"{p:.0%}" for p in (probs["home"], probs["draw"], probs["away"])],
        textposition="auto"))
    fig.update_layout(title=title, height=300, yaxis_title="probability",
                      yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=46, b=10))
    return fig


def scoreline_grid(scorelines: pd.DataFrame, *, title: str) -> go.Figure:
    if scorelines.empty:
        return _empty("No scoreline distribution.")
    piv = scorelines.pivot_table(index="home_goals", columns="away_goals",
                                 values="prob", aggfunc="sum").fillna(0.0)
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                               colorscale=SEQ, colorbar=dict(title="P")))
    fig.update_layout(title=title, height=360, xaxis_title="away goals",
                      yaxis_title="home goals", margin=dict(l=10, r=10, t=46, b=10))
    return fig


def value_vs_market(players: pd.DataFrame, *, title: str) -> go.Figure:
    if players.empty:
        return _empty("No player valuations.")
    fig = go.Figure()
    for rg, sub in players.groupby("role_group"):
        fig.add_trace(go.Scatter(
            x=sub["market_value_eur_m"], y=sub["fair_value_eur_m"], mode="markers",
            name=str(rg), text=sub["player_name"],
            marker=dict(size=8 + 16 * sub["pvs"], opacity=0.75),
            hovertemplate="%{text}<br>market €%{x}M<br>fair €%{y}M<extra></extra>"))
    lim = max(players["market_value_eur_m"].max(), players["fair_value_eur_m"].max()) * 1.05
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="fair = market",
                             line=dict(color="#aaa", dash="dash"), hoverinfo="skip"))
    fig.update_layout(title=title, height=420, xaxis_title="market value (€M)",
                      yaxis_title="model fair value (€M)",
                      legend=dict(orientation="h", y=1.02), margin=dict(l=10, r=10, t=46, b=10))
    return fig


def role_profile(players: pd.DataFrame, player_row: pd.Series, *, title: str) -> go.Figure:
    metrics = ["pvs_pct", "xt_added_90_pct", "action_value_pct"]
    labels = ["PVS", "xT added /90", "action value"]
    vals = [float(player_row.get(m, 0.0)) for m in metrics]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=TEAM_COLOR,
                           text=[f"{v:.0f}%" for v in vals], textposition="auto"))
    fig.update_layout(title=title, height=320, yaxis=dict(range=[0, 100],
                      title="percentile within role group"),
                      margin=dict(l=10, r=10, t=46, b=10))
    return fig
