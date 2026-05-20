"""Static HTML report — the no-Streamlit fallback.

Builds a single self-contained HTML page (Plotly embedded inline, so it works
fully offline) summarizing the product: a representative match workspace, team
style, valuation scatter, matchup grid, validated insights, and data quality.
Written to ``data/processed/report/index.html``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fas.ui import charts
from fas.product.loader import load_product


def _fig_html(fig, *, first: bool) -> str:
    return fig.to_html(full_html=False,
                       include_plotlyjs=("inline" if first else False),
                       default_height="430px")


def build_report(data_root: str | Path = "data") -> Path:
    p = load_product(data_root)
    tables, summary, manifest = p.tables, p.summary, p.manifest
    teams, players, matches, actions = (tables["teams"], tables["players"],
                                        tables["matches"], tables["actions"])
    tn = dict(zip(teams["team_id"], teams["team_name"]))
    pn = dict(zip(players["player_id"], players["player_name"])) if not players.empty else {}

    blocks: list[str] = []
    first = True

    # representative match: the first one
    if not matches.empty:
        row = matches.iloc[0]
        mid = int(row["match_id"])
        team = int(row["home_team_id"])
        opp = int(row["away_team_id"])
        macts = actions[actions["match_id"] == mid]
        edges = tables["pass_network_edges"]
        ed = edges[(edges["match_id"] == mid) & (edges["team_id"] == team)
                   & (edges["window"] == "all")]
        cen = tables["centralisation"]
        cen = cen[(cen["match_id"] == mid) & (cen["team_id"] == team)]
        for fig in (
            charts.shot_map(macts, team_id=team, title=f"Shot map — {tn.get(team)}"),
            charts.xt_timeline(macts, team_id=team, opponent_id=opp, team_name=tn.get(team),
                               opp_name=tn.get(opp), title="Cumulative xT added"),
            charts.pass_network(ed, names=pn, title=f"Pass network — {tn.get(team)}"),
            charts.centralisation_trend(cen, names=pn,
                                        title=f"Centralisation by phase — {tn.get(team)}"),
        ):
            blocks.append(_section(_fig_html(fig, first=first)))
            first = False

    # team style + matchups + valuation
    if not teams.empty:
        team = int(teams.iloc[0]["team_id"])
        pc = tables["pass_clusters"]
        pc = pc[pc["team_id"] == team]
        pc_agg = (pc.groupby("label").agg(
            size=("size", "sum"), x_start=("x_start", "mean"), y_start=("y_start", "mean"),
            x_end=("x_end", "mean"), y_end=("y_end", "mean"),
            mean_length=("mean_length", "mean")).reset_index())
        blocks.append(_section(_fig_html(charts.pass_cluster_atlas(
            pc_agg, title=f"Pass cluster atlas — {tn.get(team)}"), first=False)))
        blocks.append(_section(_fig_html(charts.style_distance_matrix(
            tables["matchup_artifacts"], teams, title="Style distance matrix"), first=False)))
        blocks.append(_section(_fig_html(charts.xt_surface_heatmap(
            tables["xt_surface"], title="League xT surface"), first=False)))

    blocks.append(_section(_fig_html(charts.value_vs_market(
        players, title="Fair value vs market value"), first=False)))

    # matchup grid for the first pair
    mm = tables["matchup_artifacts"]
    if not mm.empty:
        r = mm.iloc[0]
        sc = tables["scorelines"]
        sc = sc[(sc["entity_i"] == r["entity_i"]) & (sc["entity_j"] == r["entity_j"])]
        blocks.append(_section(_fig_html(charts.scoreline_grid(
            sc, title=f"Scoreline — {tn.get(r['entity_i'])} vs {tn.get(r['entity_j'])}"),
            first=False)))

    out_dir = Path(data_root) / "processed" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    html = _page(summary, manifest, tables["insights"], "\n".join(blocks))
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _section(inner: str) -> str:
    return f'<section class="card">{inner}</section>'


def _insights_html(insights: pd.DataFrame) -> str:
    if insights.empty:
        return "<p>No insights generated.</p>"
    items = []
    for _, i in insights.head(12).iterrows():
        validated = str(i["validation_status"]).startswith("validated")
        tag = "validated" if validated else "exploratory"
        items.append(
            f'<div class="insight {tag}"><h4>{"✅" if validated else "🔎"} {i["title"]}</h4>'
            f'<p><b>Claim.</b> {i["claim"]}</p>'
            f'<p><b>Evidence.</b> {i["evidence"]}</p>'
            f'<p class="meta"><b>Method:</b> {i["method"]} · '
            f'<b>n=</b>{i["sample_size"]} · <b>baseline</b> {float(i["baseline"]):.4f} · '
            f'<b>{i["validation_status"]}</b></p>'
            f'<p class="meta"><b>Caveat.</b> {i["caveats"]} <b>Next look.</b> {i["next_look"]}</p>'
            "</div>")
    return "\n".join(items)


def _page(summary: dict, manifest: dict, insights: pd.DataFrame, body: str) -> str:
    mode = manifest.get("data_mode")
    badge = ("⚠️ Deterministic synthetic demo data (event-only)" if mode == "synthetic"
             else f"Data: {manifest.get('source')}")
    lims = "".join(f"<li>{x}</li>" for x in summary.get("limitations", []))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>FAS — Football Analytics Report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
        background:#0f1116; color:#e6e6e6; }}
 header {{ padding:18px 28px; background:#161a22; border-bottom:1px solid #262b36; }}
 h1 {{ margin:0; font-size:22px; }} .sub {{ color:#9aa4b2; font-size:13px; }}
 .badge {{ display:inline-block; margin-top:8px; padding:4px 10px; border-radius:6px;
          background:#3a2f12; color:#ffcf6b; font-size:12px; }}
 main {{ padding:20px 28px; max-width:1180px; margin:auto; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }}
 .card {{ background:#161a22; border:1px solid #262b36; border-radius:10px; padding:8px; }}
 .kpis {{ display:flex; gap:24px; flex-wrap:wrap; margin:14px 0; }}
 .kpi b {{ font-size:22px; }} .kpi span {{ color:#9aa4b2; font-size:12px; display:block; }}
 .insight {{ border-left:3px solid #2d6cdf; padding:8px 12px; margin:10px 0; background:#11151c; }}
 .insight.exploratory {{ border-color:#b9892f; }}
 .insight h4 {{ margin:2px 0; font-size:15px; }} .meta {{ color:#9aa4b2; font-size:12px; }}
 h2 {{ border-bottom:1px solid #262b36; padding-bottom:6px; margin-top:34px; }}
</style></head><body>
<header><h1>⚽ FAS — Football Analytics Report</h1>
<div class="sub">{manifest.get('competition')} · {manifest.get('season')} · generated {manifest.get('generated_at')}</div>
<div class="badge">{badge}</div></header>
<main>
<div class="kpis">
 <div class="kpi"><b>{summary.get('n_matches')}</b><span>matches</span></div>
 <div class="kpi"><b>{summary.get('n_teams')}</b><span>teams</span></div>
 <div class="kpi"><b>{summary.get('n_players')}</b><span>players</span></div>
 <div class="kpi"><b>{summary.get('n_actions')}</b><span>actions</span></div>
 <div class="kpi"><b>{summary.get('n_validated_insights')}/{summary.get('n_insights')}</b>
   <span>validated / total insights</span></div>
 <div class="kpi"><b>{summary.get('n_pass_clusters')}</b><span>pass clusters</span></div>
</div>
<h2>Charts</h2>
<div class="grid">{body}</div>
<h2>Validated &amp; exploratory insights</h2>
{_insights_html(insights)}
<h2>Data quality &amp; limitations</h2>
<ul>{lims}</ul>
<p class="meta">Every chart is tied to a specific match/team/competition context and
labelled with its data mode. Synthetic data is event-only: no 360 freeze frames,
no tracking-data precision is implied.</p>
</main></body></html>"""
