"""Product artifact build orchestrator.

``product_build`` resolves the data spine, runs the six-layer materializer, and
writes a complete, self-describing artifact set under ``data/processed/``. It is
idempotent: re-running overwrites the artifacts deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fas.product import ARTIFACT_FILES
from fas.product.artifacts import (
    build_insights,
    build_match_layer,
    build_matchup_layer,
    build_player_layer,
    build_squad,
    enrich_actions,
)
from fas.product.ingest import build_spine


def product_build(
    *,
    data_path: str | None = None,
    data_root: str | Path = "data",
    allow_download: bool = True,
    seed: int = 7,
    sb_competition: str | None = None,
    sb_season: str | None = None,
    sb_team: str | None = None,
    sb_max_matches: int = 60,
    sb_per_season: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build and persist all product artifacts. Returns the summary dict.

    By default this fetches a broad sample of real StatsBomb Open Data spanning
    every available competition, season, and team. Narrow with ``sb_competition``
    / ``sb_season`` / ``sb_team``; raise ``sb_max_matches`` (0 = all) for fuller
    coverage. Pass ``allow_download=False`` to force the offline deterministic
    synthetic fallback (used by CI and the ``--no-download`` flag).
    """
    out_dir = Path(data_root) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("[1/7] resolving data spine ...")
    spine = build_spine(data_path=data_path, data_root=data_root,
                        allow_download=allow_download, seed=seed,
                        sb_competition=sb_competition, sb_season=sb_season,
                        sb_team=sb_team, sb_max_matches=sb_max_matches,
                        sb_per_season=sb_per_season)
    log(f"      data mode: {spine.manifest['data_mode']} "
        f"({spine.manifest['source']}), {spine.manifest['row_counts']}")

    log("[2/7] enriching actions (xT, action value, zones, phases) ...")
    actions, xt = enrich_actions(spine)

    log("[3/7] match layer (networks, centralisation, clusters, zones) ...")
    match_tables = build_match_layer(spine, actions, xt)

    log("[4/7] player layer (roles, PVS, form, development) ...")
    players_df, player_artifacts, pvs, pos_for_pvs = build_player_layer(spine, actions)

    log("[5/7] matchup layer (Dixon-Coles, paired comparison, style clash) ...")
    matchup_df, scorelines_df = build_matchup_layer(spine, actions, players_df)

    log("[6/7] insight engine (FDR-controlled + exploratory cards) ...")
    insights_df = build_insights(spine, actions, match_tables["centralisation"], players_df)

    log("[7/7] team layer + squad optimization + xT surface ...")
    team_artifacts = _team_artifacts(spine, match_tables, actions)
    squad = build_squad(spine, players_df, pvs, pos_for_pvs)
    xt_surface = _xt_surface_frame(xt, spine)

    # --- write everything ---
    tables: dict[str, pd.DataFrame] = {
        "actions": actions,
        "matches": spine.matches,
        "teams": spine.teams,
        "players": players_df,
        "match_artifacts": match_tables["match_artifacts"],
        "player_artifacts": player_artifacts,
        "team_artifacts": team_artifacts,
        "matchup_artifacts": matchup_df,
        "insights": insights_df,
        "pass_network_edges": match_tables["pass_network_edges"],
        "pass_clusters": match_tables["pass_clusters"],
        "centralisation": match_tables["centralisation"],
        "formations": match_tables["formations"],
        "zone_flow": match_tables["zone_flow"],
        "xt_surface": xt_surface,
        "scorelines": scorelines_df,
    }
    for name, df in tables.items():
        _write_parquet(df, out_dir / f"{name}.parquet")

    summary = _summary(spine, tables, squad)
    (out_dir / "product_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(spine.manifest, indent=2, default=str), encoding="utf-8")

    missing = [f for f in ARTIFACT_FILES if not (out_dir / f).exists()]
    if missing:  # pragma: no cover
        raise RuntimeError(f"artifact build incomplete, missing: {missing}")

    log(f"\nWrote {len(ARTIFACT_FILES)} artifacts to {out_dir.resolve()}")
    log(f"data mode: {summary['data_mode']} | matches: {summary['n_matches']} | "
        f"players: {summary['n_players']} | insights: {summary['n_insights']}")
    return summary


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        df = pd.DataFrame({"_empty": []})
    try:
        df.to_parquet(path, index=False)
    except Exception:  # pragma: no cover - parquet engine fallback
        df.to_parquet(path, index=False, engine="pyarrow")


def _team_artifacts(spine, match_tables, actions) -> pd.DataFrame:
    """Team-season aggregates from the per-match metric table."""
    ma = match_tables["match_artifacts"]
    if ma.empty:
        return pd.DataFrame()
    rows = []
    for (tid,), grp in ma.groupby(["team_id"]):
        for metric in ("network_entropy", "centralisation", "buildup_xt_reward",
                       "xt_added_total", "shots", "goals"):
            sub = grp[grp["metric_name"] == metric]
            if sub.empty:
                continue
            rows.append({
                "team_id": int(tid),
                "competition": spine.manifest.get("competition"),
                "season": spine.manifest.get("season"),
                "data_source": spine.manifest.get("data_mode"),
                "is_synthetic": spine.manifest.get("is_synthetic"),
                "metric_name": metric,
                "mean": float(sub["metric_value"].mean()),
                "std": float(sub["metric_value"].std(ddof=0)),
                "n_matches": int(sub["match_id"].nunique()),
                "model_name": sub["model_name"].iloc[0],
                "model_version": "fas-0.2",
                "limitations": "; ".join(spine.manifest.get("limitations", [])) or "none",
            })
    return pd.DataFrame(rows)


def _xt_surface_frame(xt, spine) -> pd.DataFrame:
    rows = []
    for i in range(xt.n_x):
        for j in range(xt.n_y):
            rows.append({
                "cell_x": i, "cell_y": j, "n_x": xt.n_x, "n_y": xt.n_y,
                "xt": float(xt.grid[i, j]),
                "x_center": (i + 0.5) / xt.n_x * 120.0,
                "y_center": (j + 0.5) / xt.n_y * 80.0,
                "model_name": "xt_value_iteration", "model_version": "fas-0.2",
                "data_source": spine.manifest.get("data_mode"),
                "is_synthetic": spine.manifest.get("is_synthetic"),
            })
    return pd.DataFrame(rows)


def _summary(spine, tables, squad) -> dict:
    m = spine.manifest
    ins = tables["insights"]
    return {
        "data_mode": m["data_mode"],
        "source": m["source"],
        "is_synthetic": m["is_synthetic"],
        "competition": m["competition"],
        "season": m["season"],
        "seed": m.get("seed"),
        "generated_at": m["generated_at"],
        "n_matches": int(len(tables["matches"])),
        "n_teams": int(len(tables["teams"])),
        "n_players": int(len(tables["players"])),
        "n_actions": int(len(tables["actions"])),
        "n_insights": int(len(ins)),
        "n_validated_insights": int((ins.get("validation_status", pd.Series(dtype=str))
                                     .astype(str).str.startswith("validated")).sum())
        if not ins.empty else 0,
        "n_pass_clusters": int(len(tables["pass_clusters"])),
        "has_360": m["has_360"],
        "limitations": m["limitations"],
        "squad_optimization": squad,
        "artifact_files": list(ARTIFACT_FILES),
    }
