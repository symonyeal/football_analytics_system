"""Local-data-first ingestion: the product data spine.

Data priority (highest first):

1. a user-provided canonical actions file via ``data_path``;
2. an existing ``data/processed/actions.parquet``;
3. a downloaded / cached StatsBomb Open Data sample (only if ``allow_download``);
4. the deterministic synthetic fallback.

Whatever the source, the spine normalizes everything into the same frames —
``actions``, ``matches``, ``teams``, ``players``, ``results`` — plus a manifest
recording the source, limitations, and (for synthetic data) the seed. When the
source is a bare canonical actions file we *derive* minimal metadata so the
product still has teams, players, and matches to work with.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fas.data.schema import COLUMNS, validate_actions
from fas.product.synthetic import SyntheticLeague, generate_league


@dataclass(slots=True)
class DataSpine:
    """Normalized multi-table view consumed by the artifact builder."""

    actions: pd.DataFrame
    matches: pd.DataFrame
    teams: pd.DataFrame
    players: pd.DataFrame
    results: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)


def _discover_local(root: Path) -> Path | None:
    preferred = [
        root / "processed" / "actions.parquet",
        root / "processed" / "actions.csv",
        root / "raw" / "actions.parquet",
        root / "raw" / "actions.csv",
    ]
    for p in preferred:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _is_own_artifact(source_path: Path, root: Path) -> bool:
    """True if ``source_path`` is the ``actions.parquet`` we generated ourselves.

    Detected via the sibling ``manifest.json`` (only ``product_build`` writes
    that file). Genuine user data should be passed via ``data_path`` or placed
    without our manifest alongside it.
    """
    try:
        if source_path.resolve() != (root / "processed" / "actions.parquet").resolve():
            return False
        return (root / "processed" / "manifest.json").exists()
    except Exception:
        return False


def _load_canonical(path: Path) -> pd.DataFrame:
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


def _infer_role_group(mean_x: float) -> str:
    if mean_x < 35:
        return "Defender"
    if mean_x < 70:
        return "Midfielder"
    return "Forward"


def _derive_metadata(actions: pd.DataFrame, *, competition: str, season: str) -> SyntheticLeague:
    """Build minimal matches/teams/players/results frames from bare actions."""
    teams = sorted(int(t) for t in actions["team_id"].unique())
    team_rows = [{
        "team_id": t, "team_name": f"Team {t}",
        "competition": competition, "season": season, "formation": "unknown",
    } for t in teams]

    avg_x = actions.groupby("player_id")["x_start"].mean()
    team_of = actions.groupby("player_id")["team_id"].agg(lambda s: int(s.mode().iloc[0]))
    player_rows = []
    for pid in sorted(int(p) for p in actions["player_id"].unique()):
        mx = float(avg_x.get(pid, 50.0))
        player_rows.append({
            "player_id": pid, "player_name": f"Player {pid}",
            "team_id": int(team_of.get(pid, teams[0])),
            "position": "UNK", "role_group": _infer_role_group(mx),
            "is_starter": True, "shirt": 0,
        })

    match_rows, result_rows = [], []
    for mid, grp in actions.groupby("match_id"):
        ts = sorted(int(t) for t in grp["team_id"].unique())
        home, away = (ts[0], ts[1]) if len(ts) > 1 else (ts[0], ts[0])
        hg = int(((grp["team_id"] == home) & (grp["action_type"] == "shot")
                  & grp["outcome"].astype(bool)).sum())
        ag = int(((grp["team_id"] == away) & (grp["action_type"] == "shot")
                  & grp["outcome"].astype(bool)).sum())
        match_rows.append({
            "match_id": int(mid), "competition": competition, "season": season,
            "date": None, "home_team_id": home, "away_team_id": away,
            "home_team": f"Team {home}", "away_team": f"Team {away}",
            "home_goals": hg, "away_goals": ag,
            "has_360": False, "data_source": "local", "is_synthetic": False,
        })
        result_rows.append({"match_id": int(mid), "home_team": home,
                            "away_team": away, "home_goals": hg, "away_goals": ag})

    return SyntheticLeague(
        actions=actions, matches=pd.DataFrame(match_rows),
        teams=pd.DataFrame(team_rows), players=pd.DataFrame(player_rows),
        results=pd.DataFrame(result_rows), seed=-1,
    )


def build_spine(
    *,
    data_path: str | Path | None = None,
    data_root: str | Path = "data",
    allow_download: bool = False,
    seed: int = 7,
    competition: str = "FAS Synthetic League",
    season: str = "2024/2025",
    sb_competition: str | None = None,
    sb_season: str | None = None,
    sb_team: str | None = None,
    sb_max_matches: int = 60,
    sb_per_season: int = 4,
) -> DataSpine:
    """Resolve the data source by priority and return a normalized spine."""
    root = Path(data_root)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    limitations: list[str] = []

    source_path: Path | None = None
    if data_path is not None:
        source_path = Path(data_path)
    else:
        source_path = _discover_local(root)
        # Don't consume our own previously-written artifacts as if they were
        # user-provided local data — re-ingest (real download or synthetic)
        # instead so rich metadata (names, formations, positions) is preserved.
        if source_path is not None and _is_own_artifact(source_path, root):
            source_path = None

    if source_path is not None and source_path.exists():
        actions = _load_canonical(source_path)
        league = _derive_metadata(actions, competition=competition, season=season)
        data_mode, source_desc, is_synth = "local", str(source_path), False
        limitations += [
            "metadata (lineups, formations, player roles) derived from event "
            "locations only — positions are approximate.",
            "no 360 freeze frames: spatial pressure context is event-only.",
        ]
    elif allow_download:
        spine = load_statsbomb(competition_name=sb_competition, season_name=sb_season,
                               team_name=sb_team, max_matches=sb_max_matches,
                               per_season=sb_per_season)
        if spine is not None:
            return spine
        league = generate_league(seed=seed, competition=competition, season=season)
        actions = league.actions
        data_mode, source_desc, is_synth = "synthetic", "synthetic fallback", True
        limitations.append("StatsBomb download unavailable (offline?); used synthetic fallback.")
    else:
        league = generate_league(seed=seed, competition=competition, season=season)
        actions = league.actions
        data_mode, source_desc, is_synth = "synthetic", "synthetic fallback", True
        limitations += [
            "deterministic synthetic demo data — not real matches.",
            "no 360 freeze frames: all spatial context is event-only.",
        ]

    has_360 = bool(league.matches.get("has_360", pd.Series([False])).any()) \
        if "has_360" in league.matches else False

    manifest = {
        "generated_at": now,
        "data_mode": data_mode,
        "source": source_desc,
        "is_synthetic": is_synth,
        "competition": competition,
        "season": season,
        "seed": int(league.seed) if is_synth else None,
        "match_ids": [int(m) for m in sorted(actions["match_id"].unique())],
        "row_counts": {
            "actions": int(len(actions)),
            "matches": int(len(league.matches)),
            "teams": int(len(league.teams)),
            "players": int(len(league.players)),
        },
        "has_360": has_360,
        "canonical_columns": list(COLUMNS),
        "limitations": limitations,
    }

    return DataSpine(
        actions=actions,
        matches=league.matches,
        teams=league.teams,
        players=league.players,
        results=league.results,
        manifest=manifest,
    )


def _sb_role_group(position: str | None) -> str:
    p = (position or "").lower()
    if "goalkeeper" in p:
        return "Goalkeeper"
    if "back" in p:
        return "Defender"
    if "midfield" in p:
        return "Midfielder"
    if any(k in p for k in ("wing", "forward", "striker", "center forward")):
        return "Forward"
    return "Midfielder"


def _sb_formation_str(formation) -> str:
    try:
        return "-".join(str(int(formation)))
    except Exception:
        return "unknown"


def _select_matches(sb, competition_name, season_name, team_name, per_season, max_matches):
    """Resolve the matches to ingest, sampled for breadth across the corpus.

    With ``competition_name=None`` and ``season_name=None`` this spans *every*
    competition and season in StatsBomb Open Data. To represent all leagues and
    eras even under a match cap, we take up to ``per_season`` newest matches from
    each (competition, season) cohort and then round-robin across cohorts until
    ``max_matches`` is reached (``0`` = unlimited).

    Returns a list of ``(competition_name, season_name, match_dict)``.
    """
    comps = sb.competitions()
    rows = comps
    if competition_name not in (None, "", "all"):
        rows = rows[rows["competition_name"] == competition_name]
    if season_name not in (None, "", "all"):
        rows = rows[rows["season_name"] == season_name]
    if rows.empty:
        return []
    # newest competition-seasons first
    rows = rows.sort_values(["competition_name", "season_name"], ascending=[True, False])

    cohorts: list[list[tuple[str, str, dict]]] = []
    for _, cr in rows.iterrows():
        try:
            matches = sb.matches(competition_id=int(cr["competition_id"]),
                                 season_id=int(cr["season_id"]))
        except Exception:
            continue
        if team_name:
            matches = matches[(matches["home_team"] == team_name)
                              | (matches["away_team"] == team_name)]
        if matches.empty:
            continue
        matches = matches.sort_values("match_date", ascending=False)
        if per_season and per_season > 0:
            matches = matches.head(per_season)
        cohort = [(str(cr["competition_name"]), str(cr["season_name"]), m.to_dict())
                  for _, m in matches.iterrows()]
        if cohort:
            cohorts.append(cohort)

    # round-robin across cohorts for league/era breadth under the cap
    selected: list[tuple[str, str, dict]] = []
    i = 0
    while cohorts and (not max_matches or max_matches <= 0 or len(selected) < max_matches):
        progressed = False
        for cohort in cohorts:
            if i < len(cohort):
                selected.append(cohort[i])
                progressed = True
                if max_matches and max_matches > 0 and len(selected) >= max_matches:
                    break
        if not progressed:
            break
        i += 1
    return selected


def load_statsbomb(  # pragma: no cover - network
    *,
    competition_name: str | None = None,
    season_name: str | None = None,
    team_name: str | None = None,
    max_matches: int = 60,
    per_season: int = 4,
) -> DataSpine | None:
    """Ingest real StatsBomb Open Data spanning all leagues, eras, and teams.

    Defaults pull a broad sample across *every* competition and season in the
    open-data corpus (no team filter), round-robined for league/era breadth and
    capped at ``max_matches`` (``0`` = unlimited; slow). Narrow with
    ``competition_name`` / ``season_name`` / ``team_name`` when wanted. Real
    names, modal positions, and Starting-XI formations are preserved. Returns
    ``None`` on any failure so the caller can fall back to synthetic data.
    """
    try:
        from statsbombpy import sb

        from fas.data.statsbomb import events_to_actions

        selected = _select_matches(sb, competition_name, season_name, team_name,
                                   per_season, max_matches)
        if not selected:
            return None

        frames, match_rows, result_rows = [], [], []
        name_map: dict[int, str] = {}
        team_name_map: dict[int, str] = {}
        pos_map: dict[int, str] = {}
        team_of_player: dict[int, int] = {}
        formation_map: dict[int, str] = {}
        seasons_seen: set[str] = set()
        comps_seen: set[str] = set()

        for comp_lbl, season_lbl, m in selected:
            mid = int(m["match_id"])
            try:
                events = sb.events(match_id=mid)
            except Exception:
                continue
            acts = events_to_actions(events, match_id=mid)
            if acts.empty:
                continue
            frames.append(acts)
            seasons_seen.add(season_lbl)
            comps_seen.add(comp_lbl)

            for tname, tid in events[["team", "team_id"]].dropna().drop_duplicates().itertuples(index=False):
                team_name_map[int(tid)] = str(tname)
            pl = events.dropna(subset=["player_id"])
            for pname, pid in pl[["player", "player_id"]].drop_duplicates().itertuples(index=False):
                name_map[int(pid)] = str(pname)
            for pid, tid in pl[["player_id", "team_id"]].drop_duplicates().itertuples(index=False):
                team_of_player[int(pid)] = int(tid)
            if "position" in pl:
                modal = pl.dropna(subset=["position"]).groupby("player_id")["position"].agg(
                    lambda s: s.mode().iloc[0] if len(s.mode()) else None)
                for pid, pos in modal.items():
                    pos_map.setdefault(int(pid), pos)
            sxi = events[events["type"] == "Starting XI"]
            for _, e in sxi.iterrows():
                tac = e.get("tactics")
                tid = e.get("team_id")
                if isinstance(tac, dict) and tid is not None:
                    formation_map.setdefault(int(tid), _sb_formation_str(tac.get("formation")))

            home_id = int(events[events["team"] == m["home_team"]]["team_id"].iloc[0])
            away_id = int(events[events["team"] == m["away_team"]]["team_id"].iloc[0])
            match_rows.append({
                "match_id": mid, "competition": comp_lbl, "season": season_lbl,
                "date": str(m.get("match_date")),
                "home_team_id": home_id, "away_team_id": away_id,
                "home_team": str(m["home_team"]), "away_team": str(m["away_team"]),
                "home_goals": int(m["home_score"]), "away_goals": int(m["away_score"]),
                "has_360": False, "data_source": "real", "is_synthetic": False,
            })
            result_rows.append({"match_id": mid, "home_team": home_id, "away_team": away_id,
                                "home_goals": int(m["home_score"]),
                                "away_goals": int(m["away_score"])})

        if not frames:
            return None
        actions = pd.concat(frames, ignore_index=True)
        matches_df = pd.DataFrame(match_rows)

        # team -> its (most common) competition/season from the fixtures
        team_comp: dict[int, str] = {}
        team_season: dict[int, str] = {}
        for r in match_rows:
            for tid in (r["home_team_id"], r["away_team_id"]):
                team_comp.setdefault(tid, r["competition"])
                team_season.setdefault(tid, r["season"])

        team_ids = sorted(int(t) for t in actions["team_id"].unique())
        teams = pd.DataFrame([{
            "team_id": t, "team_name": team_name_map.get(t, f"Team {t}"),
            "competition": team_comp.get(t, "multi"),
            "season": team_season.get(t, "multi"),
            "formation": formation_map.get(t, "unknown"),
        } for t in team_ids])

        player_rows = []
        for pid in sorted(int(p) for p in actions["player_id"].unique()):
            pos = pos_map.get(pid)
            player_rows.append({
                "player_id": pid, "player_name": name_map.get(pid, f"Player {pid}"),
                "team_id": team_of_player.get(pid, team_ids[0]),
                "position": pos or "UNK", "role_group": _sb_role_group(pos),
                "is_starter": True, "shirt": 0,
            })
        players = pd.DataFrame(player_rows)

        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        focus = team_name or "all teams"
        seasons_sorted = sorted(seasons_seen)
        comps_sorted = sorted(comps_seen)
        span = f"{seasons_sorted[0]} – {seasons_sorted[-1]}" if seasons_sorted else "n/a"
        comp_label = (competition_name if competition_name
                      else f"{len(comps_sorted)} competitions")
        manifest = {
            "generated_at": now, "data_mode": "real",
            "source": f"StatsBomb Open Data — {comp_label} ({focus}), seasons {span}",
            "is_synthetic": False, "competition": comp_label,
            "season": span, "seasons": seasons_sorted, "competitions": comps_sorted,
            "seed": None,
            "match_ids": [int(x) for x in sorted(actions["match_id"].unique())],
            "row_counts": {"actions": int(len(actions)), "matches": int(len(match_rows)),
                           "teams": int(len(teams)), "players": int(len(players))},
            "has_360": False, "canonical_columns": list(COLUMNS),
            "limitations": [
                "real StatsBomb event data; positions are modal in-match positions.",
                "no 360 freeze frames in this pull: pressure/line-breaking context is "
                "event-only and approximated.",
                f"broad sample: {len(match_rows)} matches across {len(comps_sorted)} "
                f"competition(s) and {len(seasons_sorted)} season(s) ({span}). "
                "Raise --sb-max-matches (0 = all) for fuller coverage.",
            ],
        }
        return DataSpine(actions=actions, matches=matches_df, teams=teams,
                         players=players, results=pd.DataFrame(result_rows), manifest=manifest)
    except Exception:
        return None
