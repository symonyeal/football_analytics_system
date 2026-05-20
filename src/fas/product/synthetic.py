"""Deterministic synthetic football league for the product fallback.

The generator produces a small but *rich* event dataset on the canonical
StatsBomb pitch ``[0,120] x [0,80]``. It is intentionally structured so that
every product view has something to show:

- multiple matches, multiple teams, full squads with positions and roles;
- passes drawn from labelled spatial *lanes* so pass clustering finds real
  clusters ("left half-space switch", "central wall pass", ...);
- a deliberate hub shift between the first and second half so centralisation,
  hub-identity drift, and style-drift charts are non-trivial;
- shots, carries, pressures, tackles, interceptions, recoveries, clearances,
  and set-piece-like deliveries;
- match metadata with scores and a results table for matchup forecasting.

Everything is seeded, so ``generate_league(seed=7)`` is byte-stable. The data
is *event-only*: no freeze frames are produced, which lets the product exercise
its "event-only" degradation paths honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fas.data.schema import PITCH_LENGTH, PITCH_WIDTH, validate_actions

# Stable per-role home coordinates, attacking toward +x (StatsBomb per-team
# normalization: a team's own actions always attack left -> right).
ROLE_COORDS: dict[str, tuple[float, float]] = {
    "GK": (10.0, 40.0),
    "LB": (32.0, 12.0),
    "LCB": (24.0, 30.0),
    "RCB": (24.0, 50.0),
    "RB": (32.0, 68.0),
    "CDM": (46.0, 40.0),
    "CM": (58.0, 30.0),
    "RCM": (58.0, 50.0),
    "CAM": (72.0, 40.0),
    "LW": (86.0, 14.0),
    "RW": (86.0, 66.0),
    "ST": (98.0, 40.0),
}

ROLE_GROUP: dict[str, str] = {
    "GK": "Goalkeeper",
    "LB": "Defender", "LCB": "Defender", "RCB": "Defender", "RB": "Defender",
    "CDM": "Midfielder", "CM": "Midfielder", "RCM": "Midfielder", "CAM": "Midfielder",
    "LW": "Forward", "RW": "Forward", "ST": "Forward",
}

# Starting XI role layout per formation (11 roles).
FORMATION_XI: dict[str, list[str]] = {
    "4-3-3": ["GK", "LB", "LCB", "RCB", "RB", "CDM", "CM", "RCM", "LW", "RW", "ST"],
    "4-2-3-1": ["GK", "LB", "LCB", "RCB", "RB", "CDM", "RCM", "CAM", "LW", "RW", "ST"],
    "3-5-2": ["GK", "LCB", "CDM", "RCB", "LB", "CM", "RCM", "RB", "CAM", "LW", "ST"],
}

# Bench roles (7) to round each squad up to 18 named players.
BENCH_ROLES = ["GK", "LCB", "RB", "CM", "CAM", "LW", "ST"]

# Spatial pass lanes: (label, x0, y0, x1, y1). Passes are sampled around these
# centroids with Gaussian noise so density clustering recovers them.
PASS_LANES: list[tuple[str, float, float, float, float]] = [
    ("build from the back", 20.0, 40.0, 44.0, 38.0),
    ("left wide progression", 34.0, 14.0, 68.0, 12.0),
    ("right wide progression", 34.0, 66.0, 68.0, 68.0),
    ("left half-space switch", 54.0, 26.0, 78.0, 60.0),
    ("central wall pass", 60.0, 40.0, 74.0, 40.0),
    ("deep diagonal", 26.0, 30.0, 62.0, 60.0),
    ("final-third cutback", 96.0, 16.0, 86.0, 40.0),
    ("through ball in behind", 78.0, 40.0, 102.0, 42.0),
]

TEAM_NAMES = {
    100: "Athletic Union",
    101: "Riverside FC",
    102: "Northgate City",
    103: "Harbor Rovers",
}

# Per-team lane preference weights (style fingerprints). Index aligns to
# PASS_LANES. Northgate is wing-heavy, Riverside is central, etc.
TEAM_LANE_STYLE: dict[int, np.ndarray] = {
    100: np.array([1.4, 1.0, 1.0, 1.3, 1.2, 1.0, 0.8, 1.0]),  # balanced, half-space
    101: np.array([1.6, 0.6, 0.6, 0.8, 1.8, 0.9, 0.6, 1.4]),  # central / vertical
    102: np.array([1.0, 1.8, 1.8, 0.7, 0.7, 1.0, 1.5, 0.8]),  # wing-heavy
    103: np.array([1.3, 1.0, 1.0, 1.0, 1.0, 1.6, 1.0, 1.0]),  # diagonal switches
}


@dataclass(frozen=True, slots=True)
class SyntheticLeague:
    """Bundle of canonical frames produced by :func:`generate_league`."""

    actions: pd.DataFrame
    matches: pd.DataFrame
    teams: pd.DataFrame
    players: pd.DataFrame
    results: pd.DataFrame
    seed: int


def _squad(team_id: int, formation: str) -> list[dict]:
    """Return 18 player records (11 starters + 7 bench) for a team."""
    roles = list(FORMATION_XI[formation]) + list(BENCH_ROLES)
    out = []
    for i, role in enumerate(roles):
        pid = team_id * 100 + (i + 1)
        out.append({
            "player_id": pid,
            "player_name": f"{TEAM_NAMES[team_id].split()[0]} {role}{i + 1:02d}",
            "team_id": team_id,
            "position": role,
            "role_group": ROLE_GROUP[role],
            "is_starter": i < 11,
            "shirt": i + 1,
        })
    return out


def _nearest_player(coords: dict[int, tuple[float, float]], x: float, y: float,
                    exclude: int | None = None) -> int:
    best, bd = None, 1e18
    for pid, (px, py) in coords.items():
        if pid == exclude:
            continue
        d = (px - x) ** 2 + (py - y) ** 2
        if d < bd:
            best, bd = pid, d
    return int(best)


def _generate_match_actions(
    match_id: int,
    home_id: int,
    away_id: int,
    home_starters: list[dict],
    away_starters: list[dict],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int, int]:
    """Generate one match's canonical actions plus the final score."""
    rows: list[dict] = []
    home_goals = away_goals = 0

    for team_id, starters, opp_id in (
        (home_id, home_starters, away_id),
        (away_id, away_starters, home_id),
    ):
        coords = {p["player_id"]: ROLE_COORDS[p["position"]] for p in starters}
        atk = [p for p in starters if p["role_group"] in ("Midfielder", "Forward")]
        # Hub identity drift: 1st-half hub is a midfielder, 2nd-half a forward.
        first_hub = min(atk, key=lambda p: ROLE_COORDS[p["position"]][0])["player_id"]
        second_hub = max(atk, key=lambda p: ROLE_COORDS[p["position"]][0])["player_id"]
        lane_w = TEAM_LANE_STYLE.get(team_id, np.ones(len(PASS_LANES)))
        lane_p = lane_w / lane_w.sum()

        n_possessions = int(rng.integers(120, 150))
        total_ms = 5_400_000  # 90 minutes; possessions spread across the match
        for poss in range(n_possessions):
            base_t = int((poss + rng.uniform(0.1, 0.9)) / n_possessions * total_ms)
            t = base_t
            period = 1 if t < 2_700_000 else 2
            hub = first_hub if period == 1 else second_hub
            chain_len = int(rng.integers(2, 7))
            # Start the possession deep.
            x = float(np.clip(rng.normal(28, 8), 2, 60))
            y = float(np.clip(rng.normal(40, 18), 2, PITCH_WIDTH - 2))
            carrier = _nearest_player(coords, x, y)

            for step in range(chain_len):
                t += int(rng.integers(400, 2200))
                period = 1 if t < 2_700_000 else 2
                li = int(rng.choice(len(PASS_LANES), p=lane_p))
                _, lx0, ly0, lx1, ly1 = PASS_LANES[li]
                x0 = float(np.clip(lx0 + rng.normal(0, 4.0), 0, PITCH_LENGTH))
                y0 = float(np.clip(ly0 + rng.normal(0, 4.0), 0, PITCH_WIDTH))
                x1 = float(np.clip(lx1 + rng.normal(0, 4.5), 0, PITCH_LENGTH))
                y1 = float(np.clip(ly1 + rng.normal(0, 4.5), 0, PITCH_WIDTH))

                # Bias the passer toward the current hub to create centralisation.
                if rng.random() < 0.35:
                    passer = hub
                else:
                    passer = _nearest_player(coords, x0, y0)
                receiver = _nearest_player(coords, x1, y1, exclude=passer)

                kind = "carry" if rng.random() < 0.18 else "pass"
                completed = bool(rng.random() > (0.12 if kind == "carry" else 0.18))
                rows.append({
                    "match_id": match_id, "period": period,
                    "timestamp_ms": min(t, 5_399_999),
                    "player_id": passer, "team_id": team_id,
                    "action_type": kind,
                    "x_start": x0, "y_start": y0,
                    "x_end": x1, "y_end": y1, "outcome": completed,
                })
                if not completed:
                    # Defensive recovery by the opponent.
                    _emit_defensive(rows, match_id, period, t, opp_id,
                                    PITCH_LENGTH - x1, PITCH_WIDTH - y1, rng)
                    break
                carrier = receiver
                x, y = x1, y1

            # Possession end: shot from advanced area or a turnover.
            if x > 78 and rng.random() < 0.55:
                t += int(rng.integers(500, 1500))
                period = 1 if t < 2_700_000 else 2
                goal = bool(rng.random() < 0.13)
                rows.append({
                    "match_id": match_id, "period": period,
                    "timestamp_ms": min(t, 5_399_999),
                    "player_id": carrier, "team_id": team_id,
                    "action_type": "shot",
                    "x_start": float(np.clip(x + rng.normal(2, 3), 80, 119)),
                    "y_start": float(np.clip(y + rng.normal(0, 6), 2, PITCH_WIDTH - 2)),
                    "x_end": 120.0, "y_end": 40.0, "outcome": goal,
                })
                if goal:
                    if team_id == home_id:
                        home_goals += 1
                    else:
                        away_goals += 1

            # Occasional set-piece-like delivery from a wide deep area.
            if rng.random() < 0.12:
                t += int(rng.integers(500, 2000))
                period = 1 if t < 2_700_000 else 2
                side_y = 2.0 if rng.random() < 0.5 else PITCH_WIDTH - 2.0
                rows.append({
                    "match_id": match_id, "period": period,
                    "timestamp_ms": min(t, 5_399_999),
                    "player_id": _nearest_player(coords, 100.0, side_y),
                    "team_id": team_id, "action_type": "pass",
                    "x_start": float(np.clip(rng.normal(102, 4), 90, 119)),
                    "y_start": side_y,
                    "x_end": float(np.clip(rng.normal(112, 3), 100, 119)),
                    "y_end": float(np.clip(rng.normal(40, 6), 20, 60)),
                    "outcome": bool(rng.random() > 0.55),
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["period", "timestamp_ms"]).reset_index(drop=True)
    return validate_actions(df), home_goals, away_goals


def _emit_defensive(rows, match_id, period, t, opp_id, x, y, rng):
    """Append a pressure + a recovery-type defensive action by the opponent."""
    x = float(np.clip(x, 0, PITCH_LENGTH))
    y = float(np.clip(y, 0, PITCH_WIDTH))
    rows.append({
        "match_id": match_id, "period": period, "timestamp_ms": t % 2_700_000,
        "player_id": opp_id * 100 + int(rng.integers(2, 9)), "team_id": opp_id,
        "action_type": "pressure", "x_start": x, "y_start": y,
        "x_end": np.nan, "y_end": np.nan, "outcome": True,
    })
    dtype = rng.choice(["tackle", "interception", "recovery", "clearance"],
                       p=[0.3, 0.3, 0.25, 0.15])
    rows.append({
        "match_id": match_id, "period": period,
        "timestamp_ms": min(t + 300, 5_399_999),
        "player_id": opp_id * 100 + int(rng.integers(2, 12)), "team_id": opp_id,
        "action_type": str(dtype), "x_start": x, "y_start": y,
        "x_end": np.nan, "y_end": np.nan, "outcome": bool(rng.random() > 0.25),
    })


def generate_league(*, seed: int = 7, competition: str = "FAS Synthetic League",
                    season: str = "2024/2025") -> SyntheticLeague:
    """Generate the full deterministic synthetic league.

    Returns a :class:`SyntheticLeague` with 6 matches across 4 teams, 18 named
    players per team, lineups/formations, and a results table.
    """
    rng = np.random.default_rng(seed)
    team_ids = [100, 101, 102, 103]
    formations = {100: "4-3-3", 101: "4-2-3-1", 102: "3-5-2", 103: "4-3-3"}

    players: list[dict] = []
    team_rows: list[dict] = []
    squads: dict[int, list[dict]] = {}
    for tid in team_ids:
        squad = _squad(tid, formations[tid])
        squads[tid] = squad
        players.extend(squad)
        team_rows.append({
            "team_id": tid, "team_name": TEAM_NAMES[tid],
            "competition": competition, "season": season,
            "formation": formations[tid],
        })

    # Fixture list: a single round-robin half (6 matches for 4 teams).
    fixtures = [
        (100, 101), (102, 103), (100, 102),
        (103, 101), (100, 103), (102, 101),
    ]
    dates = pd.date_range("2024-08-17", periods=len(fixtures), freq="7D")

    action_frames: list[pd.DataFrame] = []
    match_rows: list[dict] = []
    result_rows: list[dict] = []
    for k, (home, away) in enumerate(fixtures):
        mid = 9000 + k
        home_starters = [p for p in squads[home] if p["is_starter"]]
        away_starters = [p for p in squads[away] if p["is_starter"]]
        acts, hg, ag = _generate_match_actions(
            mid, home, away, home_starters, away_starters, rng)
        action_frames.append(acts)
        match_rows.append({
            "match_id": mid, "competition": competition, "season": season,
            "date": dates[k].strftime("%Y-%m-%d"),
            "home_team_id": home, "away_team_id": away,
            "home_team": TEAM_NAMES[home], "away_team": TEAM_NAMES[away],
            "home_goals": hg, "away_goals": ag,
            "has_360": False, "data_source": "synthetic", "is_synthetic": True,
        })
        result_rows.append({
            "match_id": mid, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag,
        })

    return SyntheticLeague(
        actions=pd.concat(action_frames, ignore_index=True),
        matches=pd.DataFrame(match_rows),
        teams=pd.DataFrame(team_rows),
        players=pd.DataFrame(players),
        results=pd.DataFrame(result_rows),
        seed=seed,
    )
