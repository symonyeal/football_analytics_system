"""Product-layer tests: synthetic data, artifacts, loader, and analytics.

All offline and deterministic. No network access required.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fas.data.schema import COLUMNS, validate_actions
from fas.product import ARTIFACT_FILES
from fas.product.build import product_build
from fas.product.centralisation import (
    centralisation_by_phase,
    centralisation_from_network,
    freeman_centralisation,
)
from fas.product.clustering import cluster_passes
from fas.product.formation import infer_formation, starting_formation_from_lineup
from fas.product.ingest import build_spine
from fas.product.loader import artifacts_present, load_product
from fas.product.synthetic import generate_league
from fas.graph import build_pass_network


# --------------------------------------------------------------------------- #
# Synthetic data spine
# --------------------------------------------------------------------------- #

def test_synthetic_league_shape():
    lg = generate_league(seed=7)
    assert lg.matches["match_id"].nunique() >= 3
    assert lg.teams["team_id"].nunique() >= 2
    per_team = lg.players.groupby("team_id").size()
    assert (per_team >= 18).all()
    types = set(lg.actions["action_type"].unique())
    for required in ("pass", "carry", "shot", "pressure", "tackle", "interception"):
        assert required in types
    # both halves populated
    assert set(lg.actions["period"].unique()) == {1, 2}
    # canonical schema holds
    validate_actions(lg.actions)


def test_synthetic_is_deterministic():
    a = generate_league(seed=7).actions
    b = generate_league(seed=7).actions
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# Centralisation
# --------------------------------------------------------------------------- #

def test_freeman_centralisation_bounds():
    n = 6
    even = np.full(n, 1.0 / n)
    star = np.zeros(n)
    star[0] = 1.0
    assert freeman_centralisation(even) == pytest.approx(0.0, abs=1e-9)
    assert freeman_centralisation(star) == pytest.approx(1.0, abs=1e-9)
    mid = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert 0.0 < freeman_centralisation(mid) < 1.0


def test_centralisation_from_network_runs():
    lg = generate_league(seed=7)
    m = lg.actions[lg.actions["match_id"] == lg.actions["match_id"].iloc[0]]
    team = int(m["team_id"].iloc[0])
    net = build_pass_network(m, team_id=team)
    res = centralisation_from_network(net)
    assert 0.0 <= res.index <= 1.0
    assert res.hub_player in net.players
    cb = centralisation_by_phase(m, team)
    assert len(cb) == 6
    assert cb["centralisation"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# Pass clustering
# --------------------------------------------------------------------------- #

def test_pass_clustering_on_toy_data():
    # two tight lanes -> should recover at least two labelled clusters
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        rows.append((30 + rng.normal(0, 1), 20 + rng.normal(0, 1),
                     60 + rng.normal(0, 1), 20 + rng.normal(0, 1)))
    for _ in range(120):
        rows.append((30 + rng.normal(0, 1), 60 + rng.normal(0, 1),
                     60 + rng.normal(0, 1), 60 + rng.normal(0, 1)))
    df = pd.DataFrame(rows, columns=["x_start", "y_start", "x_end", "y_end"])
    df["match_id"] = 1
    df["period"] = 1
    df["timestamp_ms"] = range(len(df))
    df["player_id"] = 1
    df["team_id"] = 100
    df["action_type"] = "pass"
    df["outcome"] = True
    pc = cluster_passes(validate_actions(df), team_id=100)
    assert len(pc.summary) >= 2
    assert pc.summary["label"].notna().all()
    assert pc.n_noise >= 0


# --------------------------------------------------------------------------- #
# Formation inference
# --------------------------------------------------------------------------- #

def test_formation_inference_on_toy_lineup():
    lineup = pd.DataFrame({
        "player_id": list(range(1, 12)),
        "position": ["GK", "LB", "LCB", "RCB", "RB", "CDM", "CM", "RCM", "LW", "RW", "ST"],
    })
    res = starting_formation_from_lineup(lineup)
    assert res.confidence == "lineup"
    assert res.formation == "4-3-3"
    # generic coordinate inference stays in [defenders..forwards] banding
    coords = {i: (float(x), 40.0) for i, x in enumerate([10, 25, 25, 25, 25, 55, 55, 55, 90, 90, 90])}
    assert infer_formation(coords).formation.count("-") >= 1


# --------------------------------------------------------------------------- #
# Artifact generation: synthetic fallback
# --------------------------------------------------------------------------- #

def test_product_build_synthetic(tmp_path):
    summary = product_build(data_root=tmp_path, allow_download=False, seed=7, verbose=False)
    assert summary["data_mode"] == "synthetic"
    assert summary["is_synthetic"] is True
    assert summary["seed"] == 7
    out = tmp_path / "processed"
    for f in ARTIFACT_FILES:
        path = out / f
        assert path.exists(), f"missing artifact {f}"
        assert path.stat().st_size > 0
    assert artifacts_present(tmp_path)


def test_manifest_records_synthetic(tmp_path):
    product_build(data_root=tmp_path, allow_download=False, verbose=False)
    manifest = json.loads((tmp_path / "processed" / "manifest.json").read_text())
    assert manifest["is_synthetic"] is True
    assert manifest["seed"] is not None
    assert manifest["row_counts"]["matches"] >= 3
    assert any("synthetic" in lim.lower() for lim in manifest["limitations"])


# --------------------------------------------------------------------------- #
# Artifact generation: small canonical actions file
# --------------------------------------------------------------------------- #

def _toy_actions() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for mid in (1, 2):
        for team in (100, 200):
            for _ in range(150):
                x0, y0 = rng.uniform(0, 120), rng.uniform(0, 80)
                rows.append({
                    "match_id": mid, "period": 1, "timestamp_ms": int(rng.integers(0, 5_000_000)),
                    "player_id": team + int(rng.integers(1, 12)), "team_id": team,
                    "action_type": rng.choice(["pass", "carry", "shot"], p=[0.7, 0.2, 0.1]),
                    "x_start": x0, "y_start": y0,
                    "x_end": float(np.clip(x0 + rng.normal(8, 10), 0, 120)),
                    "y_end": float(np.clip(y0 + rng.normal(0, 10), 0, 80)),
                    "outcome": bool(rng.random() > 0.3),
                })
    return validate_actions(pd.DataFrame(rows))


def test_product_build_from_canonical_file(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir(parents=True)
    _toy_actions().to_parquet(proc / "actions.parquet", index=False)
    summary = product_build(data_root=tmp_path, allow_download=False, verbose=False)
    assert summary["data_mode"] == "local"
    assert summary["is_synthetic"] is False
    p = load_product(tmp_path)
    assert set(p.tables["actions"].columns) >= set(COLUMNS)
    assert len(p.tables["matchup_artifacts"]) >= 2  # 2 teams -> ordered pairs


# --------------------------------------------------------------------------- #
# Loader + insight cards
# --------------------------------------------------------------------------- #

def test_loader_and_insight_cards(tmp_path):
    product_build(data_root=tmp_path, allow_download=False, verbose=False)
    p = load_product(tmp_path)
    assert p.summary["n_matches"] >= 3
    ins = p.tables["insights"]
    assert not ins.empty
    required = {"title", "claim", "evidence", "method", "sample_size",
                "baseline", "validation_status", "caveats", "next_look"}
    assert required.issubset(ins.columns)
    assert (ins["sample_size"] > 0).all()
    assert ins["method"].str.len().gt(0).all()
    # at least one FDR-validated card
    assert ins["validation_status"].astype(str).str.startswith("validated").any()


def test_loader_raises_without_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_product(tmp_path)


def test_statsbomb_loader_handles_flattened_columns():
    # statsbombpy flattens nested objects into columns; the loader must read
    # pass_end_location / pass_outcome / shot_outcome (regression guard).
    from fas.data.statsbomb import events_to_actions

    events = pd.DataFrame([
        {"type": "Pass", "location": [50.0, 40.0], "pass_end_location": [70.0, 45.0],
         "pass_outcome": None, "minute": 1, "second": 0, "player_id": 10, "team_id": 1,
         "period": 1},
        {"type": "Pass", "location": [60.0, 30.0], "pass_end_location": [55.0, 35.0],
         "pass_outcome": "Incomplete", "minute": 2, "second": 0, "player_id": 11,
         "team_id": 1, "period": 1},
        {"type": "Shot", "location": [110.0, 40.0], "shot_end_location": [120.0, 40.0],
         "shot_outcome": "Goal", "minute": 3, "second": 0, "player_id": 12, "team_id": 1,
         "period": 1},
    ])
    acts = events_to_actions(events, match_id=1)
    passes = acts[acts["action_type"] == "pass"]
    assert passes["x_end"].notna().all()  # end locations parsed
    assert bool(passes.iloc[0]["outcome"]) is True   # completed
    assert bool(passes.iloc[1]["outcome"]) is False  # incomplete
    shot = acts[acts["action_type"] == "shot"].iloc[0]
    assert bool(shot["outcome"]) is True  # goal


def test_match_artifacts_carry_context(tmp_path):
    product_build(data_root=tmp_path, allow_download=False, verbose=False)
    p = load_product(tmp_path)
    ma = p.tables["match_artifacts"]
    for col in ("match_id", "team_id", "opponent_id", "competition", "season",
                "phase", "model_name", "sample_size", "data_source", "is_synthetic",
                "limitations"):
        assert col in ma.columns
