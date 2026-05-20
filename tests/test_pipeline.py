"""End-to-end integration test (Part 7.2, offline synthetic variant)."""

from fas.data.schema import Action, validate_actions
from fas.examples.synthetic_pipeline import run_demo, synthetic_actions
from fas.nlp.set_piece_opt import optimize_set_piece
import numpy as np


def test_schema_validation_roundtrip():
    a = Action(1, 1, 0, 7, 100, "pass", 50, 40, 60, 40, True)
    df = synthetic_actions(seed=9)
    validate_actions(df)
    assert a.action_type == "pass"


def test_full_pipeline_produces_squad():
    out = run_demo()
    assert out["net"].n > 0
    assert out["squad"].status in ("Optimal", "Feasible", "Not Solved")
    assert out["match"].intensity_surface is not None
    assert out["summary"]["rapm_leader"] in out["net"].players
    assert out["summary"]["mdp_mean_value"] >= 0.0


def test_set_piece_lands_near_target():
    p0 = np.array([0.0, 0.0, 0.0])
    target = np.array([25.0, 5.0, 0.0])
    res = optimize_set_piece(p0, target, n_restarts=6)
    # multi-start should land within a few metres on a clean (no-wall) kick
    assert res.miss_distance < 12.0
