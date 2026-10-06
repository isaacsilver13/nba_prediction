import numpy as np
import pandas as pd
import pytest

import experiment as exp
from src.research_dashboard.capture_run import capture
from src.research_dashboard.manifest import load_manifest
from src.research_dashboard.replay_run import build_artifacts, odds_tiers, roi_detail_fn


def synthetic_preds(n=240, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 6, n).round(1)
    return pd.DataFrame({
        "GAME_DATE": pd.date_range("2024-01-01", periods=n // 4).repeat(4),
        "home_margin": -s + rng.normal(0, 12, n),
        "spread_signed": s,
        "payout_home": 0.91, "payout_away": 0.91,
        "pred_ensemble": -s + rng.normal(0, 4, n),
        "sigma_ensemble": 12.0,
        "GAME_ID_KEY": [f"G{i:04d}" for i in range(n)],
        "fold": np.repeat([1, 2, 3], n // 3),
    })


def test_detail_matches_frozen_compute_roi():
    p = synthetic_preds()
    d = roi_detail_fn(exp)(p.copy())
    assert d["bankroll"] - 1.0 == pytest.approx(exp.compute_roi(p.copy()))
    assert d["selected"].sum() > 0


def test_odds_tiers():
    raw = pd.DataFrame({"home_raw": [-110.0, -105.0, np.nan], "away_raw": [-110.0, np.nan, np.nan]},
                       index=["A", "B", "C"])
    keys = pd.Series(["A", "B", "C", "Z"])
    assert odds_tiers(keys, raw).tolist() == ["matched", "mirrored", "default", "unmatched"]


def test_built_artifacts_make_a_complete_run(tmp_path):
    p = synthetic_preds()
    d = roi_detail_fn(exp)(p.copy())
    raw = pd.DataFrame({"home_raw": -110.0, "away_raw": -110.0}, index=p["GAME_ID_KEY"][:200])
    paths = build_artifacts(None, p, None, d, raw, -110.0, tmp_path / "a", {"roi": 0.0})
    run = load_manifest(capture("r1", paths, tmp_path / "out"))
    assert run.status == "complete", run.problems
    bets = run.read("bets")
    assert len(bets) == int(d["selected"].sum())
    assert bets["bankroll"].iloc[-1] == pytest.approx(d["bankroll"], abs=1e-4)
    assert set(run.read("odds_audit")["odds_tier"]) == {"matched", "unmatched"}
