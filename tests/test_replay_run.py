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


def test_edge_sign_follows_spread_convention():
    """spread_signed is the home handicap (-10 = home favoured by 10); pred +15 beats it, so bet HOME."""
    p = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(["2024-01-01"]), "home_margin": [12.0], "spread_signed": [-10.0],
        "payout_home": 0.91, "payout_away": 0.91, "pred_ensemble": [15.0], "sigma_ensemble": 5.0,
    })
    d = roi_detail_fn(exp)(p)
    assert d["df"]["_bet_side"].iloc[0] == "HOME" and d["win_prob"][0] > 0.5


def test_walk_forward_keeps_row_identity_with_tied_dates():
    """Index labels from the splits must point at the same games in the caller's frame (ties on date)."""
    n = 600
    df = pd.DataFrame({"GAME_DATE": pd.date_range("2024-01-01", periods=n // 6).repeat(6), "gid": np.arange(n)})
    df = df.sort_values("GAME_DATE", kind="stable").reset_index(drop=True)
    for tr, te in exp.walk_forward_splits(df, "GAME_DATE", 300, 100):
        assert (df.loc[te.index, "gid"].values == te["gid"].values).all()
        assert (df.loc[tr.index, "gid"].values == tr["gid"].values).all()


def _cal_frame(seed=0, folds=4, per=400):
    rng = np.random.default_rng(seed)
    n = folds * per
    s = rng.normal(0, 6, n)
    return pd.DataFrame({
        "pred_ensemble": -s + rng.normal(0, 4, n), "spread_signed": s,
        "home_margin": -s + rng.normal(0, 13, n), "sigma_ensemble": 5.0,
        "fold": np.repeat(np.arange(1, folds + 1), per).astype(float),
    })


def test_calibrate_sigma_uses_only_earlier_folds():
    f = _cal_frame()
    sig, mults = exp.calibrate_sigma(f)
    assert mults[1] == exp.NO_BET_SIGMA_MULT            # no history -> no bets
    assert all(m >= 1.0 for m in mults.values())
    g = f.copy()
    g.loc[g.fold >= 3, "home_margin"] += 50             # rewrite the future outcomes
    sig2, mults2 = exp.calibrate_sigma(g)
    assert mults[1] == mults2[1] and mults[2] == mults2[2]
    assert (sig[f.fold <= 2] == sig2[f.fold <= 2]).all()


def test_flat_stake_stats():
    p = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(["2024-01-01"] * 4), "home_margin": [12.0, 12.0, -12.0, -12.0],
        "spread_signed": [-3.0] * 4, "payout_home": 0.9, "payout_away": 0.9,
        "pred_ensemble": [10.0] * 4, "sigma_ensemble": 3.0,
    })
    d = exp.compute_roi(p, detail=True)
    mean, se, n = exp.flat_stake_stats(d)
    assert n == 4 and mean == pytest.approx((0.9 + 0.9 - 1 - 1) / 4) and se > 0
