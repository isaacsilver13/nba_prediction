"""Regression tests for compute_roi / flat_stake_stats settlement and bet-eligibility rules."""
import numpy as np
import pandas as pd

import experiment as exp


def _row(margin, spread=-5.0, pred=10.0, matched=True, bettable=True, date="2024-01-01"):
    # pred + spread = +5 edge with sigma 10 -> home bet at 1.0 payout is +EV
    return dict(GAME_DATE=pd.Timestamp(date), home_margin=margin, spread_signed=spread, pred_ensemble=pred,
                sigma_ensemble=10.0, payout_home=0.91, payout_away=0.91, odds_matched=matched, bettable=bettable)


def _run(rows):
    d = exp.compute_roi(pd.DataFrame(rows), detail=True)
    return d, exp.flat_stake_stats(d)


def test_push_is_refunded_not_loss():
    d, (mean, _, n) = _run([_row(margin=5.0)])  # home_margin == -spread_signed -> push
    assert d["covered"][0] == "PUSH" and n == 1 and mean == 0.0 and d["pnl"][0] == 0.0


def test_win_and_loss_still_settle():
    d, (mean, _, _) = _run([_row(8.0), _row(2.0, date="2024-01-02")])
    assert d["pnl"][0] > 0 and d["pnl"][1] < 0


def test_unmatched_odds_never_bet():
    _, (_, _, n) = _run([_row(8.0, matched=False)])
    assert n == 0


def test_uncalibrated_fold_never_bets():
    _, (_, _, n) = _run([_row(8.0, bettable=False)])
    assert n == 0
