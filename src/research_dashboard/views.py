"""Pure helpers for dashboard pages (testable without Streamlit)."""

import numpy as np
import pandas as pd


def filter_tiers(df: pd.DataFrame, tiers) -> pd.DataFrame:
    return df[df["odds_tier"].isin(list(tiers))]


def bet_summary(bets: pd.DataFrame) -> dict:
    """Units are fractions of starting bankroll; ROI = P&L / total staked."""
    if bets.empty:
        return {"bets": 0}
    bk = bets.sort_values("game_date")["bankroll"]
    return {
        "bets": len(bets),
        "win_rate": bets["win"].mean(),
        "roi": bets["pnl"].sum() / bets["kelly_fraction"].sum(),
        "max_drawdown": float(((bk.cummax() - bk) / bk.cummax()).max()),
    }


def reliability(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Reliability on ALL test games (not just selected bets), with per-bin counts."""
    b = pd.cut(pred["p_home"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = pred.groupby(b, observed=True).agg(mean_pred=("p_home", "mean"), actual=("home_win", "mean"),
                                              n=("home_win", "size")).reset_index(drop=True)
    return out
