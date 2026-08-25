"""
NBA Prediction - Autoresearch Experiment Script
================================================
This is the file the AI agent modifies each iteration.
All tunable configuration lives in the AGENT-EDITABLE CONFIG block.
Data loading and metric calculation are FROZEN below the marked boundary.

On each run, appends one row to results.tsv:
    timestamp | exp_id | ensemble_rmse | roi | score | params
"""

import hashlib
import io
import json
import os
import sys
import time
from contextlib import redirect_stderr
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from xgboost import XGBRegressor

TRAIN_SIZE = 2500
TEST_SIZE = 300

EV_THRESHOLD = 0.02         # minimum expected value to place a bet
FRACTIONAL_KELLY = 0.30     # fraction of full Kelly to wager
MAX_KELLY = 0.08            # per-bet bankroll cap
TOP_N_BETS_PER_DAY = 5      # max bets selected per game-day
DAILY_MAX_EXPOSURE = 0.25   # max total bankroll at risk per day

MODEL_SPECS = [
    {
        "id": "lgb_v1",
        "model": lgb.LGBMRegressor(
            n_estimators=1200, learning_rate=0.02, max_depth=4,
            num_leaves=20, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        ),
        "needs_imputer": False,
    },
    {
        "id": "lgb_v2",
        "model": lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=40,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, verbose=-1,
        ),
        "needs_imputer": False,
    },
    {
        "id": "elasticnet",
        "model": ElasticNet(
            alpha=0.3, l1_ratio=0.7, random_state=42,
            max_iter=20000, tol=1e-3, selection="random",
        ),
        "needs_imputer": True,
    },
    {
        "id": "xgb_v1",
        "model": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42,
        ),
        "needs_imputer": False,
    },
]

ENSEMBLE_WEIGHTS = "inverse_rmse"

EXTRA_FEATURE_EXCLUSIONS: list[str] = ["home_team_net_fgm_r5", "away_team_net_fgm_r5", "home_team_net_fga_r5", "away_team_net_fga_r5"]
EXTRA_FEATURE_INCLUSIONS: list[str] = []

PYTH_EXPONENT = 16.5          # Oliver Pythagorean exponent (Basketball on Paper)
FF_WEIGHTS = {"efg": 0.40, "tov": 0.25, "oreb": 0.20, "ftr": 0.15}

# ═══════════════════════════════════════════════════════════════════════════
# FROZEN — do not modify anything below this line
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_TSV = "results.tsv"
EXPERIMENTS_DIR = "experiments"

DATA_MODEL_PATH = "data/processed/df_model_3.csv"
ODDS_PATH = "data/odds/nba_2008-2025.csv"
PROCESSED_GAMES_PATH = "data/processed/nba_games_with_game_id_processed.csv"

DEFAULT_AMERICAN_ODDS = -110.0
DEFAULT_PAYOUT = 100.0 / abs(DEFAULT_AMERICAN_ODDS)

TEAM_MAP = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",
}

DENY_EXACT = {
    # Targets
    "home_margin", "away_margin",
    # Result labels
    "did_home_cover", "home_cover", "favorite_cover",
    "favorite_cover_manual", "team_that_covered",
    # Leakage: computed by a prior model run
    "abs_edge", "edge_rank_today", "edge_percentile_last_30_days",
    "model_error_roll_last20",
    # Redundant Elo monotonic transforms (trees find nonlinearity natively)
    "abs_elo_diff", "elo_diff_squared", "sigmoid_elo_diff",
    # Manual interactions (trees find these automatically)
    "rest_diff_elo_diff", "home_elo_roll_2Y_rest_advantage",
    "pace_diff_rest_diff", "net_rating_diff_rest_diff",
    # Pace collinearity
    "combined_pace", "pace_interaction",
    # Redundant margin signals (rolling_margin_diff_10 captures the same info)
    "home_margin_last5", "away_margin_last5", "pts_diff_last5",
    "rolling_margin_diff_5",
    # Squared transforms (tree handles nonlinearity natively)
    "rest_advantage_sq", "streak_length_squared",
    # Raw spread (use signed version only)
    "spread",
}

DENY_SUBSTRINGS = (
    "model_error", "market_error", "result", "outcome", "postgame",
    "pred_", "sigma_", "error_", "abs_error", "edge_", "kelly_",
    "pnl_", "cum_pnl", "bet_",
)

# Curated static feature allowlist (Oliver principles — no leakage, no redundancy)
STATIC_FEATURES = [
    # Spread / market context
    "spread_signed",
    "is_home_favorite",
    # Rest / schedule
    "home_b2b", "away_b2b",
    "rest_diff",
    "rest_advantage",
    "rest_weighted_load",
    "games_last_3_days", "games_last_5_days", "games_last_7_days",
    # Travel / fatigue
    "travel_diff_3d", "travel_diff_7d", "travel_diff_10d",
    "travel_1d_z", "travel_3d_z", "travel_7d_z", "travel_10d_z",
    "fatigue_diff",
    # Elo quality signals
    "elo_diff", "home_elo_roll_2Y", "away_elo_roll_2Y",
    # Recent form
    "margin_diff_last5",
    "rolling_margin_diff_10",
    "win_streak", "loss_streak",
    "after_ot_flag",
    "blowout_pct_last20", "blowout_pct_season",
    "ats_roll_last20",
    # Variance / consistency
    "margin_std_last_10", "home_margin_std_last_10", "away_margin_std_last_10",
    # Schedule load
    "top8_points_diff",
    # Injury / roster
    "missing_top1_flag", "missing_top2_flag",
    "minutes_lost_last_game", "usage_lost_last_game",
    # Ratings (short-window, non-dynamic)
    "bench_net_rating_last_5",
    "off_rating_last_5", "def_rating_last_5",
    "net_rating_last_5", "net_rating_trend",
    "net_rating_diff",
    # Style matchup
    "team_style_vs_opponent_style",
    # Pace (single predictor, no collinear variants)
    "pace_diff",
    # Rolling net box score r5
    "home_team_net_points_r5",   "home_team_net_rebounds_r5",  "home_team_net_assists_r5",
    "home_team_net_STL_r5",      "home_team_net_BLK_r5",       "home_team_net_turnovers_r5",
    "home_team_net_OREB_r5",     "home_team_net_fgm_r5",       "home_team_net_fga_r5",
    "home_team_net_three_pm_r5", "home_team_net_three_pa_r5",  "home_team_net_ftm_r5",
    "home_team_net_fta_r5",
    # Rolling net box score r10
    "home_team_net_points_r10",   "home_team_net_rebounds_r10", "home_team_net_assists_r10",
    "home_team_net_STL_r10",      "home_team_net_BLK_r10",      "home_team_net_turnovers_r10",
    "home_team_net_OREB_r10",     "home_team_net_fgm_r10",      "home_team_net_fga_r10",
    "home_team_net_three_pm_r10", "home_team_net_three_pa_r10", "home_team_net_ftm_r10",
    "home_team_net_fta_r10",
    "away_team_net_points_r5",   "away_team_net_rebounds_r5",  "away_team_net_assists_r5",
    "away_team_net_STL_r5",      "away_team_net_BLK_r5",       "away_team_net_turnovers_r5",
    "away_team_net_OREB_r5",     "away_team_net_fgm_r5",       "away_team_net_fga_r5",
    "away_team_net_three_pm_r5", "away_team_net_three_pa_r5",  "away_team_net_ftm_r5",
    "away_team_net_fta_r5",
    "away_team_net_points_r10",   "away_team_net_rebounds_r10", "away_team_net_assists_r10",
    "away_team_net_STL_r10",      "away_team_net_BLK_r10",      "away_team_net_turnovers_r10",
    "away_team_net_OREB_r10",     "away_team_net_fgm_r10",      "away_team_net_fga_r10",
    "away_team_net_three_pm_r10", "away_team_net_three_pa_r10", "away_team_net_ftm_r10",
    "away_team_net_fta_r10",
    # Payout info
    "payout_home", "payout_away",
    # --- Oliver derived features (computed by add_oliver_features) ---
    "efg_diff",            # Four Factors: Shooting
    "tov_pct_diff",        # Four Factors: Turnovers
    "oreb_pct_diff",       # Four Factors: Offensive rebounding
    "ftr_diff",            # Four Factors: Free throw rate
    "oliver_ff_composite", # Weighted composite of all four factors
    "pyth_win_pct_home",   # Pythagorean W% home team
    "pyth_win_pct_away",   # Pythagorean W% away team
    "pyth_diff",           # Pythagorean quality differential
]

DYNAMIC_PREFIXES = (
    "home_team_adv_", "away_team_adv_",
    "home_team_ff_",  "away_team_ff_",
    "home_team_sc_",  "away_team_sc_",
    "home_team_usg_", "away_team_usg_",
    "home_team_misc_","away_team_misc_",
    "home_team_ptrk_","away_team_ptrk_",
)
DYNAMIC_SUFFIXES = ("_r5", "_r10", "_r20", "_ewm5", "_ewm10", "_ewm20", "_s2d")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _canonical_game_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return s.map(lambda v: str(v).lstrip("0") if pd.notna(v) else np.nan)


def _normalize_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().map(TEAM_MAP)


def _american_to_payout(american: pd.Series) -> pd.Series:
    v = pd.to_numeric(american, errors="coerce")
    p = pd.Series(np.nan, index=v.index, dtype=float)
    neg = v < 0
    pos = v > 0
    p[neg] = 100.0 / v[neg].abs()
    p[pos] = v[pos] / 100.0
    return p


# ── Oliver feature engineering ────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as float series, or NaN series if missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _pyth_win_pct(ortg: pd.Series, drtg: pd.Series, exp: float = None) -> pd.Series:
    """Oliver Pythagorean win probability: ortg^exp / (ortg^exp + drtg^exp)."""
    if exp is None:
        exp = PYTH_EXPONENT
    o = ortg.clip(lower=1.0)
    d = drtg.clip(lower=1.0)
    o_exp = o ** exp
    d_exp = d ** exp
    return o_exp / (o_exp + d_exp)


def add_oliver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Dean Oliver Basketball on Paper features inline.

    Uses 10-game rolling Four Factors (r10 window) and advanced ratings.
    Falls back to NaN gracefully when source columns are absent.
    References FF_WEIGHTS and PYTH_EXPONENT from agent-editable config.
    """
    df = df.copy()

    # --- Four Factors differentials (home minus away) ---
    h_efg  = _safe_col(df, "home_team_ff_effective_fg_pct_r10")
    a_efg  = _safe_col(df, "away_team_ff_effective_fg_pct_r10")
    h_tov  = _safe_col(df, "home_team_ff_turnover_pct_r10")
    a_tov  = _safe_col(df, "away_team_ff_turnover_pct_r10")
    h_oreb = _safe_col(df, "home_team_ff_off_reb_pct_r10")
    a_oreb = _safe_col(df, "away_team_ff_off_reb_pct_r10")
    h_ftr  = _safe_col(df, "home_team_ff_free_throw_rate_r10")
    a_ftr  = _safe_col(df, "away_team_ff_free_throw_rate_r10")

    df["efg_diff"]      = h_efg  - a_efg
    df["tov_pct_diff"]  = h_tov  - a_tov
    df["oreb_pct_diff"] = h_oreb - a_oreb
    df["ftr_diff"]      = h_ftr  - a_ftr

    df["oliver_ff_composite"] = (
        FF_WEIGHTS["efg"]  *  df["efg_diff"]
        - FF_WEIGHTS["tov"]  *  df["tov_pct_diff"]
        + FF_WEIGHTS["oreb"] *  df["oreb_pct_diff"]
        + FF_WEIGHTS["ftr"]  *  df["ftr_diff"]
    )

    # --- Pythagorean Win% (Oliver formula) ---
    h_ortg = _safe_col(df, "home_team_adv_off_rating_r10")
    h_drtg = _safe_col(df, "home_team_adv_def_rating_r10")
    a_ortg = _safe_col(df, "away_team_adv_off_rating_r10")
    a_drtg = _safe_col(df, "away_team_adv_def_rating_r10")

    df["pyth_win_pct_home"] = _pyth_win_pct(h_ortg, h_drtg)
    df["pyth_win_pct_away"] = _pyth_win_pct(a_ortg, a_drtg)
    df["pyth_diff"]         = df["pyth_win_pct_home"] - df["pyth_win_pct_away"]

    return df


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load feature table and join odds. FROZEN."""
    df = pd.read_csv(DATA_MODEL_PATH, parse_dates=["date_x"])
    df = df.rename(columns={"date_x": "GAME_DATE"})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["GAME_ID_KEY"] = _canonical_game_id(df["GAME_ID"])

    # Load team columns from processed games
    if os.path.exists(PROCESSED_GAMES_PATH):
        games = pd.read_csv(
            PROCESSED_GAMES_PATH, usecols=["GAME_ID", "date", "home", "away"]
        )
        games["GAME_ID_KEY"] = _canonical_game_id(games["GAME_ID"])
        games["date"] = pd.to_datetime(games["date"]).dt.normalize()
        games = games.drop_duplicates("GAME_ID_KEY", keep="last")
        df = df.merge(
            games[["GAME_ID_KEY", "home", "away"]],
            on="GAME_ID_KEY", how="left",
        )

    # Load and join odds
    if os.path.exists(ODDS_PATH) and os.path.exists(PROCESSED_GAMES_PATH):
        odds = pd.read_csv(ODDS_PATH, parse_dates=["date"])
        odds["date"] = pd.to_datetime(odds["date"]).dt.normalize()
        odds["home"] = _normalize_team(odds["home"])
        odds["away"] = _normalize_team(odds["away"])

        # Collect the best available spread/moneyline prices
        home_price = pd.Series(np.nan, index=odds.index, dtype=float)
        away_price = pd.Series(np.nan, index=odds.index, dtype=float)
        for col in ["spread_odds_home", "home_spread_odds", "spread_home_odds", "moneyline_home"]:
            if col in odds.columns:
                v = pd.to_numeric(odds[col], errors="coerce")
                home_price = home_price.combine_first(v)
        for col in ["spread_odds_away", "away_spread_odds", "spread_away_odds", "moneyline_away"]:
            if col in odds.columns:
                v = pd.to_numeric(odds[col], errors="coerce")
                away_price = away_price.combine_first(v)

        odds["home_american"] = home_price
        odds["away_american"] = away_price
        odds_dedup = odds.drop_duplicates(["date", "home", "away"], keep="last")

        games2 = pd.read_csv(
            PROCESSED_GAMES_PATH, usecols=["GAME_ID", "date", "home", "away"]
        )
        games2["GAME_ID_KEY"] = _canonical_game_id(games2["GAME_ID"])
        games2["date"] = pd.to_datetime(games2["date"]).dt.normalize()

        odds_gid = odds_dedup.merge(
            games2, on=["date", "home", "away"], how="inner"
        )
        odds_gid["GAME_ID_KEY"] = _canonical_game_id(odds_gid["GAME_ID"])
        odds_gid = odds_gid.drop_duplicates("GAME_ID_KEY", keep="last")

        df = df.merge(
            odds_gid[["GAME_ID_KEY", "home_american", "away_american"]],
            on="GAME_ID_KEY", how="left",
        )
    else:
        df["home_american"] = np.nan
        df["away_american"] = np.nan

    # Mirror / default missing prices
    m_h = df["home_american"].isna() & df["away_american"].notna()
    m_a = df["away_american"].isna() & df["home_american"].notna()
    df.loc[m_h, "home_american"] = df.loc[m_h, "away_american"]
    df.loc[m_a, "away_american"] = df.loc[m_a, "home_american"]
    df["home_american"] = df["home_american"].fillna(DEFAULT_AMERICAN_ODDS)
    df["away_american"] = df["away_american"].fillna(DEFAULT_AMERICAN_ODDS)

    ph = _american_to_payout(df["home_american"])
    pa = _american_to_payout(df["away_american"])
    df["payout_home"] = ph.where(ph > 0, DEFAULT_PAYOUT).fillna(DEFAULT_PAYOUT)
    df["payout_away"] = pa.where(pa > 0, DEFAULT_PAYOUT).fillna(DEFAULT_PAYOUT)

    df = add_oliver_features(df)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


# ── Feature selection ─────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> list:
    """Select feature columns using Oliver-curated allowlist. FROZEN except EXTRA_FEATURE_EXCLUSIONS/INCLUSIONS."""
    # Dynamic columns: rolling team stats from nba-api (all windows/prefixes)
    dynamic = [
        c for c in df.columns
        if c.startswith(DYNAMIC_PREFIXES) and c.endswith(DYNAMIC_SUFFIXES)
    ]

    deny = DENY_EXACT | set(EXTRA_FEATURE_EXCLUSIONS)

    # Start from curated static allowlist + dynamic columns
    candidates = sorted(set(
        [c for c in STATIC_FEATURES if c in df.columns] + dynamic
    ))

    features = []
    for c in candidates:
        if c in deny:
            continue
        cl = c.lower()
        if any(tok in cl for tok in DENY_SUBSTRINGS):
            continue
        features.append(c)

    for c in EXTRA_FEATURE_INCLUSIONS:
        if c in df.columns and c not in features:
            features.append(c)

    return sorted(set(features))


# ── CV splits ────────────────────────────────────────────────────────────────

def walk_forward_splits(df, time_col, train_size, test_size):
    """Walk-forward generator. FROZEN."""
    df = df.sort_values(time_col).reset_index(drop=True)
    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > len(df):
            break
        yield df.iloc[start:train_end].copy(), df.iloc[train_end:test_end].copy()
        start += test_size


# ── ROI computation ───────────────────────────────────────────────────────────

def compute_roi(preds_df: pd.DataFrame) -> float:
    """Compound Kelly-bankroll ROI over all predictions. FROZEN."""
    df = preds_df.sort_values("GAME_DATE").reset_index(drop=True)

    edge = df["pred_ensemble"] - df["spread_signed"]
    sigma = df["sigma_ensemble"].replace(0, np.nan).fillna(1.0)
    win_prob = norm.cdf(edge / sigma)

    ph = df["payout_home"].clip(lower=0.01)
    pa = df["payout_away"].clip(lower=0.01)

    ev_home = win_prob * ph - (1 - win_prob)
    ev_away = (1 - win_prob) * pa - win_prob

    bet_side = np.where(
        ev_home > EV_THRESHOLD, "HOME",
        np.where(ev_away > EV_THRESHOLD, "AWAY", "NONE"),
    )

    kelly = np.zeros(len(df))
    hm = bet_side == "HOME"
    am = bet_side == "AWAY"
    kelly[hm] = np.clip(
        ((win_prob[hm] * (ph.values[hm] + 1) - 1) / ph.values[hm]) * FRACTIONAL_KELLY,
        0, MAX_KELLY,
    )
    kelly[am] = np.clip(
        (((1 - win_prob[am]) * (pa.values[am] + 1) - 1) / pa.values[am]) * FRACTIONAL_KELLY,
        0, MAX_KELLY,
    )

    df = df.copy()
    df["_kelly"] = kelly
    df["_bet_side"] = bet_side
    df["_edge_abs"] = edge.abs()

    is_bet = bet_side != "NONE"
    df["_rank"] = np.nan
    if is_bet.any():
        df.loc[is_bet, "_rank"] = (
            df.loc[is_bet]
            .groupby("GAME_DATE")["_edge_abs"]
            .rank(method="first", ascending=False)
        )
    selected = is_bet & (df["_rank"] <= TOP_N_BETS_PER_DAY)

    # Scale down if daily exposure exceeds cap
    if selected.any():
        daily_exp = (
            df.loc[selected, "_kelly"]
            .groupby(df.loc[selected, "GAME_DATE"])
            .transform("sum")
        )
        scale = (DAILY_MAX_EXPOSURE / daily_exp).clip(upper=1.0)
        scale = scale.reindex(df.index).fillna(1.0)
        df.loc[selected, "_kelly"] *= scale

    did_home_cover = df["home_margin"].values > -df["spread_signed"].values
    covered = np.where(did_home_cover, "HOME", "AWAY")

    pnl = np.zeros(len(df))
    sel_idx = df.index[selected]
    for i in sel_idx:
        side = df.at[i, "_bet_side"]
        k = df.at[i, "_kelly"]
        won = covered[i] == side
        if side == "HOME":
            pnl[i] = k * ph.iloc[i] if won else -k
        else:
            pnl[i] = k * pa.iloc[i] if won else -k

    bankroll = 1.0
    for p in pnl:
        bankroll *= (1.0 + p)

    return float(bankroll - 1.0)


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment() -> dict:
    """Train all models, compute ensemble, return metrics. FROZEN."""
    df = load_data()
    features = build_features(df)

    for col in features + ["home_margin", "spread_signed", "payout_home", "payout_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n = len(df)
    t_size = TRAIN_SIZE if n >= TRAIN_SIZE + TEST_SIZE else int(n * 0.70)
    v_size = TEST_SIZE  if n >= TRAIN_SIZE + TEST_SIZE else max(int(n * 0.15), 1)

    n_folds = sum(1 for _ in walk_forward_splits(df, "GAME_DATE", t_size, v_size))
    if n_folds == 0:
        raise RuntimeError(
            f"No walk-forward folds: dataset has {n} rows, need at least {t_size + v_size}."
        )
    print(f"[experiment] folds={n_folds}  features={len(features)}  rows={n}")

    pred_store  = {s["id"]: pd.Series(np.nan, index=df.index) for s in MODEL_SPECS}
    sigma_store = {s["id"]: pd.Series(np.nan, index=df.index) for s in MODEL_SPECS}
    rmse_by_model = {s["id"]: [] for s in MODEL_SPECS}

    for fold_i, (train_df, test_df) in enumerate(
        walk_forward_splits(df, "GAME_DATE", t_size, v_size), 1
    ):
        X_tr = train_df[features]
        y_tr = train_df["home_margin"]
        X_te = test_df[features]
        y_te = test_df["home_margin"]

        # Drop columns that are all-NaN in training
        valid_cols = X_tr.columns[X_tr.notna().any()].tolist()
        X_tr = X_tr[valid_cols]
        X_te = X_te[valid_cols]

        for spec in MODEL_SPECS:
            mid = spec["id"]
            model = deepcopy(spec["model"])
            Xtr_, Xte_ = X_tr, X_te
            if spec["needs_imputer"]:
                imp = SimpleImputer(strategy="median")
                Xtr_ = pd.DataFrame(
                    imp.fit_transform(X_tr), columns=valid_cols, index=X_tr.index
                )
                Xte_ = pd.DataFrame(
                    imp.transform(X_te), columns=valid_cols, index=X_te.index
                )
            with redirect_stderr(io.StringIO()):
                model.fit(Xtr_, y_tr)
                p_te = model.predict(Xte_)
                p_tr = model.predict(Xtr_)
            rmse_by_model[mid].append(float(np.sqrt(mean_squared_error(y_te, p_te))))
            pred_store[mid].loc[test_df.index] = p_te
            sigma_store[mid].loc[test_df.index] = float(np.nanstd(y_tr.values - p_tr)) or 1.0

        if fold_i % 5 == 0:
            print(f"[experiment] fold {fold_i}/{n_folds} done")

    # Collect all test-covered rows
    covered_mask = pd.concat(
        [pred_store[s["id"]] for s in MODEL_SPECS], axis=1
    ).notna().any(axis=1)
    out = df.loc[covered_mask, [
        "GAME_DATE", "home_margin", "spread_signed",
        "payout_home", "payout_away",
    ]].copy()

    # Compute per-model mean RMSE
    rmse_means = {
        mid: float(np.mean(v)) for mid, v in rmse_by_model.items() if v
    }

    # Ensemble weights
    if isinstance(ENSEMBLE_WEIGHTS, dict):
        raw_w = {k: float(v) for k, v in ENSEMBLE_WEIGHTS.items() if k in rmse_means}
    elif ENSEMBLE_WEIGHTS == "equal":
        raw_w = {mid: 1.0 for mid in rmse_means}
    else:  # inverse_rmse
        raw_w = {mid: 1.0 / v for mid, v in rmse_means.items() if v > 0}

    total_w = sum(raw_w.values())
    if total_w == 0:
        raise RuntimeError("All ensemble weights are zero.")
    weights = {k: v / total_w for k, v in raw_w.items()}

    out["pred_ensemble"] = sum(
        pred_store[mid].loc[out.index] * w for mid, w in weights.items()
    )
    out["sigma_ensemble"] = pd.concat(
        [sigma_store[mid].loc[out.index] for mid in weights], axis=1
    ).mean(axis=1)

    ensemble_rmse = float(
        np.sqrt(mean_squared_error(out["home_margin"], out["pred_ensemble"]))
    )
    mean_model_rmse = float(np.mean(list(rmse_means.values()))) if rmse_means else float("nan")
    roi = compute_roi(out)
    score = roi / (1.0 + ensemble_rmse)  # higher is better

    return {
        "ensemble_rmse": round(ensemble_rmse, 4),
        "mean_model_rmse": round(mean_model_rmse, 4),
        "roi": round(roi, 4),
        "score": round(score, 6),
        "n_folds": n_folds,
        "n_models": len(MODEL_SPECS),
        "train_size": t_size,
        "test_size": v_size,
        "n_features": len(features),
    }


# ── Results logging ───────────────────────────────────────────────────────────

def params_summary() -> str:
    return json.dumps(
        {
            "train_size": TRAIN_SIZE,
            "test_size": TEST_SIZE,
            "ev_threshold": EV_THRESHOLD,
            "fractional_kelly": FRACTIONAL_KELLY,
            "max_kelly": MAX_KELLY,
            "top_n_bets": TOP_N_BETS_PER_DAY,
            "daily_max_exposure": DAILY_MAX_EXPOSURE,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "models": [s["id"] for s in MODEL_SPECS],
            "extra_exclusions": EXTRA_FEATURE_EXCLUSIONS,
            "extra_inclusions": EXTRA_FEATURE_INCLUSIONS,
        },
        separators=(",", ":"),
    )


def write_results(metrics: dict, exp_id: str) -> None:
    """Append one row to results.tsv. FROZEN."""
    header_needed = not os.path.exists(RESULTS_TSV)
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp\texp_id\tensemble_rmse\troi\tscore\tparams\n")
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{exp_id}\t"
            f"{metrics['ensemble_rmse']}\t{metrics['roi']}\t"
            f"{metrics['score']}\t{params_summary()}\n"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    exp_id = hashlib.md5(params_summary().encode()).hexdigest()[:8]
    print(f"[experiment] id={exp_id}")

    t0 = time.time()
    metrics = run_experiment()
    elapsed = time.time() - t0

    print(f"[experiment] ensemble_rmse={metrics['ensemble_rmse']}")
    print(f"[experiment] roi={metrics['roi']}")
    print(f"[experiment] score={metrics['score']}")
    print(f"[experiment] elapsed={elapsed:.1f}s")

    write_results(metrics, exp_id)
    print(f"[experiment] wrote to {RESULTS_TSV}")
