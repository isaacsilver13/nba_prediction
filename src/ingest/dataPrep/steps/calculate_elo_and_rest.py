"""Compute Elo ratings, rest features, and streak metrics."""

from pathlib import Path

import pandas as pd
import numpy as np

try:
    # When imported as a package module
    from ..config import get_config, resolve_step_config
    from ..helpers import load_cached_output, save_output
except ImportError:
    # When loaded dynamically or as direct import
    try:
        from ingest.dataPrep.config import get_config, resolve_step_config
        from ingest.dataPrep.helpers import load_cached_output, save_output
    except ImportError:
        # Fallback for different import contexts
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config import get_config, resolve_step_config
        from helpers import load_cached_output, save_output


# ==============================
# TEAM-LEVEL ROLLING FEATURES
# ==============================
def add_team_rolling_features(df, team_col, score_col, prefix, rolling_window):
    """Add rolling points and margin averages for a team column.

    Parameters
    ----------
    df : pd.DataFrame
        Game-level dataframe with a `date` column.
    team_col : str
        Column name for team identifier (e.g., home/away).
    score_col : str
        Column name for points scored.
    prefix : str
        Prefix used for generated feature names.
    rolling_window : int
        Window size for rolling averages.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling feature columns appended.
    """
    df = df.sort_values([team_col, "date"])
    
    # Rolling average of points
    df[f"{prefix}_pts_last{rolling_window}"] = (
        df.groupby(team_col)[score_col]
        .rolling(rolling_window)
        .mean()
        .shift(1)  # avoid future leakage
        .reset_index(level=0, drop=True)
    )

    # Rolling average of margin
    df[f"{prefix}_margin_last{rolling_window}"] = (
        df.groupby(team_col)["margin"]
        .rolling(rolling_window)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    return df

# ==============================
# REST DAYS & BACK-TO-BACK (B2B) FEATURES
# ==============================
def add_rest_features(df, team_col, prefix):
    """Add rest-day counts and back-to-back flags per team.

    Parameters
    ----------
    df : pd.DataFrame
        Game-level dataframe with a `date` column.
    team_col : str
        Column name for team identifier (home/away).
    prefix : str
        Prefix used for generated feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with rest and B2B columns appended.
    """
    df = df.sort_values([team_col, "date"])
    prev_date = df.groupby(team_col)["date"].shift(1)
    
    # Number of days since previous game
    df[f"{prefix}_rest"] = (df["date"] - prev_date).dt.days
    
    # Back-to-back flag
    df[f"{prefix}_b2b"] = (df[f"{prefix}_rest"] == 1).astype(int)
    
    return df


def _compute_streaks(results):
    """Compute win and loss streak lengths from binary results.

    Parameters
    ----------
    results : array-like
        Sequence where 1 indicates win and 0 indicates loss.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Win streak, loss streak, and net streak length arrays.
    """
    win_streak = np.zeros(len(results), dtype=int)
    loss_streak = np.zeros(len(results), dtype=int)
    current_win = 0
    current_loss = 0
    for i, res in enumerate(results):
        win_streak[i] = current_win
        loss_streak[i] = current_loss
        if res == 1:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
    streak_length = win_streak - loss_streak
    return win_streak, loss_streak, streak_length


def _compute_recent_counts(dates, windows):
    """Count games in rolling lookback windows for each date.

    Parameters
    ----------
    dates : array-like
        Sorted array of numpy datetime64 values.
    windows : list[int]
        Window sizes in days.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping of window size to count array.
    """
    counts = {w: np.zeros(len(dates), dtype=int) for w in windows}
    for i in range(len(dates)):
        for w in windows:
            cutoff = dates[i] - np.timedelta64(w, "D")
            counts[w][i] = int(((dates < dates[i]) & (dates >= cutoff)).sum())
    return counts


def _compute_rest_weighted_load(dates, window=7):
    """Compute a weighted load metric using inverse rest days.

    Parameters
    ----------
    dates : array-like
        Sorted array of numpy datetime64 values.
    window : int, default=7
        Lookback window in days for the load calculation.

    Returns
    -------
    np.ndarray
        Weighted load per game.
    """
    out = np.zeros(len(dates))
    for i in range(len(dates)):
        deltas = (dates[i] - dates[:i]).astype("timedelta64[D]").astype(int)
        mask = (deltas > 0) & (deltas <= window)
        if mask.any():
            out[i] = (1 / deltas[mask]).sum()
    return out

def run(config=None, inputs=None):
    """Run step 2 to compute Elo, rest, and rolling features.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `games_with_ids` when provided.

    Returns
    -------
    dict
        Dictionary with `games_processed` dataframe.
    """
    cfg = resolve_step_config(get_config(), "step2") if config is None else resolve_step_config(config, "step2")
    paths = cfg["paths"]
    elo_cfg = cfg["elo"]
    rolling_cfg = cfg["rolling"]
    cache_cfg = cfg.get("cache", {})

    input_path = Path(paths["input_file"])
    output_path = Path(paths["output_file"])

    # Try to load from cache
    cached_df, from_cache = load_cached_output(str(output_path), cache_cfg)
    if from_cache and not inputs:
        cached_df["date"] = pd.to_datetime(cached_df["date"])  # Ensure date is parsed
        return {"games_processed": cached_df}

    if inputs and "games_with_ids" in inputs:
        df = inputs["games_with_ids"].copy()
    else:
        df = pd.read_csv(input_path, parse_dates=["date"])

    df = df[df["id_spread"].isin([0, 1])]

    df["home_win"] = (df["score_home"] > df["score_away"]).astype(int)
    df["margin"] = df["score_home"] - df["score_away"]
    df["total_points"] = df["score_home"] + df["score_away"]

    df["is_home_favorite"] = (df["whos_favored"] == "home").astype(int)

    df["spread_signed"] = np.where(
        df["is_home_favorite"] == 1,
        -df["spread"],
        df["spread"],
    )

    rolling_window = rolling_cfg.get("window", 5)
    df = add_team_rolling_features(df, "home", "score_home", prefix="home", rolling_window=rolling_window)
    df = add_team_rolling_features(df, "away", "score_away", prefix="away", rolling_window=rolling_window)

    df = add_rest_features(df, "home", "home")
    df = add_rest_features(df, "away", "away")

    df[f"pts_diff_last{rolling_window}"] = df[f"home_pts_last{rolling_window}"] - df[f"away_pts_last{rolling_window}"]
    df[f"margin_diff_last{rolling_window}"] = (
        df[f"home_margin_last{rolling_window}"] - df[f"away_margin_last{rolling_window}"]
    )
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    df = df.dropna(
        subset=[
            f"home_pts_last{rolling_window}",
            f"away_pts_last{rolling_window}",
            f"home_margin_last{rolling_window}",
            f"away_margin_last{rolling_window}",
        ]
    )

    if "ot_home" not in df.columns:
        df["ot_home"] = 0
    if "ot_away" not in df.columns:
        df["ot_away"] = 0

    df["home_margin"] = df["score_home"] - df["score_away"]
    df["away_margin"] = -df["home_margin"]

    long_team = pd.concat(
        [
            df[["GAME_ID", "date", "season", "home", "home_margin", "spread_signed", "ot_home"]]
            .rename(columns={
                "home": "team",
                "home_margin": "margin",
                "ot_home": "ot_flag",
            })
            .assign(is_home=1, team_spread=lambda x: x["spread_signed"]),
            df[["GAME_ID", "date", "season", "away", "away_margin", "spread_signed", "ot_away"]]
            .rename(columns={
                "away": "team",
                "away_margin": "margin",
                "ot_away": "ot_flag",
            })
            .assign(is_home=0, team_spread=lambda x: -x["spread_signed"]),
        ],
        ignore_index=True,
    )

    long_team = long_team.sort_values(["team", "date"]).reset_index(drop=True)
    long_team["win"] = (long_team["margin"] > 0).astype(int)
    long_team["ats_cover"] = (long_team["margin"] > long_team["team_spread"]).astype(int)
    long_team["blowout_flag"] = (long_team["margin"].abs() >= 15).astype(int)

    recent_windows = [3, 5, 7]
    ats_window = 20
    blowout_window = 20

    features = []
    for team, g in long_team.groupby("team", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        win_streak, loss_streak, streak_length = _compute_streaks(g["win"].values)
        recent_counts = _compute_recent_counts(g["date"].values, recent_windows)
        rest_weighted = _compute_rest_weighted_load(g["date"].values, window=7)

        g["win_streak"] = win_streak
        g["loss_streak"] = loss_streak
        g["streak_length"] = streak_length
        g["streak_length_squared"] = streak_length ** 2
        g["after_ot_flag"] = g["ot_flag"].shift(1).fillna(0)

        for w in recent_windows:
            g[f"games_last_{w}_days"] = recent_counts[w]

        g["rest_weighted_load"] = rest_weighted

        g["margin_std_last_10"] = (
            g["margin"].shift(1).rolling(10, min_periods=3).std().fillna(0)
        )

        g["blowout_pct_last20"] = (
            g["blowout_flag"].shift(1).rolling(blowout_window, min_periods=5).mean().fillna(0)
        )
        g["blowout_pct_season"] = (
            g["blowout_flag"].expanding().mean().shift(1).fillna(0)
        )

        g["ats_roll_last20"] = (
            g["ats_cover"].shift(1).rolling(ats_window, min_periods=5).mean().fillna(0)
        )

        features.append(g[[
            "GAME_ID",
            "team",
            "win_streak",
            "loss_streak",
            "streak_length",
            "streak_length_squared",
            "after_ot_flag",
            "games_last_3_days",
            "games_last_5_days",
            "games_last_7_days",
            "rest_weighted_load",
            "margin_std_last_10",
            "blowout_pct_last20",
            "blowout_pct_season",
            "ats_roll_last20",
        ]])

    team_features = pd.concat(features, ignore_index=True)

    home_feats = team_features.rename(columns={
        "team": "home",
        "win_streak": "home_win_streak",
        "loss_streak": "home_loss_streak",
        "streak_length": "home_streak_length",
        "streak_length_squared": "home_streak_length_squared",
        "after_ot_flag": "home_after_ot_flag",
        "games_last_3_days": "home_games_last_3_days",
        "games_last_5_days": "home_games_last_5_days",
        "games_last_7_days": "home_games_last_7_days",
        "rest_weighted_load": "home_rest_weighted_load",
        "margin_std_last_10": "home_margin_std_last_10",
        "blowout_pct_last20": "home_blowout_pct_last20",
        "blowout_pct_season": "home_blowout_pct_season",
        "ats_roll_last20": "home_ats_roll_last20",
    })

    away_feats = team_features.rename(columns={
        "team": "away",
        "win_streak": "away_win_streak",
        "loss_streak": "away_loss_streak",
        "streak_length": "away_streak_length",
        "streak_length_squared": "away_streak_length_squared",
        "after_ot_flag": "away_after_ot_flag",
        "games_last_3_days": "away_games_last_3_days",
        "games_last_5_days": "away_games_last_5_days",
        "games_last_7_days": "away_games_last_7_days",
        "rest_weighted_load": "away_rest_weighted_load",
        "margin_std_last_10": "away_margin_std_last_10",
        "blowout_pct_last20": "away_blowout_pct_last20",
        "blowout_pct_season": "away_blowout_pct_season",
        "ats_roll_last20": "away_ats_roll_last20",
    })

    df = df.merge(home_feats, on=["GAME_ID", "home"], how="left")
    df = df.merge(away_feats, on=["GAME_ID", "away"], how="left")

    df["win_streak"] = df["home_win_streak"].fillna(0) - df["away_win_streak"].fillna(0)
    df["loss_streak"] = df["home_loss_streak"].fillna(0) - df["away_loss_streak"].fillna(0)
    df["streak_length_squared"] = (
        df["home_streak_length"].fillna(0) - df["away_streak_length"].fillna(0)
    ) ** 2

    df["after_ot_flag"] = df["home_after_ot_flag"].fillna(0) - df["away_after_ot_flag"].fillna(0)

    df["games_last_3_days"] = df["home_games_last_3_days"].fillna(0) - df["away_games_last_3_days"].fillna(0)
    df["games_last_5_days"] = df["home_games_last_5_days"].fillna(0) - df["away_games_last_5_days"].fillna(0)
    df["games_last_7_days"] = df["home_games_last_7_days"].fillna(0) - df["away_games_last_7_days"].fillna(0)
    df["rest_weighted_load"] = df["home_rest_weighted_load"].fillna(0) - df["away_rest_weighted_load"].fillna(0)

    df["margin_std_last_10"] = df["home_margin_std_last_10"].fillna(0) - df["away_margin_std_last_10"].fillna(0)

    df["blowout_pct_last20"] = df["home_blowout_pct_last20"].fillna(0) - df["away_blowout_pct_last20"].fillna(0)
    df["blowout_pct_season"] = df["home_blowout_pct_season"].fillna(0) - df["away_blowout_pct_season"].fillna(0)

    df["ats_roll_last20"] = df["home_ats_roll_last20"].fillna(0) - df["away_ats_roll_last20"].fillna(0)

    feature_cols = [
        "season",
        "date",
        "GAME_ID",
        "home",
        "away",
        "regular",
        "score_away",
        "score_home",
        "spread",
        "spread_signed",
        "is_home_favorite",
        f"home_pts_last{rolling_window}",
        f"away_pts_last{rolling_window}",
        f"home_margin_last{rolling_window}",
        f"away_margin_last{rolling_window}",
        "home_rest",
        "away_rest",
        "home_b2b",
        "away_b2b",
        f"pts_diff_last{rolling_window}",
        f"margin_diff_last{rolling_window}",
        "rest_diff",
        "home_win",
        "ot_home",
        "ot_away",
        "home_win_streak",
        "away_win_streak",
        "home_loss_streak",
        "away_loss_streak",
        "home_streak_length",
        "away_streak_length",
        "home_streak_length_squared",
        "away_streak_length_squared",
        "home_after_ot_flag",
        "away_after_ot_flag",
        "home_games_last_3_days",
        "away_games_last_3_days",
        "home_games_last_5_days",
        "away_games_last_5_days",
        "home_games_last_7_days",
        "away_games_last_7_days",
        "home_rest_weighted_load",
        "away_rest_weighted_load",
        "home_margin_std_last_10",
        "away_margin_std_last_10",
        "home_blowout_pct_last20",
        "away_blowout_pct_last20",
        "home_blowout_pct_season",
        "away_blowout_pct_season",
        "home_ats_roll_last20",
        "away_ats_roll_last20",
        "win_streak",
        "loss_streak",
        "streak_length_squared",
        "after_ot_flag",
        "games_last_3_days",
        "games_last_5_days",
        "games_last_7_days",
        "rest_weighted_load",
        "margin_std_last_10",
        "blowout_pct_last20",
        "blowout_pct_season",
        "ats_roll_last20",
    ]
    df = df[feature_cols].reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    df["home_margin"] = df["score_home"] - df["score_away"]
    df["away_margin"] = df["score_away"] - df["score_home"]

    df["favorite_cover_manual"] = np.where(
        df["is_home_favorite"] == 1,
        (df["score_home"] - df["score_away"]) > df["spread"],
        (df["score_away"] - df["score_home"]) > df["spread"],
    ).astype(int)

    df = df[df["home_margin"].abs() != df["spread"]].copy()

    teams = pd.unique(df[["home", "away"]].values.ravel())
    elo = {team: elo_cfg.get("start", 1500) for team in teams}

    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        home = row["home"]
        away = row["away"]

        elo_home = elo[home] + elo_cfg.get("home_advantage", 65)
        elo_away = elo[away]

        elo_home_list.append(elo_home)
        elo_away_list.append(elo_away)

        exp_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
        exp_away = 1 - exp_home

        act_home, act_away = (1, 0) if row["score_home"] > row["score_away"] else (0, 1)

        margin = abs(row["score_home"] - row["score_away"])
        mult = np.log(margin + 1) * (2.2 / ((elo_home - elo_away) * 0.001 + 2.2))

        elo[home] += elo_cfg.get("k", 20) * mult * (act_home - exp_home)
        elo[away] += elo_cfg.get("k", 20) * mult * (act_away - exp_away)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["home_flag"] = 1

    df = df.sort_values("date").reset_index(drop=True)

    df["elo_diff"] = np.where(
        df["is_home_favorite"] == 1,
        df["elo_home"] - df["elo_away"],
        df["elo_away"] - df["elo_home"],
    )

    df["home_margin"] = df["score_home"] - df["score_away"]
    df["away_margin"] = -df["home_margin"]

    long = pd.concat(
        [
            df[["date", "home", "home_margin", "elo_home"]].rename(
                columns={"home": "team", "home_margin": "margin", "elo_home": "ELO"}
            ),
            df[["date", "away", "away_margin", "elo_away"]].rename(
                columns={"away": "team", "away_margin": "margin", "elo_away": "ELO"}
            ),
        ],
        ignore_index=True,
    )

    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    margin_short = rolling_cfg.get("margin_short", 5)
    margin_long = rolling_cfg.get("margin_long", 10)
    min_short = rolling_cfg.get("margin_short_min_periods", 3)
    min_long = rolling_cfg.get("margin_long_min_periods", 5)
    elo_roll = rolling_cfg.get("elo_rolling_window", 164)

    long["rolling_margin_5"] = (
        long.groupby("team")["margin"].transform(
            lambda x: x.shift(1).rolling(window=margin_short, min_periods=min_short).mean()
        )
    )

    long["rolling_margin_10"] = (
        long.groupby("team")["margin"].transform(
            lambda x: x.shift(1).rolling(margin_long, min_periods=min_long).mean()
        )
    )

    long["ELO_Rolling_2YR"] = (
        long.groupby("team")["ELO"].transform(lambda x: x.shift(1).rolling(elo_roll).mean())
    )

    for col in ["rolling_margin_5", "rolling_margin_10"]:
        long[col] = long[col].clip(-20, 20)

    df = df.merge(
        long[["date", "team", "rolling_margin_5", "rolling_margin_10", "ELO_Rolling_2YR"]],
        left_on=["date", "home"],
        right_on=["date", "team"],
        how="left",
    ).rename(
        columns={
            "rolling_margin_5": "home_roll_5",
            "rolling_margin_10": "home_roll_10",
            "ELO_Rolling_2YR": "home_elo_roll_2Y",
        }
    ).drop(columns="team")

    df = df.merge(
        long[["date", "team", "rolling_margin_5", "rolling_margin_10", "ELO_Rolling_2YR"]],
        left_on=["date", "away"],
        right_on=["date", "team"],
        how="left",
    ).rename(
        columns={
            "rolling_margin_5": "away_roll_5",
            "rolling_margin_10": "away_roll_10",
            "ELO_Rolling_2YR": "away_elo_roll_2Y",
        }
    ).drop(columns="team")

    df["rolling_margin_diff_5"] = np.where(
        df["is_home_favorite"] == 1,
        df["home_roll_5"] - df["away_roll_5"],
        df["away_roll_5"] - df["home_roll_5"],
    ).clip(-20, 20)

    df["rolling_margin_diff_10"] = np.where(
        df["is_home_favorite"] == 1,
        df["home_roll_10"] - df["away_roll_10"],
        df["away_roll_10"] - df["home_roll_10"],
    ).clip(-15, 15)

    df["last_game_home"] = df.groupby("home")["date"].shift(1)
    df["last_game_away"] = df.groupby("away")["date"].shift(1)

    df["home_rest"] = (df["date"] - df["last_game_home"]).dt.days.clip(0, 5)
    df["away_rest"] = (df["date"] - df["last_game_away"]).dt.days.clip(0, 5)

    df["rest_advantage"] = (df["home_rest"] - df["away_rest"]).clip(-3, 3)
    df["rest_advantage_sq"] = np.sign(df["rest_advantage"]) * (df["rest_advantage"] ** 2)

    df["favorite_margin"] = np.where(
        df["is_home_favorite"] == 1,
        df["score_home"] - df["score_away"],
        df["score_away"] - df["score_home"],
    )

    df["market_error"] = df["favorite_margin"] - df["spread"]
    df["favorite_cover"] = (df["favorite_margin"] > df["spread"]).astype(int)

    df["is_favorite"] = 1
    df["favorite_is_home"] = df["is_home_favorite"]

    df["abs_elo_diff"] = df["elo_diff"].abs()
    df["elo_diff_squared"] = df["elo_diff"] ** 2
    df["sigmoid_elo_diff"] = 1 / (1 + np.exp(-df["elo_diff"] / 100))

    # Save to cache if enabled
    save_output(df, str(output_path), cache_cfg)

    return {"games_processed": df}


if __name__ == "__main__":
    run(get_config())