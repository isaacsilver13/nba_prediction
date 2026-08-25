import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import norm
import time
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.library.parameters import LeagueID
from pathlib import Path
from nba_api.stats.library.http import NBAStatsHTTP

# ==============================
# CONFIGURATION
# ==============================
INPUT_FILE = "data/raw/kaggle_nba_2008-2025.csv"  # Raw Kaggle NBA game data
OUTPUT_PATH = "data/processed/nba_games_with_elo.csv"
ELO_START = 1500           # Initial Elo rating for all teams
ELO_K = 20                 # Elo K-factor (adjusts sensitivity)
HOME_ADVANTAGE = 65        # Elo boost for home team
ROLLING_WINDOW = 5         # Window for rolling average calculations

# ==============================
# LOAD AND PREPARE DATA
# ==============================
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Keep only regular season games
df = df[df["regular"] == True].copy()
# Focus on recent seasons (2020+)
df = df[df["season"] >= 2020]

# Sort chronologically for rolling calculations
df = df.sort_values("date").reset_index(drop=True)

# ==============================
# CREATE BASIC TARGETS / LABELS
# ==============================
df["home_win"] = (df["score_home"] > df["score_away"]).astype(int)
df["margin"] = df["score_home"] - df["score_away"]
df["total_points"] = df["score_home"] + df["score_away"]

# Notes: 
# - id_spread and id_total already provided in raw data
# - id_spread: 1=favorite covered, 0=underdog covered, 2=push
# - id_total: 1=over, 0=under, 2=push

# ==============================
# SPREAD FEATURES
# ==============================
# Flag if home team is the betting favorite
df["is_home_favorite"] = (df["whos_favored"] == "home").astype(int)

# Signed spread from home team's perspective
df["spread_signed"] = df["spread"] * df["is_home_favorite"].replace({0: -1})

# ==============================
# TEAM-LEVEL ROLLING FEATURES
# ==============================
def add_team_rolling_features(df, team_col, score_col, prefix):
    """
    Adds rolling average features for points scored and margins.

    Parameters:
        df: DataFrame with game-level stats
        team_col: column name for team identifier (home/away)
        score_col: column name for points scored
        prefix: string prefix for feature names

    Returns:
        DataFrame with new rolling columns
    """
    df = df.sort_values([team_col, "date"])
    
    # Rolling average of points
    df[f"{prefix}_pts_last{ROLLING_WINDOW}"] = (
        df.groupby(team_col)[score_col]
        .rolling(ROLLING_WINDOW)
        .mean()
        .shift(1)  # avoid future leakage
        .reset_index(level=0, drop=True)
    )

    # Rolling average of margin
    df[f"{prefix}_margin_last{ROLLING_WINDOW}"] = (
        df.groupby(team_col)["margin"]
        .rolling(ROLLING_WINDOW)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    return df

# Add home & away rolling features
df = add_team_rolling_features(df, "home", "score_home", prefix="home")
df = add_team_rolling_features(df, "away", "score_away", prefix="away")

# ==============================
# REST DAYS & BACK-TO-BACK (B2B) FEATURES
# ==============================
def add_rest_features(df, team_col, prefix):
    """
    Adds rest and back-to-back game flags per team.

    Parameters:
        df: DataFrame with games
        team_col: home/away identifier
        prefix: string prefix for feature names

    Returns:
        DataFrame with rest and B2B columns
    """
    df = df.sort_values([team_col, "date"])
    prev_date = df.groupby(team_col)["date"].shift(1)
    
    # Number of days since previous game
    df[f"{prefix}_rest"] = (df["date"] - prev_date).dt.days
    
    # Back-to-back flag
    df[f"{prefix}_b2b"] = (df[f"{prefix}_rest"] == 1).astype(int)
    
    return df

df = add_rest_features(df, "home", "home")
df = add_rest_features(df, "away", "away")

# ==============================
# DIFFERENTIAL FEATURES
# ==============================
# Difference in rolling points, margins, and rest days
df["pts_diff_last5"] = df["home_pts_last5"] - df["away_pts_last5"]
df["margin_diff_last5"] = df["home_margin_last5"] - df["away_margin_last5"]
df["rest_diff"] = df["home_rest"] - df["away_rest"]

# ==============================
# CLEANUP
# ==============================
# Drop rows with insufficient rolling history
df = df.dropna(subset=[
    "home_pts_last5", "away_pts_last5",
    "home_margin_last5", "away_margin_last5"
])

# Drop pushes for ATS and totals
df = df[df["id_spread"] != 2]
df = df[df["id_total"] != 2]

# ==============================
# FINAL FEATURE SET
# ==============================
FEATURE_COLS = [
    "season", "date",
    "home", "away", "regular", "score_away", "score_home",
    "spread", "spread_signed", "is_home_favorite",
    "home_pts_last5", "away_pts_last5",
    "home_margin_last5", "away_margin_last5",
    "home_rest", "away_rest",
    "home_b2b", "away_b2b",
    "pts_diff_last5", "margin_diff_last5", "rest_diff",
    "home_win", "id_spread", "id_total"
]
df = df[FEATURE_COLS].reset_index(drop=True)
df = df.sort_values("date").reset_index(drop=True)

# ==============================
# BASIC GAME FEATURES
# ==============================
df["home_margin"] = df["score_home"] - df["score_away"]
df["away_margin"] = df["score_away"] - df["score_home"]

# Manually check if favorite covered the spread
df["favorite_cover_manual"] = np.where(
    df["is_home_favorite"] == 1,
    (df["score_home"] - df["score_away"]) > df["spread"],
    (df["score_away"] - df["score_home"]) > df["spread"]
).astype(int)

# Drop pushes
df = df[df["home_margin"].abs() != df["spread"]].copy()

# ==============================
# ELO RATINGS
# ==============================
# Initialize Elo dictionary for all teams
teams = pd.unique(df[["home", "away"]].values.ravel())
elo = {team: ELO_START for team in teams}

elo_home_list = []
elo_away_list = []

# Loop through games to update Elo ratings
for _, row in df.iterrows():
    home = row["home"]
    away = row["away"]

    elo_home = elo[home] + HOME_ADVANTAGE
    elo_away = elo[away]

    elo_home_list.append(elo_home)
    elo_away_list.append(elo_away)

    # Expected outcome based on Elo
    exp_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    exp_away = 1 - exp_home

    # Actual outcome
    act_home, act_away = (1, 0) if row["score_home"] > row["score_away"] else (0, 1)

    # Margin-of-victory multiplier
    margin = abs(row["score_home"] - row["score_away"])
    mult = np.log(margin + 1) * (2.2 / ((elo_home - elo_away) * 0.001 + 2.2))

    # Update Elo ratings
    elo[home] += ELO_K * mult * (act_home - exp_home)
    elo[away] += ELO_K * mult * (act_away - exp_away)

# Attach Elo columns
df["elo_home"] = elo_home_list
df["elo_away"] = elo_away_list
df["home_flag"] = 1  # Always 1 for home team

# ==============================
# FINAL DATASET PREP
# ==============================
final_cols = [
    "date", "season", "home", "away",
    "score_home", "score_away",
    "spread", "is_home_favorite",
    "home_margin", "away_margin",
    "favorite_cover_manual",
    "elo_home", "elo_away",
    "home_flag"
]
df_elo = df[final_cols].copy()
df = df_elo.sort_values("date").reset_index(drop=True)

# Compute Elo difference from favorite perspective
df["elo_diff"] = np.where(
    df["is_home_favorite"] == 1,
    df["elo_home"] - df["elo_away"],
    df["elo_away"] - df["elo_home"]
)

# Update margin columns for consistency
df["home_margin"] = df["score_home"] - df["score_away"]
df["away_margin"] = -df["home_margin"]

# ==============================
# BUILD LONG-FORM TEAM TABLE
# ==============================
long = pd.concat(
    [
        df[["date", "home", "home_margin", "elo_home"]]
        .rename(columns={"home": "team", "home_margin": "margin", "elo_home": "ELO"}),

        df[["date", "away", "away_margin", "elo_away"]]
        .rename(columns={"away": "team", "away_margin": "margin", "elo_away": "ELO"})
    ],
    ignore_index=True
)

# Sort for rolling calculations
long = long.sort_values(["team", "date"]).reset_index(drop=True)

# -----------------------------
# ROLLING STATISTICS
# -----------------------------
# 5-game rolling average margin
long["rolling_margin_5"] = (
    long.groupby("team")["margin"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=3).mean())
)

# 10-game rolling average margin
long["rolling_margin_10"] = (
    long.groupby("team")["margin"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
)

# 2-year rolling Elo
long["ELO_Rolling_2YR"] = (
    long.groupby("team")["ELO"]
    .transform(lambda x: x.shift(1).rolling(164).mean())
)

# Clip extreme margins
for col in ["rolling_margin_5", "rolling_margin_10"]:
    long[col] = long[col].clip(-20, 20)

# Merge back into main dataframe for home & away teams
df = df.merge(
    long[["date", "team", "rolling_margin_5", "rolling_margin_10", "ELO_Rolling_2YR"]],
    left_on=["date", "home"],
    right_on=["date", "team"],
    how="left"
).rename(columns={"rolling_margin_5": "home_roll_5", 
                  "rolling_margin_10": "home_roll_10",
                  "ELO_Rolling_2YR": "home_elo_roll_2Y"}).drop(columns="team")

df = df.merge(
    long[["date", "team", "rolling_margin_5", "rolling_margin_10", "ELO_Rolling_2YR"]],
    left_on=["date", "away"],
    right_on=["date", "team"],
    how="left"
).rename(columns={"rolling_margin_5": "away_roll_5", 
                  "rolling_margin_10": "away_roll_10",
                  "ELO_Rolling_2YR": "away_elo_roll_2Y"}).drop(columns="team")

# Rolling differentials
df["rolling_margin_diff_5"] = np.where(
    df["is_home_favorite"] == 1,
    df["home_roll_5"] - df["away_roll_5"],
    df["away_roll_5"] - df["home_roll_5"]
).clip(-20, 20)

df["rolling_margin_diff_10"] = np.where(
    df["is_home_favorite"] == 1,
    df["home_roll_10"] - df["away_roll_10"],
    df["away_roll_10"] - df["home_roll_10"]
).clip(-15, 15)

# ==============================
# REST ADVANTAGE
# ==============================
df["last_game_home"] = df.groupby("home")["date"].shift(1)
df["last_game_away"] = df.groupby("away")["date"].shift(1)

df["home_rest"] = (df["date"] - df["last_game_home"]).dt.days.clip(0, 5)
df["away_rest"] = (df["date"] - df["last_game_away"]).dt.days.clip(0, 5)

df["rest_advantage"] = (df["home_rest"] - df["away_rest"]).clip(-3, 3)
df["rest_advantage_sq"] = np.sign(df["rest_advantage"]) * (df["rest_advantage"] ** 2)

# ==============================
# MARKET / SPREAD ERRORS
# ==============================
df["favorite_margin"] = np.where(
    df["is_home_favorite"] == 1,
    df["score_home"] - df["score_away"],
    df["score_away"] - df["score_home"]
)

df["market_error"] = df["favorite_margin"] - df["spread"]  # difference vs Vegas line
df["favorite_cover"] = (df["favorite_margin"] > df["spread"]).astype(int)

df["is_favorite"] = 1
df["favorite_is_home"] = df["is_home_favorite"]

# Map team abbreviations to NBA standard
TEAM_MAP = {
    "atl": "ATL","bkn": "BKN","bos": "BOS","cha": "CHA","chi": "CHI","cle": "CLE","dal": "DAL",
    "den": "DEN","det": "DET","gs": "GSW","hou": "HOU","ind": "IND","lac": "LAC","lal": "LAL",
    "mem": "MEM","mia": "MIA","mil": "MIL","min": "MIN","no": "NOP","ny": "NYK","okc": "OKC",
    "orl": "ORL","phi": "PHI","phx": "PHX","por": "POR","sa": "SAS","sac": "SAC","tor": "TOR",
    "utah": "UTA","wsh": "WAS"
}

df["home"] = df["home"].str.lower().map(TEAM_MAP)
df["away"] = df["away"].str.lower().map(TEAM_MAP)
df["date"] = df["date"].dt.normalize()


INPUT_PATH_BOXSCORES = "data/processed/all_boxscores.csv"

# ==============================
# NBA API SETUP (FOR GAME IDs)
# ==============================
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

# Error logging path
ERROR_LOG_PATH = Path("data/logs/scoreboard_failures.csv")
ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

REQUEST_SLEEP = 0.25   # polite delay between API calls

# -------------------------
# LOAD OR INIT ERROR LOG
# -------------------------
if ERROR_LOG_PATH.exists():
    error_log = pd.read_csv(ERROR_LOG_PATH, parse_dates=["date"])
else:
    error_log = pd.DataFrame(columns=["date", "error"])

# ==============================
# FUNCTIONS FOR NBA API
# ==============================
def parse_gamecode(gamecode):
    """
    Parse NBA API gamecode into date, away team, home team.
    """
    date_part, teams = gamecode.split("/")
    away = teams[:3]
    home = teams[3:]
    return date_part, away, home

def fetch_games_for_date(date, max_retries=5):
    """
    Fetch NBA games for a given date using ScoreboardV2.

    Returns a DataFrame with game details (GAME_ID, date, teams, season, arena).
    """
    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
    
    for attempt in range(max_retries):
        try:
            sb = ScoreboardV2(
                game_date=date_str,
                league_id=LeagueID.default,
                timeout=40,
                headers=HEADERS
            )
            games = sb.get_data_frames()[0]
            games = games.rename(columns={"GAME_DATE_EST": "GAME_DATE"})
            games[["GAME_DATE_STR", "AWAY", "HOME"]] = (
                games["GAMECODE"].apply(lambda x: pd.Series(parse_gamecode(x)))
            )
            games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE_STR"], format="%Y%m%d")
            return games[['GAME_ID', 'GAME_DATE', 'GAMECODE', 'HOME', 'AWAY', 'SEASON', 'ARENA_NAME']]
        
        except Exception as e:
            wait = 2 ** attempt
            NBAStatsHTTP._session = None
            print(f"[Retry {attempt + 1}/{max_retries}] {date_str} – waiting {wait}s")
            time.sleep(wait)

    print(f"[FAILED – EMPTY RETURN] {date_str}")
    return pd.DataFrame(columns=["GAME_ID","GAME_DATE","GAMECODE","HOME","AWAY","SEASON","ARENA_NAME"])

# ==============================
# MERGE EXISTING GAME DATA WITH NBA API GAME IDs
# ==============================

df_games = df.copy()

unique_dates = set(df_games["date"])



print(f"📅 Fetching scoreboard for {len(unique_dates)} dates")


all_games = []

for d in unique_dates:
    daily_games = fetch_games_for_date(d)
    if not daily_games.empty:
        all_games.append(daily_games)
    time.sleep(REQUEST_SLEEP)

# Save error log for retry later
error_log.drop_duplicates("date").to_csv(ERROR_LOG_PATH, index=False)

games_lookup = pd.concat(all_games, ignore_index=True)
print(f"✅ Collected {len(games_lookup):,} games")
print(f"⚠️ Failed dates logged: {error_log['date'].nunique()}")

# Merge NBA API game IDs into existing dataset
games_lookup["GAME_DATE"] = pd.to_datetime(games_lookup["GAME_DATE"]).dt.normalize()
df_games = df_games.merge(
    games_lookup,
    left_on=["date", "home", "away"],
    right_on=["GAME_DATE", "HOME", "AWAY"],
    how="left"
)

missing = df_games["GAME_ID"].isna().sum()
print(f"🔍 Missing GAME_IDs after merge: {missing}")

df_games.to_csv("data/processed/nba_games_with_game_id.csv", index=False)
print("💾 Saved games with GAME_ID")

# Retry failed dates
failed_dates = pd.read_csv(ERROR_LOG_PATH, parse_dates=["date"])["date"].unique()
retry_games = []

for d in failed_dates:
    daily_games = fetch_games_for_date(d)
    if not daily_games.empty:
        retry_games.append(daily_games)
    time.sleep(REQUEST_SLEEP)

retry_df = pd.concat(retry_games, ignore_index=True)
retry_df.to_csv("data/processed/scoreboard_retry_games.csv", index=False)


