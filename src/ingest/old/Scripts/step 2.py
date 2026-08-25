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



INPUT_PATH_GAMES = "data/processed/nba_games_with_game_id.csv"
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

df_games = pd.read_csv(INPUT_PATH_GAMES, parse_dates=["date"])
df_players = pd.read_csv(INPUT_PATH_BOXSCORES, parse_dates=["date"])

missing_dates = sorted(
    set(df_games["date"]) - set(df_players["date"])
)


print(f"📅 Fetching scoreboard for {len(missing_dates)} dates")
sys.exit()

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
