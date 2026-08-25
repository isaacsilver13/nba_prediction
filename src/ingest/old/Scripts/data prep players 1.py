#####this gets game ids from nba api and merges them with the game data already existing, 
# so that we can add player data by game.


import pandas as pd
import time
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.library.parameters import LeagueID
from pathlib import Path

HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

# -------------------------
# CONFIG
# -------------------------
ERROR_LOG_PATH = Path("data/logs/scoreboard_failures.csv")
ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

REQUEST_SLEEP = 0.25   # be polite to NBA servers


# -------------------------
# LOAD / INIT ERROR LOG
# -------------------------
if ERROR_LOG_PATH.exists():
    error_log = pd.read_csv(ERROR_LOG_PATH, parse_dates=["date"])
else:
    error_log = pd.DataFrame(columns=["date", "error"])


# -------------------------
# FUNCTION
# -------------------------

def parse_gamecode(gamecode):
    date_part, teams = gamecode.split("/")
    away = teams[:3]
    home = teams[3:]
    return date_part, away, home

def fetch_games_for_date(date, max_retries = 5):
    """
    Fetch NBA games for a single date using ScoreboardV2.

    Returns:
        DataFrame with columns:
        ['GAME_ID', 'GAME_DATE', 'GAMECODE', 'SEASON', 'ARENA_NAME']
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
            print(f"Successfully pulled data for {date_str}")
            games[["GAME_DATE_STR", "AWAY", "HOME"]] = (
            games["GAMECODE"]
            .apply(lambda x: pd.Series(parse_gamecode(x))))
            games["GAME_DATE"] = pd.to_datetime(
            games["GAME_DATE_STR"], format="%Y%m%d")
        
            return games[['GAME_ID', 'GAME_DATE', 'GAMECODE',
                      'HOME', 'AWAY','SEASON', 'ARENA_NAME']]
            

        except Exception as e:
            wait = 2 ** attempt
            print(f"[Retry {attempt + 1}/{max_retries}] {date_str} – waiting {wait}s")
            time.sleep(wait)

    # ----- FINAL FAILURE FALLBACK -----
    print(f"[FAILED – EMPTY RETURN] {date_str}")

    return pd.DataFrame(
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "GAMECODE",
            "HOME",
            "AWAY",
            "SEASON",
            "ARENA_NAME",
        ]
    )


# -------------------------
# LOAD YOUR GAME DATA
# -------------------------
df_games = pd.read_csv(
    "data/processed/nba_games_with_elo.csv",
    parse_dates=["date"]
)

TEAM_MAP = {
    "atl": "ATL","bkn": "BKN","bos": "BOS","cha": "CHA","chi": "CHI","cle": "CLE","dal": "DAL",
    "den": "DEN","det": "DET","gs": "GSW","hou": "HOU","ind": "IND","lac": "LAC","lal": "LAL",
    "mem": "MEM","mia": "MIA","mil": "MIL","min": "MIN","no": "NOP","ny": "NYK","okc": "OKC",
    "orl": "ORL","phi": "PHI","phx": "PHX","por": "POR","sa": "SAS","sac": "SAC","tor": "TOR",
    "utah": "UTA","wsh": "WAS"
}

df_games["home"] = df_games["home"].str.lower().map(TEAM_MAP)
df_games["away"] = df_games["away"].str.lower().map(TEAM_MAP)

df_games["date"] = df_games["date"].dt.normalize()


unique_dates = df_games["date"].unique()
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

# Normalize merge keys
games_lookup["GAME_DATE"] = pd.to_datetime(games_lookup["GAME_DATE"]).dt.normalize()

df_games = df_games.merge(
    games_lookup,
    left_on=["date", "home", "away"],
    right_on=["GAME_DATE", "HOME", "AWAY"],
    how="left"
)


missing = df_games["GAME_ID"].isna().sum()
print(f"🔍 Missing GAME_IDs after merge: {missing}")

df_games.to_csv(
    "data/processed/nba_games_with_game_id.csv",
    index=False
)

print("💾 Saved games with GAME_ID")

failed_dates = pd.read_csv(ERROR_LOG_PATH, parse_dates=["date"])["date"].unique()

retry_games = []

for d in failed_dates:
    daily_games = fetch_games_for_date(d)
    if not daily_games.empty:
        retry_games.append(daily_games)

    time.sleep(REQUEST_SLEEP)

retry_df = pd.concat(retry_games, ignore_index=True)

retry_df.to_csv("data/processed/scoreboard_retry_games.csv", index=False)
