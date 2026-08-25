import pandas as pd
games = pd.read_csv("data/raw/games.csv")

import time
import os
from typing import List
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# ==============================
# CONFIG (can be overridden in functions)
# ==============================
DEFAULT_SEASON_TYPE = "Regular Season"
DEFAULT_SLEEP_SECONDS = 1.2
DEFAULT_OUTPUT_DIR = "data/raw"
DEFAULT_OUTPUT_FILE = "games.csv"

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


# ==============================
# HELPERS
# ==============================
def season_str(year: int) -> str:
    """Convert 2015 -> '2015-16'"""
    return f"{year}-{str(year + 1)[-2:]}"


def parse_home_away(matchup: str, team: str) -> int:
    """
    Determine if 'team' is home in a matchup string.
    Returns 1 if home, 0 if away.
    """
    if not isinstance(matchup, str):
        return 0

    clean = matchup.replace("*", "").strip()

    if " vs. " in clean:
        home_team = clean.split(" vs. ")[0].strip()
        return 1 if team == home_team else 0
    elif " @ " in clean:
        home_team = clean.split(" @ ")[1].strip()
        return 1 if team == home_team else 0
    else:
        return 0


def extract_opponent(matchup: str, team: str) -> str:
    """Extract opponent team abbreviation from matchup string."""
    parts = matchup.replace("vs.", "@").split("@")
    parts = [p.strip() for p in parts]
    return parts[1] if parts[0] == team else parts[0]


# ==============================
# MAIN FUNCTION
# ==============================
def fetch_games(start_season: int = 2020,
                end_season: int = 2024,
                season_type: str = DEFAULT_SEASON_TYPE,
                sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
                exclude_game_ids: List[str] = None) -> pd.DataFrame:
    """
    Fetch NBA game logs from nba_api for multiple seasons.
    Returns a DataFrame with one row per team per game.
    """
    
    all_games: List[pd.DataFrame] = []

    for year in range(start_season, end_season + 1):
        season = season_str(year)
        print(f"Fetching season {season}...")

        gamelog = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type
        )

        df = gamelog.get_data_frames()[0]

        # Normalize columns
        df = df.rename(columns={
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "SEASON_ID": "season_id",
            "TEAM_ABBREVIATION": "team",
            "MATCHUP": "matchup",
            "PTS": "points_for",
            "PTS_OPP": "points_against"
        })

        df["season"] = year
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df["home"] = df.apply(lambda r: parse_home_away(r["matchup"], r["team"]), axis=1)

        df["opponent"] = df.apply(
            lambda r: extract_opponent(r["matchup"], r["team"]),
            axis=1
        )
        if "points_against" not in df.columns:
        # Calculate points_against if missing
        # For NBA API, you can compute later, but for now fill with NaN
            df["points_against"] = pd.NA

        keep_cols = [
            "game_id",
            "matchup",
            "game_date",
            "season",
            "team",
            "opponent",
            "home",
            "points_for",
            "points_against"
        ]

        df = df[keep_cols]
        df = df.drop_duplicates(subset=["game_id", "team"])

        all_games.append(df)

        time.sleep(sleep_seconds)

    # ==============================
    # FINALIZE
    # ==============================

    games = pd.concat(all_games, ignore_index=True)
    
    exclude_game_ids = ['0022400147', '0022401229', '0022401230', '0022400621','0022400633']
    exclude_game_ids = set(exclude_game_ids)
    
    games = games[~games['game_id'].isin(exclude_game_ids)]

    home_count = games.groupby("game_id")["home"].sum()
    if not (home_count == 1).all():
        problem_games = home_count[home_count != 1]
        print("[WARN] Games with bad home assignment:\n", problem_games)
    # Sanity checks
    assert games.groupby("game_id").size().min() == 2
    assert games.groupby("game_id")["home"].sum().eq(1).all()

    ##tranform to calculate points against

    # Select the relevant columns for merging
    points_df = games[['game_id', 'team', 'points_for']].rename(
        columns={'team': 'opponent', 'points_for': 'points_against'}
    )

    # Merge on game_id + opponent to get points against
    games = games.merge(points_df, left_on=['game_id', 'opponent'], right_on=['game_id', 'opponent'], how='left')

    games = games.rename(columns={'points_against_y':'points_against'})
    games = games.drop(columns=["points_against_x"])

    # Ensure home/away assignment still makes sense
    home_count = games.groupby("game_id")["home"].sum()
    assert (home_count == 1).all(), "Some games have wrong home assignment!"

    # Home win flag
    games['home_win'] = ((games['home'] == 1) & (games['points_for'] > games['points_against'])).astype(int)

    # Away win flag
    games['away_win'] = ((games['home'] == 0) & (games['points_for'] > games['points_against'])).astype(int)

    return games


# ==============================
# OPTIONAL: CLI EXECUTION
# ==============================
if __name__ == "__main__":
    df_games = fetch_games()
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE)
    df_games.to_csv(output_path, index=False)
    print(f"Saved {len(df_games):,} rows to {output_path}")
