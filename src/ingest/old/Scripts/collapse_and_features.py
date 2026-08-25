import os
import pandas as pd

# ==============================
# CONFIG
# ==============================
RAW_FILE = "data/processed/games_prepped_2020_2024.csv"
OUTPUT_FILE = "data/processed/games_ml_ready.csv"

ROLLING_WINDOW = 5  # ensure matches your preprocessing
NEUTRAL_SITE_IDS = []  # optional: add any game_ids for neutral-site games

# Optional: dictionary for team cities (for travel distance)
TEAM_CITIES = {
    "ATL": "Atlanta, GA",
    "BOS": "Boston, MA",
    "BKN": "Brooklyn, NY",
    "CHA": "Charlotte, NC",
    "CHI": "Chicago, IL",
    "CLE": "Cleveland, OH",
    "DAL": "Dallas, TX",
    "DEN": "Denver, CO",
    "DET": "Detroit, MI",
    "GSW": "San Francisco, CA",
    "HOU": "Houston, TX",
    "IND": "Indianapolis, IN",
    "LAC": "Los Angeles, CA",
    "LAL": "Los Angeles, CA",
    "MEM": "Memphis, TN",
    "MIA": "Miami, FL",
    "MIL": "Milwaukee, WI",
    "MIN": "Minneapolis, MN",
    "NOP": "New Orleans, LA",
    "NYK": "New York, NY",
    "OKC": "Oklahoma City, OK",
    "ORL": "Orlando, FL",
    "PHI": "Philadelphia, PA",
    "PHX": "Phoenix, AZ",
    "POR": "Portland, OR",
    "SAC": "Sacramento, CA",
    "SAS": "San Antonio, TX",
    "TOR": "Toronto, ON",
    "UTA": "Salt Lake City, UT",
    "WAS": "Washington, DC"
}


# ==============================
# LOAD RAW TEAM-LEVEL DATA
# ==============================
games = pd.read_csv(RAW_FILE, parse_dates=["game_date"])
games = games.sort_values(["team", "game_date"])

# ==============================
# STEP 1: ADD REST DAYS AND B2B
# ==============================
games['prev_game_date'] = games.groupby('team')['game_date'].shift(1)
games['rest_days'] = (games['game_date'] - games['prev_game_date']).dt.days
games['b2b'] = (games['rest_days'] == 1).astype(int)

# ==============================
# STEP 2: COLLAPSE TO GAME-LEVEL
# ==============================
home = games[games["home"] == 1].copy()
away = games[games["home"] == 0].copy()

# Rename columns for clarity
home_cols = {
    "team": "home_team",
    "points_for": "home_points",
    "points_against": "away_points",
    f"points_for_last{ROLLING_WINDOW}": f"home_points_for_last{ROLLING_WINDOW}",
    f"points_against_last{ROLLING_WINDOW}": f"home_points_against_last{ROLLING_WINDOW}",
    f"margin_last{ROLLING_WINDOW}": f"home_margin_last{ROLLING_WINDOW}",
    "b2b": "home_b2b",
    "rest_days": "home_rest_days"
}
away_cols = {
    "team": "away_team",
    "points_for": "away_points",
    "points_against": "home_points",
    f"points_for_last{ROLLING_WINDOW}": f"away_points_for_last{ROLLING_WINDOW}",
    f"points_against_last{ROLLING_WINDOW}": f"away_points_against_last{ROLLING_WINDOW}",
    f"margin_last{ROLLING_WINDOW}": f"away_margin_last{ROLLING_WINDOW}",
    "b2b": "away_b2b",
    "rest_days": "away_rest_days"
}

home = home.rename(columns=home_cols)
away = away.rename(columns=away_cols)

# Merge home and away
games_ml = pd.merge(
    home[["game_id", "game_date", "home_team", "home_points",
          f"home_points_for_last{ROLLING_WINDOW}",
          f"home_points_against_last{ROLLING_WINDOW}",
          f"home_margin_last{ROLLING_WINDOW}",
          "home_b2b",
          "home_rest_days"]],
    away[["game_id", "away_team", "away_points",
          f"away_points_for_last{ROLLING_WINDOW}",
          f"away_points_against_last{ROLLING_WINDOW}",
          f"away_margin_last{ROLLING_WINDOW}",
          "away_b2b",
          "away_rest_days"]],
    on="game_id",
    how="inner"
)

# Keep useful columns
keep_cols = ["game_id", "game_date", "home_team", "away_team",
             "home_points", "away_points",
             f"home_points_for_last{ROLLING_WINDOW}", f"home_points_against_last{ROLLING_WINDOW}", f"home_margin_last{ROLLING_WINDOW}",
             f"away_points_for_last{ROLLING_WINDOW}", f"away_points_against_last{ROLLING_WINDOW}", f"away_margin_last{ROLLING_WINDOW}",
             "home_b2b", "away_b2b",
             "home_rest_days", "away_rest_days"]

games_ml = games_ml[keep_cols]

# ==============================
# STEP 3: ADD OPTIONAL IMPROVEMENTS
# ==============================

# 3a. Home win target
games_ml['home_win'] = (games_ml['home_points'] > games_ml['away_points']).astype(int)

# 3b. Neutral site detection (for your international/neutral games)
games_ml['neutral_site'] = games_ml['game_id'].isin(NEUTRAL_SITE_IDS).astype(int)

# 3c. Travel distance placeholder (future feature)
# You can implement with city coordinates later
games_ml['home_travel_distance'] = pd.NA
games_ml['away_travel_distance'] = pd.NA

# ==============================
# STEP 4: SAVE FINAL DATA
# ==============================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
games_ml.to_csv(OUTPUT_FILE, index=False)
print(f"ML-ready dataset saved: {OUTPUT_FILE}")
print(f"Shape: {games_ml.shape}")