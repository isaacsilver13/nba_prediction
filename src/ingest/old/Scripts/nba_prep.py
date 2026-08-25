import os
import pandas as pd
from ingest_nba_games import fetch_games

# ==============================
# CONFIG
# ==============================
START_SEASON = 2020
END_SEASON = 2024
ROLLING_WINDOW = 5
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Optional: list of games to exclude
EXCLUDE_GAME_IDS = ['0022400147', '0022401229', '0022401230', '0022400621','0022400633']

# ==============================
# STEP 1: LOAD RAW DATA
# ==============================
games = fetch_games(start_season=START_SEASON,
                    end_season=END_SEASON,
                    exclude_game_ids=EXCLUDE_GAME_IDS)

print(f"Loaded {len(games):,} rows")


# ==============================
# STEP 2: CREATE TARGETS
# ==============================
# Home win
games['home_win'] = ((games['home'] == 1) & (games['points_for'] > games['points_against'])).astype(int)

# Point margin
games['margin'] = games['points_for'] - games['points_against']

# Placeholder for ATS target (home_cover), to fill after merging odds
games['home_cover'] = pd.NA

# ==============================
# STEP 3: ADD ROLLING FEATURES
# ==============================
# Sort for rolling calculation
games = games.sort_values(['team', 'game_date'])

# Rolling averages for last N games
rolling = games.groupby('team').rolling(ROLLING_WINDOW, on='game_date').agg({
    'points_for': 'mean',
    'points_against': 'mean',
}).reset_index()

rolling = rolling.rename(columns={
    'points_for': f'points_for_last{ROLLING_WINDOW}',
    'points_against': f'points_against_last{ROLLING_WINDOW}'
})

games = games.merge(
    rolling[['team', 'game_date', f'points_for_last{ROLLING_WINDOW}', f'points_against_last{ROLLING_WINDOW}']],
    on=['team', 'game_date'],
    how='left'
)

# Optional: add margin last N games
games[f'margin_last{ROLLING_WINDOW}'] = games[f'points_for_last{ROLLING_WINDOW}'] - games[f'points_against_last{ROLLING_WINDOW}']

# ==============================
# STEP 4: SAVE PROCESSED DATA
# ==============================
output_path = os.path.join(PROCESSED_DIR, f'games_prepped_{START_SEASON}_{END_SEASON}.csv')
games.to_csv(output_path, index=False)
print(f"Processed data saved: {output_path}")
