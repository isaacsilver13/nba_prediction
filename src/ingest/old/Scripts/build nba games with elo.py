import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

INPUT_PATH = "data/processed/nba_features.csv"
OUTPUT_PATH = "data/processed/nba_games_with_elo.csv"

ELO_START = 1500
ELO_K = 20
HOME_ADVANTAGE = 65

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# =========================
# BASIC GAME FEATURES
# =========================

df["home_margin"] = df["score_home"] - df["score_away"]
df["away_margin"] = df["score_away"] - df["score_home"]

# Favorite cover (manual, no leakage)
df["favorite_cover_manual"] = np.where(
    df["is_home_favorite"] == 1,
    (df["score_home"] - df["score_away"]) > df["spread"],
    (df["score_away"] - df["score_home"]) > df["spread"]
).astype(int)

# Drop pushes
df = df[df["home_margin"].abs() != df["spread"]].copy()

# =========================
# REST / BACK-TO-BACK
# =========================

df["home_rest"] = (
    df.groupby("home")["date"]
    .diff()
    .dt.days
    .fillna(7)
    .clip(0, 7)
)

df["away_rest"] = (
    df.groupby("away")["date"]
    .diff()
    .dt.days
    .fillna(7)
    .clip(0, 7)
)

# =========================
# ELO INITIALIZATION
# =========================

teams = pd.unique(df[["home", "away"]].values.ravel())
elo = {team: ELO_START for team in teams}

elo_home_list = []
elo_away_list = []

# =========================
# ELO UPDATE LOOP (PRE-GAME)
# =========================

for _, row in df.iterrows():
    home = row["home"]
    away = row["away"]

    elo_home = elo[home] + HOME_ADVANTAGE
    elo_away = elo[away]

    elo_home_list.append(elo_home)
    elo_away_list.append(elo_away)

    # Expected outcome
    exp_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    exp_away = 1 - exp_home

    # Actual result
    if row["score_home"] > row["score_away"]:
        act_home, act_away = 1, 0
    else:
        act_home, act_away = 0, 1

    # Margin-of-victory multiplier
    margin = abs(row["score_home"] - row["score_away"])
    mult = np.log(margin + 1) * (2.2 / ((elo_home - elo_away) * 0.001 + 2.2))

    # Update ratings
    elo[home] += ELO_K * mult * (act_home - exp_home)
    elo[away] += ELO_K * mult * (act_away - exp_away)

# Attach Elo columns
df["elo_home"] = elo_home_list
df["elo_away"] = elo_away_list

#Attach Home Flag column
df["home_flag"] = 1

# =========================
# FINAL DATASET
# =========================

final_cols = [
    "date",
    "season",
    "home",
    "away",
    "score_home",
    "score_away",
    "spread",
    "is_home_favorite",
    "home_margin",
    "away_margin",
    "favorite_cover_manual",
    "home_rest",
    "away_rest",
    "elo_home",
    "elo_away",
    "home_flag"
]

df_final = df[final_cols].copy()

df_final.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved {len(df_final):,} games to {OUTPUT_PATH}")
print(df_final.head())
