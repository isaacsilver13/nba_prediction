import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import norm

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
    "elo_home",
    "elo_away",
    "home_flag"
]

df_elo = df[final_cols].copy()

df = df_elo.sort_values("date").reset_index(drop=True)

df["elo_diff"] = np.where(
    df["is_home_favorite"] == 1,
    df["elo_home"] - df["elo_away"],
    df["elo_away"] - df["elo_home"]
)

#Add Margin Columns
df["home_margin"] = df["score_home"] - df["score_away"]
df["away_margin"] = -df["home_margin"]

df = df.sort_values("date")

# -------------------------
# BUILD LONG-FORM TEAM TABLE
# -------------------------

long = pd.concat(
    [
        df[["date", "home", "home_margin", "elo_home"]]
        .rename(columns={"home": "team", "home_margin": "margin", "elo_home": "ELO"}),

        df[["date", "away", "away_margin", "elo_away"]]
        .rename(columns={"away": "team", "away_margin": "margin", "elo_away": "ELO"}),
    ],
    ignore_index=True
)

# Ensure the data is sorted correctly BEFORE rolling
# Rolling assumes temporal order
long = long.sort_values(["team", "date"]).reset_index(drop=True)

# ------------------------------------------------------------------------------
# 5-game rolling average margin
# ------------------------------------------------------------------------------
long["rolling_margin_5"] = (
    long
    .groupby("team", group_keys=False)["margin"]
    .transform(
        lambda x: (
            x.shift(1)              # IMPORTANT: prevent leakage (only past games)
             .rolling(
                 window=5,
                 min_periods=3       # allow early-season values
             )
             .mean()
        )
    )
)

# ------------------------------------------------------------------------------
# 10-game rolling average margin (example extension)
# ------------------------------------------------------------------------------
long["rolling_margin_10"] = (
    long
    .groupby("team", group_keys=False)["margin"]
    .transform(
        lambda x: (
            x.shift(1)
             .rolling(10, min_periods=5)
             .mean()
        )
    )
)

long["ELO_Rolling_2YR"] = (
    long
    .groupby("team", group_keys=False)["ELO"]
    .transform(
        lambda x: (
            x.shift(1)              # IMPORTANT: prevent leakage (only past games)
             .rolling(
                 window=164
             )
             .mean()
        )
    )
)

for col in ["rolling_margin_5", "rolling_margin_10"]:
    long[col] = long[col].clip(-20, 20)


df = df.merge(
    long[["date", "team", "rolling_margin_5", "rolling_margin_10","ELO_Rolling_2YR"]],
    left_on=["date", "home"],
    right_on=["date", "team"],
    how="left"
).rename(
    columns={
        "rolling_margin_5": "home_roll_5",
        "rolling_margin_10": "home_roll_10"
    }
).drop(columns="team")

df = df.merge(
    long[["date", "team", "rolling_margin_5", "rolling_margin_10", "ELO_Rolling_2YR"]],
    left_on=["date", "away"],
    right_on=["date", "team"],
    how="left"
).rename(
    columns={
        "rolling_margin_5": "away_roll_5",
        "rolling_margin_10": "away_roll_10"
    }
).drop(columns="team")
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

#Add Rest Columns
df["last_game_home"] = df.groupby("home")["date"].shift(1)
df["last_game_away"] = df.groupby("away")["date"].shift(1)

df["home_rest"] = (df["date"] - df["last_game_home"]).dt.days.clip(0, 5)
df["away_rest"] = (df["date"] - df["last_game_away"]).dt.days.clip(0, 5)

df["rest_advantage"] = df["home_rest"] - df["away_rest"]
df["rest_advantage"] = df["rest_advantage"].clip(-3, 3)
df["rest_advantage_sq"] = np.sign(df["rest_advantage"]) * (df["rest_advantage"] ** 2)

df["favorite_margin"] = np.where(
    df["is_home_favorite"] == 1,
    df["score_home"] - df["score_away"],
    df["score_away"] - df["score_home"]
)

df["market_error"] = df["favorite_margin"] - df["spread"]#how wrong was Vegas

df["favorite_cover"] = (df["favorite_margin"] > df["spread"]).astype(int)#did the favorite cover

df["is_favorite"] = 1  # all rows are framed from favorite POV
df["favorite_is_home"] = df["is_home_favorite"]

df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved {len(df):,} games to {OUTPUT_PATH}")
