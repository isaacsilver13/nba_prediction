import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm

# -----------------------------
# CONFIG
# -----------------------------
EV_THRESHOLD = 2  # minimum predicted edge to place a bet (points)
INITIAL_BANKROLL = 1000  # starting bankroll
MAX_FRACTION = 0.05      # max fraction of bankroll per bet

# -----------------------------
# PREP DATA
# -----------------------------
INPUT_PATH = "outputs/betting_edges_lightgbm2.csv"

df_model = pd.read_csv(INPUT_PATH)

df_model = df_model.sort_values("GAME_DATE").copy()


# -----------------------------
# CALCULATE BETS AND EDGE
# -----------------------------

STD_MARGIN = 12  # NBA margin std ~11–13

df_model["p_cover"] = norm.cdf(
    df_model["model_margin"] / STD_MARGIN
)

df_model["edge_home"] = df_model["p_cover"] - 0.5
df_model["edge_away"] = -df_model["p_cover"] + 0.5


# # Compute predicted edge from the perspective of the team you would bet on
# df_model["edge_home"] = df_model["model_margin"] - df_model["spread"]
# df_model["edge_away"] = -df_model["model_margin"] - df_model["spread"]

# -----------------------------
# APPLY INJURY ADJUSTMENT (OPTIONAL)
# -----------------------------
# Example: if key players didn't play, reduce edge by 50%
# if "did_play" in df_model.columns:
#     df_model["home_adjustment"] = np.where(df_model["did_play"] == 0, 0.5, 1.0)
#     df_model["away_adjustment"] = np.where(df_model["did_play"] == 0, 0.5, 1.0)
#     df_model["edge_home"] *= df_model["home_adjustment"]
#     df_model["edge_away"] *= df_model["away_adjustment"]

EV_THRESHOLD = norm.cdf(EV_THRESHOLD/STD_MARGIN)- 0.5
print(EV_THRESHOLD)
# Determine bets
df_model["bet_home"] = df_model["edge_home"] > EV_THRESHOLD
df_model["bet_away"] = df_model["edge_away"] > EV_THRESHOLD



# -----------------------------
# KELLY BET SIZING
# -----------------------------
# Simplified Kelly: fraction = edge / (spread variance)
# We'll normalize edge to max fraction
SCALE_FACTOR = 10  # tune this based on model
df_model["kelly_home"] = np.clip(df_model["edge_home"] / SCALE_FACTOR, 0, MAX_FRACTION)
df_model["kelly_away"] = np.clip(df_model["edge_away"] / SCALE_FACTOR, 0, MAX_FRACTION)


# -------------------------
# Run backtest with dynamic bankroll
# -------------------------
bankroll = INITIAL_BANKROLL
pnl_list = []

for idx, row in df_model.iterrows():
    pnl = 0
    # Home bet
    if row["bet_home"]:
        bet_size = row["kelly_home"] * bankroll
        won = row["score_home"] - row["score_away"] > row["spread"]
        pnl += bet_size if won else -bet_size

    # Away bet
    if row["bet_away"]:
        bet_size = row["kelly_away"] * bankroll
        won = row["score_away"] - row["score_home"] > -row["spread"]
        pnl += bet_size if won else -bet_size

    bankroll += pnl
    pnl_list.append(pnl)

df_model["pnl"] = pnl_list
df_model["cumulative_bankroll"] = INITIAL_BANKROLL + df_model["pnl"].cumsum()

# -------------------------
# Evaluate results by season
# -------------------------

# Using a dictionary to map column -> aggregation
results = (
    df_model.groupby("season")
    .agg({
        "pnl": ["count", "sum"]  # count = total bets, sum = season profit
    })
)

# Flatten MultiIndex columns
results.columns = ["total_bets", "season_profit"]
results = results.reset_index()

# Add bankroll info

results["season_start_bankroll"] = INITIAL_BANKROLL
results["season_end_bankroll"] = INITIAL_BANKROLL + results["season_profit"]
results["roi"] = results["season_profit"] / INITIAL_BANKROLL * 100

print(results)

print(df_model["edge_home"].describe())
print(df_model["edge_away"].describe())
print(df_model["kelly_home"].describe())
print(df_model["kelly_away"].describe())
print(df_model["pnl"].describe())


df_model.to_csv("outputs/unrealistic1.csv", index = False)