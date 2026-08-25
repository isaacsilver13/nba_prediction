import pandas as pd
import numpy as np
from scipy.stats import norm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "outputs/betting_edges_lightgbm2.csv"

INITIAL_BANKROLL = 1000
MAX_KELLY = 0.05          # max 5% of bankroll
KELLY_FRACTION = 0.5      # half-Kelly for safety
EV_THRESHOLD_PTS = 2  # min edge in points
STD_MARGIN = 12           # NBA margin std
ODDS_PAYOUT = 100 / 110   # -110 odds

df = pd.read_csv(INPUT_PATH)

df = (
    df.sort_values("GAME_DATE")
      .reset_index(drop=True)
)

# Probability home covers spread
df["p_cover"] = norm.cdf(
    (df["model_margin"] - df["spread"]) / STD_MARGIN
).clip(0.45, 0.58)

# Convert EV threshold (points → probability)
EV_THRESHOLD_PROB = norm.cdf(EV_THRESHOLD_PTS / STD_MARGIN) - 0.5

df["edge_home"] = df["p_cover"] - 0.5
df["edge_away"] = -df["edge_home"]

# Choose Bet Side
df["bet_side"] = np.where(
    df["edge_home"] > df["edge_away"], "home", "away"
)

df["bet_edge"] = df[["edge_home", "edge_away"]].max(axis=1)
df["bet"] = df["bet_edge"] > EV_THRESHOLD_PROB

# True Kelly fraction
df["p_win"] = np.where(
    df["bet_side"] == "home",
    df["p_cover"],
    1 - df["p_cover"]
)

df["kelly_frac"] = (
    (ODDS_PAYOUT * df["p_win"] - (1 - df["p_win"])) / ODDS_PAYOUT
)

# Fractional + capped Kelly
df["kelly_frac"] = (
    df["kelly_frac"]
    .clip(lower=0)
    .clip(upper=MAX_KELLY)
    * KELLY_FRACTION
)

bankroll = INITIAL_BANKROLL
pnl = []

for _, row in df.iterrows():
    bet_pnl = 0

    if row["bet"] and row["kelly_frac"] > 0:
        stake = bankroll * row["kelly_frac"]

        if row["bet_side"] == "home":
            won = row["score_home"] - row["score_away"] > row["spread"]
        else:
            won = row["score_away"] - row["score_home"] > -row["spread"]

        bet_pnl = stake * ODDS_PAYOUT if won else -stake
        bankroll += bet_pnl

    pnl.append(bet_pnl)

df["pnl"] = pnl
df["bankroll"] = INITIAL_BANKROLL + df["pnl"].cumsum()


results = (
    df.groupby("season")
      .agg(
          total_bets=("pnl", lambda x: (x != 0).sum()),
          season_profit=("pnl", "sum"),
      )
      .reset_index()
)

results["start_bankroll"] = INITIAL_BANKROLL
results["end_bankroll"] = INITIAL_BANKROLL + results["season_profit"]
results["roi_pct"] = results["season_profit"] / INITIAL_BANKROLL * 100

print(results)

print(df["bet_edge"].describe())
print(df["kelly_frac"].describe())
print(df["pnl"].describe())

df["peak"] = df["bankroll"].cummax()
df["drawdown"] = df["bankroll"] / df["peak"] - 1
print("Max Drawdown:", df["drawdown"].min())

wins = (df["pnl"] > 0).sum()
bets = (df["pnl"] != 0).sum()
print("Win Rate:", wins / bets)
