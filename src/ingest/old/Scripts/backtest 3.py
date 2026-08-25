import pandas as pd
import numpy as np
from scipy.stats import norm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "outputs/betting_edges_lightgbm2.csv"

INITIAL_BANKROLL = 1000
MAX_KELLY = 0.05          # max fraction of bankroll per bet
KELLY_FRACTION = 0.5      # half-Kelly for safety
EV_THRESHOLD_PTS = 2      # minimum predicted edge in points
STD_MARGIN = 12           # NBA margin std
ODDS_PAYOUT = 100 / 110   # -110 odds
MAX_EDGE = 0.25           # cap edge to prevent crazy Kelly

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_PATH)
df = df.sort_values("GAME_DATE").reset_index(drop=True)

# -----------------------------
# CALCULATE PROBABILITIES AND EDGE
# -----------------------------
# Prob home covers spread
df["p_cover"] = norm.cdf((df["model_margin"] - df["spread"]) / STD_MARGIN)

# Cap probabilities to prevent extreme edges
df["p_cover"] = df["p_cover"].clip(0.05, 0.95)

# Convert EV threshold (points -> probability)
EV_THRESHOLD_PROB = norm.cdf(EV_THRESHOLD_PTS / STD_MARGIN) - 0.5

df["edge_home"] = (df["p_cover"] - 0.5).clip(-MAX_EDGE, MAX_EDGE)
df["edge_away"] = (-df["edge_home"]).clip(-MAX_EDGE, MAX_EDGE)

# Determine which side to bet
df["bet_side"] = np.where(df["edge_home"] > df["edge_away"], "home", "away")
df["bet_edge"] = df[["edge_home", "edge_away"]].max(axis=1)
df["bet"] = df["bet_edge"] > EV_THRESHOLD_PROB

# -----------------------------
# KELLY FRACTION
# -----------------------------
df["p_win"] = np.where(
    df["bet_side"] == "home",
    df["p_cover"],
    1 - df["p_cover"]
)

df["kelly_frac"] = ((ODDS_PAYOUT * df["p_win"] - (1 - df["p_win"])) / ODDS_PAYOUT)
df["kelly_frac"] = df["kelly_frac"].clip(0, MAX_KELLY) * KELLY_FRACTION

# -----------------------------
# RUN BACKTEST
# -----------------------------
bankroll = INITIAL_BANKROLL
pnl_list = []

for _, row in df.iterrows():
    bet_pnl = 0

    if row["bet"] and row["kelly_frac"] > 0:
        stake = bankroll * row["kelly_frac"]

        if row["bet_side"] == "home":
            won = (row["score_home"] - row["score_away"]) > row["spread"]
        else:
            won = (row["score_away"] - row["score_home"]) > -row["spread"]

        bet_pnl = stake * ODDS_PAYOUT if won else -stake
        bankroll += bet_pnl

    pnl_list.append(bet_pnl)

df["pnl"] = pnl_list
df["bankroll"] = INITIAL_BANKROLL + df["pnl"].cumsum()

# -----------------------------
# SEASON METRICS
# -----------------------------
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

# -----------------------------
# DRAWDOWN AND WIN RATE
# -----------------------------
df["peak"] = df["bankroll"].cummax()
df["drawdown"] = df["bankroll"] / df["peak"] - 1
max_drawdown = df["drawdown"].min()

wins = (df["pnl"] > 0).sum()
bets = (df["pnl"] != 0).sum()
win_rate = wins / bets if bets > 0 else 0

# -----------------------------
# OUTPUT
# -----------------------------
print("Season Summary:")
print(results)

print("\nBet Edge Stats:")
print(df["bet_edge"].describe())

print("\nKelly Fraction Stats:")
print(df["kelly_frac"].describe())

print("\nPnL Stats:")
print(df["pnl"].describe())

print(f"\nMax Drawdown: {max_drawdown:.2%}")
print(f"Win Rate: {win_rate:.2%}")
