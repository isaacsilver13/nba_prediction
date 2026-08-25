# kelly_betting_analysis.py

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

# -----------------------
# Settings
# -----------------------
INPUT_PATH = "data/processed/df_model_3.csv"
INPUT_PATH = 'outputs/betting_edges_lightgbm.csv'
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAYOUT = 0.909           # decimal odds -1 for -110
FRACTIONAL_KELLY = 0.1   # fraction of full Kelly to reduce volatility
EV_THRESHOLD = 0.1      # minimum expected value to place a bet

# -----------------------
# Read data
# -----------------------
df_test = pd.read_csv(INPUT_PATH, parse_dates=['GAME_DATE'])

sigma_hat = df_test["edge"].std()

# -----------------------
# Only calibrate probabilities where edge exists
mask = df_test["bet_side"] != "NO BET"  # use only non-NaN bets for calibration
ir = IsotonicRegression(out_of_bounds='clip')
# Fake target: home covered or not
df_test["home_cover"] = df_test["home_margin"] > df_test["spread_signed"]
df_test["win_prob_home_calibrated"] = ir.fit_transform(df_test.loc[mask, "win_prob_home"],
                                                      df_test.loc[mask, "home_cover"].astype(int))
# Use calibrated probabilities
df_test["win_prob_home"] = df_test["win_prob_home_calibrated"]

# -----------------------
# Define margin coverage
# -----------------------
df_test["home_margin_needed_to_cover"] = np.where(
    df_test["is_home_favorite"] == 1,
    df_test["spread"],
    -1 * df_test["spread"]
)
df_test["away_margin_needed_to_cover"] = -1 * df_test["home_margin_needed_to_cover"]
df_test["did_home_cover"] = df_test["home_margin"] > df_test["home_margin_needed_to_cover"]
df_test["did_away_cover"] = df_test["away_margin"] > df_test["away_margin_needed_to_cover"]
df_test["team_that_covered"] = np.where(df_test["did_home_cover"], "HOME", "AWAY")
df_test["bet_win"] = (df_test["bet_side"] == df_test["team_that_covered"]).astype(int)

# -----------------------
# Compute EVs for home/away
# -----------------------
df_test["ev_home"] = df_test["win_prob_home"] * PAYOUT - (1 - df_test["win_prob_home"])
df_test["ev_away"] = (1 - df_test["win_prob_home"]) * PAYOUT - df_test["win_prob_home"]

# Apply EV threshold to determine bet side
df_test["bet_side"] = np.where(
    df_test["ev_home"] > EV_THRESHOLD, "HOME",
    np.where(df_test["ev_away"] > EV_THRESHOLD, "AWAY", "NO BET")
)

# -----------------------
# Kelly sizing
# -----------------------
def kelly_fraction(p, b):
    f = (p * (b + 1) - 1) / b
    return np.clip(f, 0, 1)

df_test["kelly_frac"] = 0.0
home_mask = df_test["bet_side"] == "HOME"
away_mask = df_test["bet_side"] == "AWAY"

df_test.loc[home_mask, "kelly_frac"] = kelly_fraction(df_test.loc[home_mask, "win_prob_home"], PAYOUT)
df_test.loc[away_mask, "kelly_frac"] = kelly_fraction(1 - df_test.loc[away_mask, "win_prob_home"], PAYOUT)
df_test["kelly_frac"] *= FRACTIONAL_KELLY
MAX_KELLY = 0.1  # max 10% of bankroll per bet
df_test["kelly_frac"] = np.minimum(df_test["kelly_frac"], MAX_KELLY)


# -----------------------
# Compute Kelly-scaled P&L
# -----------------------
df_test["pnl_kelly"] = 0.0
df_test.loc[df_test["bet_win"] == 1, "pnl_kelly"] = df_test.loc[df_test["bet_win"] == 1, "kelly_frac"] * PAYOUT
df_test.loc[df_test["bet_win"] == 0, "pnl_kelly"] = -df_test.loc[df_test["bet_win"] == 0, "kelly_frac"]
df_test.loc[df_test["bet_side"] == "NO BET", "pnl_kelly"] = 0

# -----------------------
# Summarize overall Kelly bets
# -----------------------
bets = df_test[df_test["bet_side"] != "NO BET"].copy()
bets["cum_pnl_kelly"] = bets["pnl_kelly"].cumsum()

summary_kelly = {
    "num_bets": len(bets),
    "win_rate": bets["bet_win"].mean(),
    "total_pnl": bets["pnl_kelly"].sum(),
    "roi": bets["pnl_kelly"].sum() / (len(bets)+0.00001),
    "avg_ev": bets["bet_ev"].mean(),
}

print("Kelly Betting Summary:", summary_kelly)


# -----------------------
# Bankroll simulation (compounded Kelly)
# -----------------------
INITIAL_BANKROLL = 1000  # starting bankroll in units
bankroll = INITIAL_BANKROLL
bankroll_history = []

for i, row in bets.iterrows():
    bet_amount = bankroll * row["kelly_frac"]   # fraction of current bankroll
    if row["bet_win"] == 1:
        bankroll += bet_amount * PAYOUT
    else:
        bankroll -= bet_amount
    bankroll_history.append(bankroll)

bets["bankroll"] = bankroll_history

# -----------------------
# Plot compounded bankroll
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(bets["GAME_DATE"], bets["bankroll"])
plt.title("Compounded Bankroll Over Time (Kelly Sizing)")
plt.xlabel("Date")
plt.ylabel("Bankroll")
plt.grid(True)
plt.show()

# -----------------------
# Optional: final bankroll summary
# -----------------------
print(f"Initial bankroll: {INITIAL_BANKROLL}")
print(f"Final bankroll: {bankroll:.2f}")
print(f"Total return: {(bankroll / INITIAL_BANKROLL - 1) * 100:.2f}%")


# -----------------------
# Edge bucket analysis
# -----------------------
bets["edge_bucket"] = pd.cut(
    bets["edge"],
    bins=[-10, -4, -2, 0, 2, 4, 10]
)
edge_summary = bets.groupby("edge_bucket", observed=False).agg(
    num_bets=("pnl_kelly", "count"),
    total_pnl=("pnl_kelly", "sum"),
    roi=("pnl_kelly", lambda x: x.sum() / (x.count()+0.00000001)),
    win_rate=("bet_win", "mean")
).reset_index()
print("\nEdge Bucket Summary (Kelly-Weighted):")
print(edge_summary)

# -----------------------
# Probability bucket analysis
# -----------------------
bets["prob_bucket"] = pd.cut(
    bets["win_prob_home"],
    bins=np.linspace(0, 1, 11)
)
prob_summary = bets.groupby("prob_bucket", observed=False).agg(
    num_bets=("pnl_kelly", "count"),
    total_pnl=("pnl_kelly", "sum"),
    roi=("pnl_kelly", lambda x: x.sum() / (x.count()+0.00000001)),
    win_rate=("bet_win", "mean")
).reset_index()
print("\nProbability Bucket Summary (Kelly-Weighted):")
print(prob_summary)

# -----------------------
# Plot cumulative P&L
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(bets["GAME_DATE"], bets["cum_pnl_kelly"])
plt.title("Cumulative P&L – LightGBM Spread Model (Kelly Sizing)")
plt.xlabel("Date")
plt.ylabel("Units")
plt.grid(True)
plt.show()

# -----------------------
# Save output
# -----------------------
bets.to_csv(os.path.join(OUTPUT_DIR, "kelly_bets_summary.csv"), index=False)
print(f"\nKelly betting output saved to {OUTPUT_DIR}/kelly_bets_summary.csv")
