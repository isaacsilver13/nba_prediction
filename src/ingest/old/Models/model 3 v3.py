import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import norm

# =========================
# NBA ATS MODEL — MARGIN REGRESSION
# =========================



# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data/processed/nba_games_with_elo.csv"
BANKROLL_START = 10000
MAX_KELLY = 0.05  # max fraction of bankroll to bet per game
ODDS = 1  # assume +100 payout for simplicity, adjust if using actual odds
PERCENTILES = [75, 80, 85, 90, 95, 97.5]  # edge percentiles to test
SIGMA_MARGIN = 12.0        # Std dev of NBA margin
# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["elo_diff"] = np.where(
    df["is_home_favorite"] == 1,
    df["elo_home"] - df["elo_away"],
    df["elo_away"] - df["elo_home"]
)

#Add Margin Columns
df["home_margin"] = df["score_home"] - df["score_away"]
df["away_margin"] = -df["home_margin"]

df = df.sort_values("date")

home_roll_5 = (
    df.groupby("home")["home_margin"]
    .rolling(5)
    .mean()
    .reset_index(level=0, drop=True)
)

away_roll_5 = (
    df.groupby("away")["away_margin"]
    .rolling(5)
    .mean()
    .reset_index(level=0, drop=True)
)

home_roll_10 = (
    df.groupby("home")["home_margin"]
    .rolling(10)
    .mean()
    .reset_index(level=0, drop=True)
)

away_roll_10 = (
    df.groupby("away")["away_margin"]
    .rolling(10)
    .mean()
    .reset_index(level=0, drop=True)
)

df["rolling_margin_diff_5"] = np.where(
    df["is_home_favorite"] == 1,
    home_roll_5 - away_roll_5,
    away_roll_5 - home_roll_5
)

df["rolling_margin_diff_10"] = np.where(
    df["is_home_favorite"] == 1,
    home_roll_10 - away_roll_10,
    away_roll_10 - home_roll_10
)



#Add Rest Columns
df["last_game_home"] = df.groupby("home")["date"].shift(1)
df["last_game_away"] = df.groupby("away")["date"].shift(1)

df["home_rest"] = (df["date"] - df["last_game_home"]).dt.days
df["away_rest"] = (df["date"] - df["last_game_away"]).dt.days

df["rest_advantage"] = df["home_rest"] - df["away_rest"]


# -------------------------
# TARGET VARIABLE
# Favorite margin (corrected for home/away)
# -------------------------
df["favorite_margin"] = np.where(
    df["is_home_favorite"] == 1,
    df["score_home"] - df["score_away"],
    df["score_away"] - df["score_home"]
)

df["favorite_cover"] = (df["favorite_margin"] > df["spread"]).astype(int)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
FEATURES = [
    "elo_diff",
    "rolling_margin_diff_5",
    "rolling_margin_diff_10",
    "rest_advantage",
    "home_flag"
]

df = df.dropna(subset=FEATURES + ["favorite_margin", "spread"])

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
split_season = 2022

train = df[df["season"] < split_season]
test = df[df["season"] >= split_season]

X_train = train[FEATURES]
y_train = train["favorite_margin"]

X_test = test[FEATURES]
y_test = test["favorite_margin"]

# -------------------------
# LIGHTGBM DATASETS
# -------------------------
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# -------------------------
# MODEL
# -------------------------
params = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.85,
    "seed": 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=800,
    valid_sets=[valid_data]
)

# -------------------------
# PREDICTIONS
# -------------------------
test["predicted_margin"] = model.predict(X_test)

# Probability favorite covers spread
test["p_cover"] = norm.cdf(
    (test["predicted_margin"] - test["spread"]) / SIGMA_MARGIN
)

# -------------------------
# IMPLIED PROBABILITY
# -------------------------
implied_prob = abs(ODDS) / (abs(ODDS) + 100)

test["edge"] = test["p_cover"] - implied_prob
EDGE_THRESHOLDS = np.percentile(test["edge"], PERCENTILES)
print("Using edge thresholds:", EDGE_THRESHOLDS)
test["clv"] = test["predicted_margin"] - test["spread"]




# -------------------------------
# PREP DATA
# -------------------------------

# Drop rows with NaNs in key columns
test = test.dropna(subset=["predicted_margin", "spread", "clv", "p_cover", "favorite_cover"]).copy()

# Compute edge
test["edge"] = test["predicted_margin"] - test["spread"]

# -------------------------------
# CHOOSE EDGE THRESHOLDS BASED ON PERCENTILES
# -------------------------------
EDGE_THRESHOLDS = np.percentile(test["edge"], PERCENTILES)
print("Using edge thresholds:", EDGE_THRESHOLDS)

# -------------------------------
# LOOP THROUGH THRESHOLDS AND CALCULATE ROI/CLV
# -------------------------------
summary = []

for edge_th in EDGE_THRESHOLDS:
    # Select bets above threshold
    bets = test.loc[test["edge"] >= edge_th].copy()
    
    if len(bets) == 0:
        continue

    # Kelly fraction sizing
    b = ODDS
    bets["kelly"] = np.clip(
        (bets["p_cover"] * (b + 1) - 1) / b,
        0,
        MAX_KELLY
    )

    bets["stake"] = bets["kelly"] * BANKROLL_START

    # Profit calculation
    bets["profit"] = np.where(
        bets["favorite_cover"] == 1,
        bets["stake"] * b,
        -bets["stake"]
    )

    # ROI and CLV
    roi = bets["profit"].sum() / bets["stake"].sum()
    avg_clv = bets["clv"].mean()

    summary.append({
        "edge_threshold": edge_th,
        "roi_pct": roi * 100,
        "avg_clv": avg_clv,
        "bets": len(bets)
    })

# -------------------------------
# CREATE SUMMARY TABLE
# -------------------------------
summary_df = pd.DataFrame(summary)
if summary_df.empty:
    raise ValueError("No bets qualified for any edge threshold. Check edge distribution.")
summary_df = summary_df.set_index("edge_threshold").sort_index()

print("\n===== EDGE THRESHOLD SUMMARY =====")
print(summary_df)

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 6))

# X-axis: edge thresholds
x = summary_df.index

# ROI on left axis
ax1.plot(x, summary_df["roi_pct"], marker="o")
ax1.set_xlabel("Edge Threshold")
ax1.set_ylabel("ROI (%)")
ax1.axhline(0)  # break-even line

# CLV on right axis
ax2 = ax1.twinx()
ax2.plot(x, summary_df["avg_clv"], marker="s")
ax2.set_ylabel("Average CLV")

for edge, bets in zip(summary_df.index, summary_df["bets"]):
    ax1.annotate(
        str(bets),
        (edge, summary_df.loc[edge, "roi_pct"]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8
    )



plt.title("ROI and CLV vs Edge Threshold")
plt.tight_layout()
plt.show()
