# =========================
# NBA ATS MODEL — MARGIN REGRESSION
# =========================

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import norm

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data/processed/nba_games_with_elo.csv"
BANKROLL_START = 10000
ODDS = -110
SIGMA_MARGIN = 12.0        # Std dev of NBA margin
MAX_KELLY = 0.01           # Conservative sizing
#EDGE_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
EDGE_THRESHOLDS = np.arange(0.01, 0.5, 0.05)
PERCENTILES = [75, 80, 85, 90, 95, 97.5]  # edge percentiles to test

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


# -------------------------
# EDGE THRESHOLD ANALYSIS
# -------------------------
summary = []

for edge_th in EDGE_THRESHOLDS:
    bets = test.loc[test["edge"] >= edge_th].copy()

    if len(bets) == 0:
        continue

    # Kelly sizing
    b = 100 / abs(ODDS)
    bets["kelly"] = np.clip(
        (bets["p_cover"] * (b + 1) - 1) / b,
        0,
        MAX_KELLY
    )

    bets["stake"] = bets["kelly"] * BANKROLL_START

    bets["profit"] = np.where(
        bets["favorite_cover"] == 1,
        bets["stake"] * b,
        -bets["stake"]
    )

    roi = bets["profit"].sum() / bets["stake"].sum()

    summary.append({
        "edge_threshold": edge_th,
        "roi_pct": roi * 100,
        "avg_clv": bets["clv"].mean(),
        "bets": len(bets)
    })

summary_df = pd.DataFrame(summary)
if summary_df.empty:
    raise ValueError("No bets qualified for any edge threshold. Check edge distribution.")

summary_df = summary_df.set_index("edge_threshold").sort_index()

print("\n===== EDGE THRESHOLD SUMMARY =====")
print(summary_df)

# -------------------------
# FINAL BANKROLL (BEST EDGE)
# -------------------------
best_edge = summary_df["roi_pct"].idxmax()
final_bets = test[test["edge"] >= best_edge]

bankroll = BANKROLL_START
for _, row in final_bets.iterrows():
    b = 100 / abs(ODDS)
    k = min(
        (row["p_cover"] * (b + 1) - 1) / b,
        MAX_KELLY
    )
    stake = bankroll * k
    bankroll += stake * b if row["favorite_cover"] == 1 else -stake

print("\n===== ATS MODEL RESULTS =====")
print(f"Best edge threshold: {best_edge}")
print(f"Final bankroll: ${bankroll:,.2f}")
print(f"ROI: {(bankroll / BANKROLL_START - 1) * 100:.2f}%")
print(f"Bets placed: {len(final_bets)}")

print("\nFeature importance:")
print(pd.Series(model.feature_importance(), index=FEATURES))


import matplotlib.pyplot as plt

plt.hist(test["edge"], bins=50)
plt.title("Edge distribution")
plt.xlabel("Edge")
plt.ylabel("Number of games")
plt.show()
