import pandas as pd
import numpy as np
import lightgbm as lgb

from scipy.stats import norm
from sklearn.metrics import mean_squared_error, brier_score_loss, roc_auc_score

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/nba_games_with_elo.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

# =========================
# FEATURES / TARGET
# =========================

FEATURES = [
    "elo_diff",
    "rolling_margin_diff_5",
    "rolling_margin_diff_10",
    "rest_advantage",
    "spread"  # MARKET EXPECTATION
]

TARGET_MARGIN = "favorite_margin"
TARGET_COVER  = "favorite_cover"

# =========================
# TIME-BASED SPLIT
# =========================

n = len(df)

train_end = int(n * 0.70)
test_start = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
test_df  = df.iloc[test_start:].copy()

# =========================
# LIGHTGBM MARGIN MODEL
# =========================

lgb_train = lgb.Dataset(
    train_df[FEATURES],
    label=train_df[TARGET_MARGIN]
)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=900
)

print(model.feature_importance())

# =========================
# PREDICT EXPECTED MARGIN
# =========================

test_df["pred_margin"] = model.predict(test_df[FEATURES])

# =========================
# ESTIMATE GLOBAL SIGMA
# =========================

residuals = train_df[TARGET_MARGIN] - model.predict(train_df[FEATURES])
sigma_global = residuals.std(ddof=1)

print(f"Estimated global margin sigma: {sigma_global:.2f}")

# Home favorites slightly lower variance
test_df["sigma"] = np.where(
    test_df["favorite_is_home"] == 1,
    sigma_global * 0.95,
    sigma_global * 1.05
)

# =========================
# CONVERT MARGIN → COVER PROB
# =========================

z = (test_df["pred_margin"] - test_df["spread"]) / test_df["sigma"]
test_df["p_cover"] = norm.cdf(z)

# Probability clipping (fat-tail protection)
test_df["p_cover"] = test_df["p_cover"].clip(0.35, 0.65)

# =========================
# MODEL DIAGNOSTICS
# =========================

rmse = mean_squared_error(
    test_df[TARGET_MARGIN],
    test_df["pred_margin"]
) ** 0.5

print(f"RMSE (margin): {rmse:.2f}")

print("Brier score:",
      brier_score_loss(test_df[TARGET_COVER], test_df["p_cover"]))

print("AUC:",
      roc_auc_score(test_df[TARGET_COVER], test_df["p_cover"]))

# =========================
# DECILE CALIBRATION
# =========================

test_df["decile"] = pd.qcut(
    test_df["p_cover"],
    q=10,
    duplicates="drop"
)

print("\n===== DECILE CALIBRATION =====")
print(
    test_df
    .groupby("decile", observed=True)[TARGET_COVER]
    .mean()
)

print("Bins used:", test_df["decile"].nunique())

# =========================
# BETTING LOGIC (VIG-AWARE)
# =========================

IMPLIED_PROB = 0.5238  # -110 odds

test_df["edge"] = test_df["p_cover"] - IMPLIED_PROB

def edge_threshold(p):
    if p < 0.45:
        return 0.02
    elif p < 0.50:
        return 0.03
    else:
        return 0.06

test_df["edge_cut"] = test_df["p_cover"].apply(edge_threshold)

bets = test_df[
    (test_df["edge"] > test_df["edge_cut"])].copy()

# =========================
# KELLY SIZING (FRACTIONAL)
# =========================

b = 0.91  # payout for -110
bets["kelly"] = (
    (bets["p_cover"] * b - (1 - bets["p_cover"])) / b
)

bets["kelly"] = bets["kelly"].clip(0, 0.05)  # 5% max stake
bets["stake"] = bets["kelly"] * 100  # bankroll units

# =========================
# BET SUMMARY
# =========================

print("\n===== BET SUMMARY =====")
print("Bets:", len(bets))
print("Win rate:", bets[TARGET_COVER].mean())
print("Avg edge:", bets["edge"].mean())
print("Avg Kelly stake:", bets["stake"].mean())
