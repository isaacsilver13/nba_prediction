import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import norm

from sklearn.metrics import mean_squared_error, roc_auc_score, brier_score_loss

# ============================================================
# LOAD DATA
# ============================================================
# Dataset is framed from the FAVORITE's perspective
# favorite_margin = favorite_score - opponent_score
# ============================================================

df = pd.read_csv(
    "data/processed/nba_games_with_elo.csv",
    parse_dates=["date"]
)

# Ensure strict chronological ordering (critical for betting models)
df = df.sort_values("date").reset_index(drop=True)

# ============================================================
# FEATURE / TARGET DEFINITIONS
# ============================================================
# Spread is INCLUDED — the model learns when Vegas is wrong
# ============================================================

FEATURES = [
    "spread",                   # market expectation (anchor)
    "elo_diff",
    "rolling_margin_diff_5",
    "rolling_margin_diff_10",
    "rest_advantage"
]

MARGIN_TARGET = "favorite_margin"         # must already exist in dataset
COVER_TARGET  = "favorite_cover" # binary: did favorite cover?
# ============================================================
# TIME-BASED SPLIT
# ============================================================
# 70% train
# 15% calibration (for sigma estimation)
# 15% test
# ============================================================

n = len(df)

train_end = int(n * 0.70)
cal_end   = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
cal_df   = df.iloc[train_end:cal_end].copy()
test_df  = df.iloc[cal_end:].copy()

# ============================================================
# LIGHTGBM DATASET
# ============================================================

lgb_train = lgb.Dataset(
    train_df[FEATURES],
    label=train_df[MARGIN_TARGET]
)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

# ============================================================
# TRAIN MODEL
# ============================================================

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000
)

# ============================================================
# PREDICT EXPECTED MARGINS
# ============================================================

cal_df.loc[:, "pred_margin"]  = model.predict(cal_df[FEATURES])
test_df.loc[:, "pred_margin"] = model.predict(test_df[FEATURES])

# Residuals represent uncertainty around predicted margin
# ============================================================

cal_df.loc[:, "residual"] = cal_df[MARGIN_TARGET] - cal_df["pred_margin"]

# Clip extreme garbage-time outcomes
cal_df.loc[:, "residual"] = cal_df["residual"].clip(-40, 40)

# ROBUST CONDITIONAL SIGMA MODEL
# ============================================================

# Absolute spread
cal_df.loc[:, "abs_spread"] = cal_df["spread"].abs()

# Keep only clean rows
sigma_df = cal_df[
    cal_df["abs_spread"].notna() &
    cal_df["residual"].notna()
].copy()

# Bucket spreads to avoid conditioning issues
sigma_df.loc[:, "spread_bucket"] = sigma_df["abs_spread"].round(1)

# Compute empirical sigma per bucket
sigma_table = (
    sigma_df
    .groupby("spread_bucket")["residual"]
    .apply(lambda x: np.sqrt(np.mean(x**2)))
    .reset_index(name="sigma")
)

# Smooth with rolling average
sigma_table.loc[:, "sigma"] = (
    sigma_table["sigma"]
    .rolling(window=3, center=True, min_periods=1)
    .mean()
)

# Fallback global sigma
GLOBAL_SIGMA = sigma_table["sigma"].median()

print(f"Estimated global margin sigma: {GLOBAL_SIGMA:.2f}")

# Lookup function with fallback
def sigma_fn(spread):
    key = round(abs(spread), 1)
    row = sigma_table[sigma_table["spread_bucket"] == key]
    if not row.empty:
        return float(row["sigma"].iloc[0])
    return GLOBAL_SIGMA

# Apply to test set
test_df.loc[:, "sigma"] = test_df["spread"].apply(sigma_fn)

# Home favorites are less volatile
test_df["sigma"] = np.where(
    test_df["favorite_is_home"] == 1,
    test_df["sigma"] * 0.95,
    test_df["sigma"] * 1.05
)

# ============================================================
# MARGIN → COVER PROBABILITY
# ============================================================

# P( margin > spread )
test_df.loc[:, "p_cover"] = norm.cdf(
    (test_df["pred_margin"] - test_df["spread"]) / test_df["sigma"]
).clip(0.38, 0.62)

# ============================================================
# EVALUATION
# ============================================================

rmse = np.sqrt(mean_squared_error(
    test_df[MARGIN_TARGET],
    test_df["pred_margin"]
))

print(f"RMSE (margin): {rmse:.2f}")

print("Brier score:",
      brier_score_loss(test_df[COVER_TARGET], test_df["p_cover"]))

print("AUC:",
      roc_auc_score(test_df[COVER_TARGET], test_df["p_cover"]))


# ============================================================
# CALIBRATION CHECK VIA DECILES
# ============================================================

test_df.loc[:, "decile"] = pd.qcut(
    test_df["p_cover"],
    q=10,
    duplicates="drop"
)

print("\n===== DECILE CALIBRATION =====")
print(
    test_df
    .groupby("decile")[COVER_TARGET]
    .mean()
)

print("Bins used:", test_df["decile"].nunique())

# ============================================================
# EDGE CALCULATION (FIXED ODDS)
# ============================================================
# -110 implied probability ≈ 52.38%
# ============================================================

IMPLIED_PROB = 0.5238  # -110 baseline

test_df.loc[:, "edge"] = test_df["p_cover"] - IMPLIED_PROB

bets = test_df[test_df["edge"] > 0.03]


print("\n===== BET SUMMARY =====")
print("Bets:", len(bets))
print("Win rate:", bets[COVER_TARGET].mean())
print("Avg edge:", bets["edge"].mean())