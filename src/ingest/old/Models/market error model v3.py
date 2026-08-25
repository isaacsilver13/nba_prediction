import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

df = pd.read_csv(
    "data/processed/nba_games_with_elo.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

FEATURES = [
    "elo_diff",
    "rolling_margin_diff_5",
    "rolling_margin_diff_10",
    "rest_advantage"
]

TARGET = "favorite_cover"

n = len(df)

train_end = int(n * 0.70)
cal_end   = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
cal_df   = df.iloc[train_end:cal_end].copy()
test_df  = df.iloc[cal_end:].copy()

lgb_train = lgb.Dataset(
    train_df[FEATURES],
    label=train_df[TARGET]
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=800
)

cal_df["raw_p"] = model.predict(cal_df[FEATURES])
test_df["raw_p"] = model.predict(test_df[FEATURES])

iso = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip"
)

iso.fit(
    cal_df["raw_p"],
    cal_df[TARGET]
)

test_df["p_cover"] = iso.predict(test_df["raw_p"])

print("Brier (raw):",
      brier_score_loss(test_df[TARGET], test_df["raw_p"]))

print("Brier (calibrated):",
      brier_score_loss(test_df[TARGET], test_df["p_cover"]))

print("AUC:",
      roc_auc_score(test_df[TARGET], test_df["p_cover"]))

test_df["decile"] = pd.qcut(
    test_df["p_cover"],
    q=10,
    duplicates="drop"
)

print(
    test_df
    .groupby("decile")[TARGET]
    .mean()
)

print("Actual bins:", test_df["decile"].nunique())

IMPLIED_PROB = 0.5238

test_df["edge"] = test_df["p_cover"] - IMPLIED_PROB

bets = test_df[test_df["edge"] > 0.03]   # start conservative

print("Bets:", len(bets))
print("Win rate:", bets["favorite_cover"].mean())
print("Avg edge:", bets["edge"].mean())
