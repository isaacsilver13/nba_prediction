import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
FEATURES = [
    "spread", "spread_signed", "is_home_favorite",
    "home_pts_last5", "away_pts_last5",
    "home_margin_last5", "away_margin_last5",
    "home_rest", "away_rest",
    "home_b2b", "away_b2b",
    "pts_diff_last5", "margin_diff_last5",
    "rest_diff"
]

TARGET = "id_spread"  # 1=favorite covered, 0=underdog covered

MIN_CONFIDENCE = 0.55  # Only bet if model predicts >55% probability
BET_AMOUNT = 100       # Flat bet per game

INPUT_FILE = "data/processed/nba_features.csv"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Keep regular season only
df = df[df["regular"] == True]
# Keep only valid ATS rows
df = df[df["id_spread"].isin([0,1])].copy()

# Drop rows with missing feature values
df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


# Sort by date for each team
df = df.sort_values(["season", "date"]).reset_index(drop=True)

# ==============================
# BACKTEST PER SEASON
# ==============================
seasons = sorted(df["season"].unique())
results = []

for i, season in enumerate(seasons):
    # Train on all previous seasons
    if i == 0:
        continue  # skip first season, no training data

    train_df = df[df["season"] < season]
    test_df = df[df["season"] == season]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    spread_test = test_df["spread"]  # actual spreads

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": 42
        },
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        
        callbacks=[lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]  # prints every 50 rounds
    )

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Accuracy metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    # ROI simulation
    bet_mask = (y_pred_prob >= MIN_CONFIDENCE) | (y_pred_prob <= (1-MIN_CONFIDENCE))
    test_bets = X_test.loc[bet_mask]
    test_probs = y_pred_prob[bet_mask]
    test_actual = y_test.loc[bet_mask]
    test_spreads = spread_test.loc[bet_mask]

    payouts = []
    for pred_prob, actual, spread in zip(test_probs, test_actual, test_spreads):
        # Use standard -110 odds (1.91 decimal) for each bet
        win = int((pred_prob >= 0.5 and actual==1) or (pred_prob < 0.5 and actual==0))
        payout = BET_AMOUNT * (1.91-1) if win else -BET_AMOUNT
        payouts.append(payout)

    total_profit = np.sum(payouts)
    roi = total_profit / (BET_AMOUNT * len(payouts)) * 100 if len(payouts) > 0 else 0

    results.append({
        "season": season,
        "accuracy": acc,
        "roc_auc": auc,
        "bets": len(payouts),
        "profit": total_profit,
        "roi": roi
    })

# ==============================
# RESULTS SUMMARY
# ==============================
results_df = pd.DataFrame(results)
print(results_df)

# Plot ROI per season
plt.figure(figsize=(12,5))
plt.bar(results_df["season"], results_df["roi"])
plt.xlabel("Season")
plt.ylabel("ROI (%)")
plt.title("Baseline ATS Model - Season-by-Season ROI")
plt.grid(True)
plt.show()
