import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ======================================================
# CONFIG
# ======================================================

DATA_PATH = "data/processed/nba_features.csv"

BANKROLL_START = 10_000
ODDS_DECIMAL = 1.91        # -110
KELLY_FRACTION = 0.5       # Half Kelly
EDGE_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

BASE_FEATURES = [
    "is_home_favorite",
    "home_pts_last5",
    "away_pts_last5",
    "home_margin_last5",
    "away_margin_last5",
    "home_rest",
    "away_rest",
    "home_b2b",
    "away_b2b",
    "pts_diff_last5",
    "margin_diff_last5",
    "rest_diff"
]

# ======================================================
# ELO FUNCTIONS
# ======================================================

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, result, k=20):
    exp_a = expected_score(elo_a, elo_b)
    elo_a += k * (result - exp_a)
    elo_b += k * ((1 - result) - (1 - exp_a))
    return elo_a, elo_b

# ======================================================
# KELLY BET SIZING
# ======================================================

def kelly_bet_size(prob, odds, bankroll):
    b = odds - 1
    k = (prob * b - (1 - prob)) / b
    k = max(0, k)
    return bankroll * k * KELLY_FRACTION

# ======================================================
# LOAD + PREP DATA
# ======================================================

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Target: favorite margin
df["favorite_margin"] = np.where(
    df["is_home_favorite"] == 1,
    df["score_home"] - df["score_away"],
    df["score_away"] - df["score_home"]
)

df = df.dropna(subset=["favorite_margin"]).reset_index(drop=True)

# ======================================================
# BUILD ELO RATINGS (NO LEAKAGE)
# ======================================================

elos = defaultdict(lambda: 1500)

df["elo_home"] = np.nan
df["elo_away"] = np.nan

df = df.sort_values("date").reset_index(drop=True)

for i, row in df.iterrows():
    home, away = row["home"], row["away"]

    df.loc[i, "elo_home"] = elos[home]
    df.loc[i, "elo_away"] = elos[away]

    result = 1 if row["score_home"] > row["score_away"] else 0
    elos[home], elos[away] = update_elo(elos[home], elos[away], result)

df["elo_diff"] = df["elo_home"] - df["elo_away"]

FEATURES = BASE_FEATURES + ["elo_diff"]

# ======================================================
# SEASON-AWARE BACKTEST
# ======================================================

results = []

seasons = sorted(df["season"].unique())

for season in seasons[1:]:
    print(f"\n===== SEASON {season} =====")

    train_df = df[df["season"] < season]
    test_df = df[df["season"] == season].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["favorite_margin"]
    X_test = test_df[FEATURES]

    # Train / validation split (safe)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": 42
        },
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50)]
    )

    # Predict margins
    test_df["pred_margin"] = model.predict(X_test)
    test_df["edge"] = test_df["pred_margin"] - test_df["spread"]

    # CLV proxy (model vs market)
    test_df["clv"] = test_df["edge"]

    # ==================================================
    # EDGE THRESHOLD TUNING + KELLY BETTING
    # ==================================================

    for edge_thresh in EDGE_GRID:
        bankroll = BANKROLL_START
        bets = test_df[abs(test_df["edge"]) >= edge_thresh]

        if bets.empty:
            continue

        clv_vals = []
        bets_placed = 0

        for _, r in bets.iterrows():
            bet_on_fav = r["edge"] > 0

            # Simple probability mapping from edge
            prob = 0.55 + min(abs(r["edge"]) / 10, 0.15)

            stake = kelly_bet_size(prob, ODDS_DECIMAL, bankroll)
            if stake < 1:
                continue

            win = (
                (bet_on_fav and r["favorite_margin"] > r["spread"]) or
                (not bet_on_fav and r["favorite_margin"] < r["spread"])
            )

            bankroll += stake * (ODDS_DECIMAL - 1) if win else -stake
            clv_vals.append(r["clv"])
            bets_placed += 1

        if bets_placed == 0:
            continue

        results.append({
            "season": season,
            "edge_threshold": edge_thresh,
            "bets": bets_placed,
            "roi_pct": (bankroll - BANKROLL_START) / BANKROLL_START * 100,
            "avg_clv": np.mean(clv_vals)
        })

# ======================================================
# RESULTS SUMMARY
# ======================================================

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("edge_threshold")[["roi_pct", "avg_clv", "bets"]]
    .mean()
    .sort_values("roi_pct", ascending=False)
)

print("\n===== EDGE THRESHOLD SUMMARY =====")
print(summary)
# Recompute ATS outcome manually
df["favorite_cover_manual"] = (
    df["favorite_margin"] > df["spread"]
).astype(int)

print(df[["id_spread", "favorite_cover_manual"]].value_counts())
