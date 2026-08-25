import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
# =========================
# NBA ATS MODEL — MARGIN REGRESSION
# =========================

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data/processed/nba_games_processed.csv"
MAX_KELLY = 0.05  # max fraction of bankroll to bet per game
PERCENTILES = [75, 80, 85, 90, 95, 97.5]  # edge percentiles to test
SIGMA_MARGIN = 12.0        # Std dev of NBA margin
EDGE_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0]
EDGE_THRESHOLDS = np.arange(0.5, 4.5, 0.5)
ODDS = -110
BANKROLL_START = 10_000
# Decimal odds payout multiplier
b = 100 / abs(ODDS)  # ≈ 0.909


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)



# -------------------------
# FEATURE ENGINEERING
# -------------------------
FEATURES = [
    "spread",                 # market anchor
    "elo_diff",
    "rolling_margin_diff_5",
    "rolling_margin_diff_10",
    "rest_advantage",
    "home_flag"
]

df = df.dropna(subset=FEATURES + ["market_error",  "favorite_is_home"])

split_season = 2022

train = df[df["season"] < split_season]
test = df[df["season"] >= split_season]

X_train = train[FEATURES]
y_train = train[["market_error", "favorite_is_home"]]#we want to predict how far off Vegas is by Home and Away

X_test = test[FEATURES]
y_test = test[["market_error", "favorite_is_home"]]

train_home_fav = train[train["favorite_is_home"] == 1]
train_away_fav = train[train["favorite_is_home"] == 0]


train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "verbosity": -1
}

def train_model(train_df):
    X = train_df[FEATURES]
    y = train_df["market_error"]

    dtrain = lgb.Dataset(X, label=y)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "verbosity": -1
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300
    )
    return model, y.std()


home_model, home_resid_std = train_model(train_home_fav)#train a model for margin errors and home favorites 
away_model, away_resid_std = train_model(train_away_fav)#ditto for away teams


# --- TRAIN SET CALIBRATION ---
train["predicted_market_error"] = np.nan

home_mask_train = train["favorite_is_home"] == 1
away_mask_train = train["favorite_is_home"] == 0

train.loc[home_mask_train, "predicted_market_error"] = home_model.predict(
    train.loc[home_mask_train, FEATURES]
)

train.loc[away_mask_train, "predicted_market_error"] = away_model.predict(
    train.loc[away_mask_train, FEATURES]
)


# Fit isotonic calibration
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(
    train["predicted_market_error"],
    train["market_error"]
)


test_home_fav = test[test["favorite_is_home"] == 1].copy()
test_away_fav = test[test["favorite_is_home"] == 0].copy()

test_home_fav["predicted_market_error"] = home_model.predict(
    test_home_fav[FEATURES]
)

test_away_fav["predicted_market_error"] = away_model.predict(
    test_away_fav[FEATURES]
)

test = pd.concat([test_home_fav, test_away_fav]).sort_index()


# Predicted error vs market
test["predicted_market_error"] = np.nan
home_mask = test["favorite_is_home"] == 1
away_mask = test["favorite_is_home"] == 0

test.loc[home_mask, "predicted_market_error"] = home_model.predict(
    test.loc[home_mask, FEATURES]
)

test.loc[away_mask, "predicted_market_error"] = away_model.predict(
    test.loc[away_mask, FEATURES]
)

test["predicted_market_error_cal"] = iso.transform(
    test["predicted_market_error"]
)

test["adjusted_spread"] = test["spread"] + test["predicted_market_error_cal"]

# Edge = model spread minus market spread
test["edge"] = test["predicted_market_error_cal"]

MARGIN_STD = 12

test["p_cover"] = norm.cdf(
    test["predicted_market_error_cal"] / MARGIN_STD
)




# ✅ THIS IS NOW YOUR EDGE
test["edge"] = test["predicted_market_error"]

#resid_std = (y_train - model.predict(X_train)).std()

#print(home_model.feature_importance())
#print(away_model.feature_importance())

test["edge"] = test["predicted_market_error"]

test["p_cover"] = np.where(
    test["favorite_is_home"] == 1,
    norm.cdf(test["edge"] / home_resid_std),
    norm.cdf(test["edge"] / away_resid_std)
)

test = test[test["p_cover"] > 0.5].copy()


test["clv"] = test["edge"]
test["adjusted_spread"] = test["spread"] + test["predicted_market_error"]

def flat_bet(bankroll, stake_pct=0.01):
    return bankroll * stake_pct

def kelly_fraction(p, b, cap=0.05, frac=0.25):
    raw = (p * (b + 1) - 1) / b
    return np.clip(raw * frac, 0, cap)

def full_kelly(p, b, cap=0.10):
    raw = (p * (b + 1) - 1) / b
    return np.clip(raw, 0, cap)

def simulate_strategy(df, sizing, edge_th):
    bankroll = BANKROLL_START
    peak = bankroll
    max_dd = 0

    bets = df[df["edge"] >= edge_th].copy()
    if len(bets) == 0:
        return None

    for _, r in bets.iterrows():
        p = r["p_cover"]
        won = r["favorite_cover"] == 1

        if sizing == "flat":
            stake = flat_bet(bankroll)
        elif sizing == "quarter_kelly":
            stake = bankroll * kelly_fraction(p, b, frac=0.25)
        elif sizing == "full_kelly":
            stake = bankroll * full_kelly(p, b)
        else:
            raise ValueError("Unknown sizing")

        if stake <= 0:
            continue

        profit = stake * b if won else -stake
        bankroll += profit

        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)

    return {
        "edge_threshold": edge_th,
        "sizing": sizing,
        "final_bankroll": bankroll,
        "roi_pct": (bankroll / BANKROLL_START - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "bets": len(bets)
    }

results = []

for edge_th in EDGE_THRESHOLDS:
    for sizing in ["flat", "quarter_kelly", "full_kelly"]:
        out = simulate_strategy(test, sizing, edge_th)
        if out:
            results.append(out)

summary_df = pd.DataFrame(results)
print(summary_df.sort_values(["edge_threshold", "sizing"]))


# season_summary = []

# for season, grp in test.groupby("season"):
#     roi = grp["profit"].sum() / grp["stake"].sum()
#     season_summary.append({
#         "season": season,
#         "roi_pct": roi * 100,
#         "bets": len(grp)
#     })

# season_df = pd.DataFrame(season_summary).sort_values("season")
# print("\n===== ROI BY SEASON =====")
# print(season_df)

# home_away_summary = test.groupby("favorite_is_home").apply(
#     lambda x: pd.Series({
#         "roi_pct": x["profit"].sum() / x["stake"].sum() * 100,
#         "bets": len(x)
#     })
# )

# print("\n===== HOME VS AWAY FAVORITES =====")
# print(home_away_summary)

# test["spread_bucket"] = pd.cut(
#     test["spread"],
#     bins=[0, 3, 6, 10, 20],
#     labels=["small", "medium", "large", "huge"]
# )

# spread_summary = test.groupby("spread_bucket").apply(
#     lambda x: pd.Series({
#         "roi_pct": x["profit"].sum() / x["stake"].sum() * 100,
#         "bets": len(x)
#     })
# )

# print("\n===== ROI BY SPREAD SIZE =====")
# print(spread_summary)


# shuffled = test.copy()
# shuffled["favorite_cover"] = np.random.permutation(shuffled["favorite_cover"])

# roi = shuffled["profit"].sum() / shuffled["stake"].sum()
# print("ROI after label shuffle:", roi)

# print(test.groupby(test["clv"] > 0)["profit"].mean())


bet_cols = [
    "date",
    "season",
    "spread",
    "edge",
    "p_cover",
    "favorite_cover"
]

bets = test[bet_cols].copy()

bankroll = BANKROLL_START
bankroll_curve = []

for _, row in bets.sort_values("date").iterrows():
    if row["edge"] < OPTIMAL_EDGE:
        continue

    stake = bankroll * 0.25 * row["kelly"]
    stake = min(stake, bankroll * 0.02)

    if row["favorite_cover"] == 1:
        bankroll += stake * (100 / abs(ODDS))
    else:
        bankroll -= stake

    bankroll_curve.append(bankroll)

curve = pd.Series(bankroll_curve)
drawdown = (curve.cummax() - curve) / curve.cummax()

print("Max drawdown:", drawdown.max())
