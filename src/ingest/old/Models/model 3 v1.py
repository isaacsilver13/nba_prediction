import pandas as pd
import numpy as np
from scipy.stats import norm
import lightgbm as lgb

# =========================
# CONFIG
# =========================

SIGMA_MARGIN = 12
ODDS = -110
PAYOUT = 100 / 110
MAX_KELLY = 0.02
BANKROLL_START = 10_000

ELO_REGRESSION = 0.75
ROLL_WINDOWS = [5, 10]

TARGET = "favorite_cover_manual"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/nba_games_with_elo.csv",
    parse_dates=["date"]
)

df = df[df[TARGET].isin([0,1])].copy()
df = df.sort_values("date").reset_index(drop=True)

# =========================
# SEASON-START ELO REGRESSION
# =========================

league_avg_elo = 1500

df["elo_home_regressed"] = df["elo_home"]
df["elo_away_regressed"] = df["elo_away"]

for season in df["season"].unique():
    season_mask = df["season"] == season
    prev_season = season - 1

    if prev_season not in df["season"].values:
        continue

    last_elos = (
        df[df["season"] == prev_season]
        .groupby("home")["elo_home"]
        .last()
    )

    for team, elo in last_elos.items():
        regressed = ELO_REGRESSION * elo + (1 - ELO_REGRESSION) * league_avg_elo
        df.loc[
            season_mask & (df["home"] == team),
            "elo_home_regressed"
        ] = regressed

        df.loc[
            season_mask & (df["away"] == team),
            "elo_away_regressed"
        ] = regressed

df["elo_diff"] = df["elo_home_regressed"] - df["elo_away_regressed"]

# =========================
# REST / B2B FEATURES
# =========================

for side in ["home", "away"]:
    df[f"{side}_b2b"] = df[f"{side}_rest"] == 0

df["rest_advantage"] = df["home_rest"] - df["away_rest"]

# =========================
# ROLLING TEAM STATS
# =========================

def add_rolling_features(df, team_col, margin_col):
    for w in ROLL_WINDOWS:
        df[f"{team_col}_margin_last_{w}"] = (
            df.groupby(team_col)[margin_col]
            .shift(1)
            .rolling(w)
            .mean()
        )

add_rolling_features(df, "home", "home_margin")
add_rolling_features(df, "away", "away_margin")

df["rolling_margin_diff_5"] = (
    df["home_margin_last_5"] - df["away_margin_last_5"]
)

df["rolling_margin_diff_10"] = (
    df["home_margin_last_10"] - df["away_margin_last_10"]
)

# =========================
# FEATURE SET (NO ODDS)
# =========================

FEATURES = [
    "elo_diff",
    "home_flag",
    "rest_advantage",
    "home_b2b",
    "away_b2b",
    "rolling_margin_diff_5",
    "rolling_margin_diff_10"
]

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

# =========================
# TRAIN / TEST SPLIT
# =========================

split_date = "2021-01-01"
train = df["date"] < split_date
test  = df["date"] >= split_date

X_train = df.loc[train, FEATURES]
y_train = df.loc[train, TARGET]
X_test  = df.loc[test, FEATURES]
y_test  = df.loc[test, TARGET]

# =========================
# LIGHTGBM
# =========================

train_data = lgb.Dataset(X_train, label=y_train)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.85,
    "seed": 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=400
)

# =========================
# PREDICTIONS → EV
# =========================

df_test = df.loc[test].copy()

df_test["p_favorite"] = model.predict(X_test)
df_test["predicted_margin"] = norm.ppf(df_test["p_favorite"]) * SIGMA_MARGIN

df_test["p_cover"] = norm.cdf(
    (df_test["predicted_margin"] - df_test["spread"]) / SIGMA_MARGIN
)

df_test["ev"] = (
    df_test["p_cover"] * PAYOUT -
    (1 - df_test["p_cover"])
)

df_test["bet"] = df_test["ev"] > 0

df_test["kelly_frac"] = (
    (df_test["p_cover"] * (PAYOUT + 1) - 1) / PAYOUT
).clip(0, MAX_KELLY)

# =========================
# BACKTEST
# =========================

bankroll = BANKROLL_START
curve = []

for _, r in df_test.iterrows():
    if not r["bet"]:
        curve.append(bankroll)
        continue

    stake = bankroll * r["kelly_frac"]

    if r[TARGET] == 1:
        bankroll += stake * PAYOUT
    else:
        bankroll -= stake

    curve.append(bankroll)

df_test["bankroll"] = curve

roi = (bankroll - BANKROLL_START) / BANKROLL_START * 100

# =========================
# RESULTS
# =========================

print("\n===== ATS MODEL V2 RESULTS =====")
print(f"Final bankroll: ${bankroll:,.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Bets placed: {df_test['bet'].sum()}")

print("\nFeature importance:")
print(
    pd.Series(model.feature_importance(), index=FEATURES)
    .sort_values(ascending=False)
)
