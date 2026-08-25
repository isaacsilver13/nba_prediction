import pandas as pd
import numpy as np
import lightgbm as lgb

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Path to preprocessed, game-level dataset
# IMPORTANT: This dataset must only contain PRE-GAME features
DATA_PATH = "data/processed/nba_games_with_elo.csv"

# Walk-forward cutoff
# All seasons BEFORE this are training
# All seasons AT OR AFTER this are test (simulated live betting)
TRAIN_END_SEASON = 2022

# -------------------------
# MODEL FEATURES
# -------------------------
# These are all known before tip-off
# Spread is included intentionally as the market anchor
FEATURES = [
    "spread",                 # Vegas consensus opinion
    "elo_diff",               # favorite Elo advantage
    "rolling_margin_diff_5",  # recent performance proxy
    "rolling_margin_diff_10",
    "rest_advantage",         # fatigue asymmetry
    "favorite_is_home"        # structural bias feature
]

# Target we model:
# How wrong the market was, in POINTS, from favorite POV
TARGET = "market_error"

# Quantiles used to approximate the conditional distribution
# of market error (non-parametric, no Gaussian assumption)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# -------------------------
# BETTING CONFIG
# -------------------------
ODDS = -110

# Decimal payout multiplier for ATS (-110)
# Bet $1 → win $0.909 if correct
B = 100 / abs(ODDS)

# Starting bankroll for simulation
BANKROLL_START = 10_000

# Hard Kelly cap (VERY IMPORTANT)
# Prevents model error from blowing up bankroll
MAX_KELLY = 0.03   # 3% max per bet

# Minimum edge thresholds (in POINTS)
EDGE_THRESHOLDS = np.arange(0.5, 4.5, 0.5)

# ============================================================
# 2. LOAD AND PREP DATA
# ============================================================

# Load dataset and ensure chronological order
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Drop rows with missing required inputs
# (usually early-season rolling stats)
df = df.dropna(subset=FEATURES + [TARGET, "favorite_cover"])

# Walk-forward split
train = df[df["season"] < TRAIN_END_SEASON].copy()
test  = df[df["season"] >= TRAIN_END_SEASON].copy()

# ============================================================
# 3. TRAIN QUANTILE REGRESSION MODELS
# ============================================================

def train_quantile_model(train_df, quantile):
    """
    Train a LightGBM quantile regressor to estimate
    the conditional quantile of market_error.

    Each model answers:
    "Given these pre-game features, what is the Xth percentile
     of how wrong Vegas will be?"
    """

    X = train_df[FEATURES]
    y = train_df[TARGET]

    dtrain = lgb.Dataset(X, label=y)

    params = {
        # Quantile regression objective
        "objective": "quantile",

        # Target quantile (alpha ∈ (0,1))
        "alpha": quantile,

        # Conservative learning rate for stability
        "learning_rate": 0.05,

        # Moderate tree complexity
        "num_leaves": 31,

        # Prevent overfitting on small sample slices
        "min_data_in_leaf": 50,

        # Feature subsampling
        "feature_fraction": 0.9,

        "verbosity": -1
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=400
    )

    return model


# ------------------------------------------------------------
# Separate models for HOME favorites and AWAY favorites
# ------------------------------------------------------------
# This avoids forcing one distribution to fit two regimes
models = {
    "home": {},
    "away": {}
}

for q in QUANTILES:
    # Home favorites
    models["home"][q] = train_quantile_model(
        train[train["favorite_is_home"] == 1],
        q
    )

    # Away favorites
    models["away"][q] = train_quantile_model(
        train[train["favorite_is_home"] == 0],
        q
    )

# ============================================================
# 4. PREDICT CONDITIONAL DISTRIBUTIONS (TEST SET)
# ============================================================

def predict_distribution(row):
    """
    For a single game, predict the full conditional
    distribution of market_error using quantiles.

    Output:
        {0.10: value, 0.25: value, ..., 0.90: value}
    """
    side = "home" if row["favorite_is_home"] == 1 else "away"
    preds = {}

    for q in QUANTILES:
        preds[q] = models[side][q].predict(
            row[FEATURES].values.reshape(1, -1)
        )[0]

    return preds


# Apply distribution prediction row-by-row
dist_preds = test.apply(predict_distribution, axis=1)

# Convert list-of-dicts into dataframe
dist_df = pd.DataFrame(dist_preds.tolist(), index=test.index)
dist_df.columns = [f"q_{int(q*100)}" for q in QUANTILES]

# Attach quantile columns to test set
test = pd.concat([test, dist_df], axis=1)
test["edge_points"] = test["q_50"]

# ============================================================
# 5. COMPUTE PROBABILITY OF COVER
# ============================================================

def prob_cover_from_quantiles(row):
    """
    Estimate P(market_error > 0), i.e. probability that
    the favorite covers the spread.

    Uses linear interpolation of the empirical CDF
    implied by predicted quantiles.

    No normality assumption.
    """

    qs = np.array(QUANTILES)
    vals = np.array([row[f"q_{int(q*100)}"] for q in QUANTILES])

    # If entire predicted distribution is below zero
    if vals[-1] <= 0:
        return 0.0

    # If entire predicted distribution is above zero
    if vals[0] >= 0:
        return 1.0

    # Interpolate CDF at 0
    # CDF(x=0) ≈ quantile level
    cdf_0 = np.interp(0, vals, qs)
    return 1.0 - cdf_0



# Apply probability calculation
test["p_cover"] = test.apply(prob_cover_from_quantiles, axis=1)

print(test["p_cover"].describe())

# ============================================================
# 6. EDGE CALCULATION
# ============================================================

# Market-implied cover probability for -110
p_market = 1 / (1 + B)

# Probability edge (what Kelly actually uses)
test["edge_prob"] = test["p_cover"] - p_market

# Point edge (median predicted market error)
# Used for filtering / confidence
test["edge_points"] = test["q_50"]

# ============================================================
# 7. KELLY BET SIZING
# ============================================================

def kelly_fraction(p, b, cap=MAX_KELLY, frac=0.25):
    """
    Fractional Kelly betting.

    p   = model probability of cover
    b   = payout multiplier
    cap = hard maximum fraction of bankroll
    frac= Kelly fraction (0.25 = quarter Kelly)
    """
    raw = (p * (b + 1) - 1) / b
    return np.clip(raw * frac, 0, cap)

# ============================================================
# 8. BACKTEST SIMULATION
# ============================================================

def simulate(df, edge_th):
    """
    Simulate betting strategy for a given edge threshold.
    """
    bankroll = BANKROLL_START
    peak = bankroll
    max_dd = 0
    bets = 0

    # Only consider games where model disagrees
    # with market by at least edge_th points
    for _, r in df[df["edge_points"] >= edge_th].iterrows():

        # Must beat market-implied probability
        if r["p_cover"] <= p_market:
            continue

        stake_frac = kelly_fraction(r["p_cover"], B)
        if stake_frac <= 0:
            continue

        stake = bankroll * stake_frac
        bets += 1

        if r["favorite_cover"] == 1:
            bankroll += stake * B
        else:
            bankroll -= stake

        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)

    return {
        "edge_threshold": edge_th,
        "final_bankroll": bankroll,
        "roi_pct": (bankroll / BANKROLL_START - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "bets": bets
    }


# Run simulation across thresholds
results = [simulate(test, th) for th in EDGE_THRESHOLDS]
summary = pd.DataFrame(results)

print("\n===== STRATEGY SUMMARY =====")
print(summary)

test["actual_cover_rate"] = (
    test["favorite_cover"].astype(int)
)

print(test.groupby(pd.qcut(test["p_cover"], 10))["actual_cover_rate"].mean())
