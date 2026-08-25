import os
import time
import pandas as pd
import sys
import numpy as np

from nba_api.stats.static import teams
from sklearn.linear_model import Ridge

# ============================================================
# CONFIG
# ============================================================

INPUT_GAMES = "data/processed/nba_games_with_game_id.csv"
INPUT_PLAYERS = "data/processed/all_boxscores.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROLL_WINDOWS = [5, 10]
MIN_GAMES = 3

def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error between true and predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    
    Returns
    -------
    float
        Mean absolute error
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_true) == 0:
        raise ValueError("y_true is empty, cannot compute MAE")
    
    return np.mean(np.abs(y_true - y_pred))


def parse_minutes(x):
    if pd.isna(x):
        return 0.0
    if ":" in str(x):
        m, s = x.split(":")
        return int(m) + int(s) / 60
    return 0.0

df_players = pd.read_csv(INPUT_PLAYERS)

df_games = pd.read_csv(INPUT_GAMES,
    dtype={"GAME_ID": "string"},parse_dates=["date"]
)

df_players["minutes_played"] = df_players["minutes"].apply(parse_minutes)
df_players["did_play"] = df_players["minutes_played"] > 0
df_players["GAME_ID"] = df_players["GAME_ID"].astype(str)

df_players["GAME_ID"] = (
    df_players["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
)

df_games["GAME_ID"] = df_games["GAME_ID"].astype(str)

df_players = df_players.merge(
    df_games[["GAME_ID", "date", "home", "away","SEASON"]],
    on="GAME_ID",how="left"
)

df_players = df_players.sort_values(["PLAYER_ID", "date"])

df_players["is_home"] = df_players["team"] == df_players["home"]
df_players["points_per_min"] = df_players["points"] / df_players["minutes_played"]
df_players["rebounds_per_min"] = df_players["rebounds"] / df_players["minutes_played"]
df_players["assists_per_min"] = df_players["assists"] / df_players["minutes_played"]

df_players["usage_proxy"] = (
    df_players["fga"] + 0.44 * df_players["fta"] + df_players["turnovers"]
) / df_players["minutes_played"]


df_players["poss_proxy"] = (
    df_players["fga"]
    + 0.44 * df_players["fta"]
    + df_players["turnovers"]
).clip(lower=1)

df_players["pts_per_100"] = df_players["points"] / df_players["poss_proxy"] * 100

df_players["ast_rate"] = df_players["assists"] / df_players["fga"]
df_players["tov_rate"] = df_players["turnovers"] / df_players["poss_proxy"]

df_players["3pa_rate"] = df_players["three_pa"] / df_players["fga"]
df_players["2pa_rate"] = (df_players["fga"] - df_players["three_pa"]) / df_players["fga"]
df_players["ft_rate"]  = df_players["fta"] / df_players["fga"]

df_players["3p_pct"] = df_players["three_pm"] / df_players["three_pa"].replace(0, np.nan)
df_players["2p_pct"] = (df_players["fgm"] - df_players["three_pm"]) / (df_players["fga"] - df_players["three_pa"]).replace(0, np.nan)
df_players["ft_pct"] = df_players["ftm"] / df_players["fta"].replace(0, np.nan)

df_players["oreb_rate"] = df_players["OREB"] / (df_players["OREB"] + df_players["OREB"])
df_players["dreb_rate"] = df_players["DREB"] / (df_players["DREB"] + df_players["DREB"])

for w in ROLL_WINDOWS:
    for stat in ["points", "rebounds", "assists","usage_proxy", "minutes_played", "PLUS_MINUS",
                 "fgm", "fga", "three_pm", "three_pa", "OREB", "DREB", "STL", "BLK", "turnovers"]:
        df_players[f"player_{stat}_r{w}"] = (
            df_players
            .groupby("PLAYER_ID")[stat]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=MIN_GAMES).mean())
        )

ROLL_COLS = [
    "pts_per_100",
    "ast_rate", "tov_rate",
    "3pa_rate", "2pa_rate", "ft_rate",
    "3p_pct", "2p_pct", "ft_pct",
    "oreb_rate", "dreb_rate",
]

for w in ROLL_WINDOWS:
    for col in ROLL_COLS:
        df_players[f"team_{col}_r{w}"] = (
            df_players.groupby("TEAM_ID")[col]
              .shift(1)
              .rolling(w, min_periods=3)
              .mean()
        )

opp_allowed = df_players.copy()

opp_allowed_cols = {
    "pts_per_100": "opp_pts_allowed",
    "tov_rate": "opp_tov_forced",
    "3p_pct": "opp_3p_pct_allowed",
    "2p_pct": "opp_2p_pct_allowed",
    "oreb_rate": "opp_oreb_allowed",
}

opp_allowed = opp_allowed.rename(columns=opp_allowed_cols)

for w in ROLL_WINDOWS:
    for col in opp_allowed_cols.values():
        opp_allowed[f"{col}_r{w}"] = (
            opp_allowed.groupby("TEAM_ID")[col]
              .shift(1)
              .rolling(w, min_periods=3)
              .mean()
        )

df_players = df_players.merge(
    opp_allowed[
        ["GAME_ID", "TEAM_ID"]
        + [f"{c}_r{w}" for c in opp_allowed_cols.values() for w in ROLL_WINDOWS]
    ],
    left_on=["GAME_ID", "OPPONENT_TEAM_ID"],
    right_on=["GAME_ID", "TEAM_ID"],
    suffixes=("", "_opp"),
    how="left"
)

for w in ROLL_WINDOWS:
    df_players[f"3p_matchup_r{w}"] = df_players[f"3p_pct_r{w}"] - df_players[f"opp_3p_pct_allowed_r{w}"]
    df_players[f"tov_matchup_r{w}"] = df_players[f"tov_rate_r{w}"] - df_players[f"opp_tov_forced_r{w}"]
    df_players[f"oreb_matchup_r{w}"] = df_players[f"oreb_rate_r{w}"] - df_players[f"opp_oreb_allowed_r{w}"]

# HOME_COLS = [
#     "pts_per_100_r5",
#     "3p_matchup_r5",
#     "tov_matchup_r5",
# ]

# for col in HOME_COLS:
#     df_players[f"{col}_home_adj"] = df_players[col] * df_players["is_home"]


ROLL = 10  # recent window

player_strength = (
    df_players
    .groupby(["TEAM_ID", "PLAYER_ID"])
    .agg(
        avg_minutes=("minutes_played", "mean"),
        avg_points=("points", "mean"),
        avg_usage=("usage_proxy", "mean"),
    )
    .reset_index()
)

# Normalize within team
player_strength["minutes_z"] = (
    player_strength
    .groupby("TEAM_ID")["avg_minutes"]
    .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
)

player_strength["usage_z"] = (
    player_strength
    .groupby("TEAM_ID")["avg_usage"]
    .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
)

player_strength["key_score"] = (
    0.6 * player_strength["minutes_z"] +
    0.4 * player_strength["usage_z"]
)

player_strength["is_key_player"] = player_strength["key_score"] > 0.75

df_players = df_players.merge(
    player_strength[["TEAM_ID", "PLAYER_ID", "is_key_player"]],
    on=["TEAM_ID", "PLAYER_ID"],
    how="left"
)

df_players = df_players.sort_values(["PLAYER_ID", "GAME_DATE"])

df_players["last_played_date"] = (
    df_players
    .where(df_players["did_play"])
    .groupby("PLAYER_ID")["GAME_DATE"]
    .ffill()
)

df_players["days_since_played"] = (
    df_players["GAME_DATE"] - df_players["last_played_date"]
).dt.days.fillna(30)

# Tunable
HALF_LIFE_DAYS = 7

df_players["injury_decay"] = np.exp(
    -df_players["days_since_played"] / HALF_LIFE_DAYS
)

# Only apply to injury-related absences
df_players["injury_impact"] = np.where(
    df_players["injury_absence"],
    df_players["injury_decay"],
    0.0
)

team_absences = (
    df_players[~df_players["did_play"]]
    .groupby(["GAME_ID", "TEAM_ID"])
    .agg(
        injury_minutes_out=("minutes_played_r5",
                             lambda x: x[df_players.loc[x.index, "injury_absence"]].sum()),

        key_players_out=("is_key_player", "sum"),

        injury_impact_sum=("injury_impact", "sum"),

        rest_minutes_out=("minutes_played_r5",
                           lambda x: x[df_players.loc[x.index, "rest_absence"]].sum()),
    )
    .reset_index()
)

df_team = df_players.copy()

df_team = df_team.merge(
    team_absences,
    on=["GAME_ID", "TEAM_ID"],
    how="left"
).fillna(0)

df_team = df_team.merge(
    team_absences,
    left_on=["GAME_ID", "OPPONENT_TEAM_ID"],
    right_on=["GAME_ID", "TEAM_ID"],
    suffixes=("", "_opp"),
    how="left"
).fillna(0)

df_team["injury_minutes_diff"] = (
    df_team["injury_minutes_out"]
    - df_team["injury_minutes_out_opp"]
)

df_team["key_players_diff"] = (
    df_team["key_players_out"]
    - df_team["key_players_out_opp"]
)

df_team["injury_impact_diff"] = (
    df_team["injury_impact_sum"]
    - df_team["injury_impact_sum_opp"]
)

TEAM_STATS = [
    "points",
    "rebounds",
    "assists",
    "turnovers",
    "three_pa",
    "three_pm",
    "fta",
    "ftm",
]

team_game = (
    df_players[df_players["did_play"]]
    .groupby(["GAME_ID", "TEAM_ID"])
    .sum(numeric_only=True)
    .reset_index()
)

team_game["three_rate"] = team_game["three_pa"] / (
    team_game["three_pa"] + team_game["fga"] + 1e-6
)

team_game["ft_rate"] = team_game["fta"] / (
    team_game["fga"] + 1e-6
)

team_game["ast_to_to"] = team_game["assists"] / (
    team_game["turnovers"] + 1e-6
)


ROLLS = [5, 10]

team_game = team_game.sort_values(["TEAM_ID", "GAME_DATE"])

for r in ROLLS:
    for col in [
        "points", "rebounds", "assists",
        "three_rate", "ft_rate", "ast_to_to"
    ]:
        team_game[f"{col}_r{r}"] = (
            team_game
            .groupby("TEAM_ID")[col]
            .transform(lambda x: x.rolling(r, min_periods=3).mean())
        )

for r in ROLLS:
    for col in [
        "points", "rebounds", "assists",
        "three_rate", "ft_rate", "ast_to_to"
    ]:
        df_team[f"{col}_diff_r{r}"] = (
            df_team[f"{col}_r{r}"] -
            df_team[f"{col}_r{r}_opp"]
        )



team_features = (
    df_players[df_players["did_play"]]
    .sort_values("minutes_played", ascending=False)
    .groupby(["GAME_ID", "team"])
    .head(8)
    .groupby(["GAME_ID", "team"], as_index=False)["points_r5"]
    .mean()
    .rename(columns={"points_r5": "top8_points_r5"})
)

home_feats = team_features.rename(columns={
    "team": "home",
    "top8_points_r5": "home_top8_points_r5"
})

away_feats = team_features.rename(columns={
    "team": "away",
    "top8_points_r5": "away_top8_points_r5"
})

df_model = df_games.merge(home_feats, on=["GAME_ID", "home"])
df_model = df_model.merge(away_feats, on=["GAME_ID", "away"])

df_model = df_model.dropna(
    subset=["home_top8_points_r5", "away_top8_points_r5"]
)
df_model = df_model.merge(df_players, on = ["GAME_ID"])

df_model["top8_points_diff"] = (
    df_model["home_top8_points_r5"] -
    df_model["away_top8_points_r5"]
)

FEATURES = [
    c for c in df_model.columns
    if c.endswith(f"_r{ROLL}")
]
TARGET = "points"

# -----------------------------
# 1. Drop obvious junk columns
# -----------------------------
DROP_COLS = [
    "COMMENT",
    "source_file",
    
    # Player metadata
    "TEAM_CITY", "NICKNAME", "START_POSITION",
    
    # Duplicate home/away/date fields
    "date_y", "home_y", "away_y", "SEASON_y",
    
 
]

df_model = df_model.drop(columns=[c for c in DROP_COLS if c in df_model.columns])

# Player expected contribution
df_players["expected_points"] = (
    df_players["minutes_played_r5"] *
    df_players["points_per_min"]
)

# Mark players who did NOT play
df_players["missed_expected_points"] = np.where(
    df_players["did_play"] == False,
    df_players["expected_points"],
    0
)

injury_impact = (
    df_players
    .groupby(["GAME_ID", "team"], as_index=False)["missed_expected_points"]
    .sum()
)

df_model = df_model.rename(columns=lambda x: x.replace('_x',''))


df_model = df_model.merge(
    injury_impact.rename(columns={"team": "home"}),
    on=["GAME_ID", "home"],
    how="left"
).rename(columns={"missed_expected_points": "home_injury_pts"})

df_model = df_model.merge(
    injury_impact.rename(columns={"team": "away"}),
    on=["GAME_ID", "away"],
    how="left"
).rename(columns={"missed_expected_points": "away_injury_pts"})

df_model[["home_injury_pts", "away_injury_pts"]] = (
    df_model[["home_injury_pts", "away_injury_pts"]].fillna(0)
)

df_model["injury_diff"] = (
    df_model["home_injury_pts"] - df_model["away_injury_pts"]
)

df_model.rename(columns= {"ELO_Rolling_2YR": "ELO_Rolling_2YR_home",
                          "ELO_Rolling_2YR_y": "ELO_Rolling_2YR_away"})

# After final df_model is assembled
df_model["home_flag"] = (df_model["team"] == df_model["home"]).astype(int)


df_model.to_csv("data/processed/player data ml ready.csv", index=False)

