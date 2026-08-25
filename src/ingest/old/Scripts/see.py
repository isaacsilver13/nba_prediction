

from tqdm import tqdm
import math
import pandas as pd

import pandas as pd
from tqdm import tqdm
import time
import sys
import datatable as dt 

INPUT_PATH = "data/processed/df_model.csv"

INPUT_PATH2 = "data/processed/team_game.csv"

df_model = dt.fread(INPUT_PATH)
df_model = df_model.to_pandas()


team_games = dt.fread(INPUT_PATH2)
team_games = team_games.to_pandas()

print(df_model.describe())
print(team_games.describe())

cols = df_model.columns.tolist()

for i in range(0, len(cols), 5):
    batch = cols[i:i+5]
    print(f"\n=== Columns {i}–{i+len(batch)-1} ===")
    print(df_model[batch].describe())

cols = team_games.columns.tolist()

for i in range(0, len(cols), 5):
    batch = cols[i:i+5]
    print(f"\n=== Columns {i}–{i+len(batch)-1} ===")
    print(team_games[batch].describe())

