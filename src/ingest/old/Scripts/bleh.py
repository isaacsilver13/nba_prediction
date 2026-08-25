import pandas as pd
df = pd.read_csv('data/processed/bleh.csv', parse_dates=["date"], nrows = 100000)
print(df[["GAME_ID", "date" ,"TEAM_ID"]].describe())