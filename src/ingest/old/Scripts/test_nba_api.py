from nba_api.stats.endpoints import leaguegamelog

gamelog = leaguegamelog.LeagueGameLog(
    season="2023-24",
    season_type_all_star="Regular Season"
)

df = gamelog.get_data_frames()[0]
print(df.head())
