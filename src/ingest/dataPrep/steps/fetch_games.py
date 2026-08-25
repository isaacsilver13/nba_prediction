"""Fetch game-level data from the NBA API and cache results."""

import pandas as pd
import time
from pathlib import Path

from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.library.parameters import LeagueID
from nba_api.stats.library.http import NBAStatsHTTP

try:
    # When imported as a package module
    from ..config import get_config, resolve_step_config
    from ..helpers import load_cached_output, save_output
except ImportError:
    # When loaded dynamically or as direct import
    try:
        from ingest.dataPrep.config import get_config, resolve_step_config
        from ingest.dataPrep.helpers import load_cached_output, save_output
    except ImportError:
        # Fallback for different import contexts
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config import get_config, resolve_step_config
        from helpers import load_cached_output, save_output

TEAM_MAP = {
    "atl": "ATL","bkn": "BKN","bos": "BOS","cha": "CHA","chi": "CHI","cle": "CLE","dal": "DAL",
    "den": "DEN","det": "DET","gs": "GSW","hou": "HOU","ind": "IND","lac": "LAC","lal": "LAL",
    "mem": "MEM","mia": "MIA","mil": "MIL","min": "MIN","no": "NOP","ny": "NYK","okc": "OKC",
    "orl": "ORL","phi": "PHI","phx": "PHX","por": "POR","sa": "SAS","sac": "SAC","tor": "TOR",
    "utah": "UTA","wsh": "WAS"
}

# ==============================
# FUNCTIONS FOR NBA API
# ==============================
def parse_gamecode(gamecode):
    """Parse an NBA API GAMECODE string into components.

    Parameters
    ----------
    gamecode : str
        Gamecode string formatted as YYYYMMDD/AAAHHH.

    Returns
    -------
    tuple[str, str, str]
        Date portion, away abbreviation, home abbreviation.
    """
    date_part, teams = gamecode.split("/")
    away = teams[:3]
    home = teams[3:]
    return date_part, away, home

def fetch_games_for_date(date, headers, timeout, max_retries=5):
    """Fetch NBA games for a given date using ScoreboardV2.

    Parameters
    ----------
    date : datetime-like
        Date to query.
    headers : dict
        HTTP headers for NBA Stats API.
    timeout : int or float
        Request timeout in seconds.
    max_retries : int, default=5
        Number of retry attempts with exponential backoff.

    Returns
    -------
    pd.DataFrame
        Dataframe of game metadata for the date.
    """
    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")

    for attempt in range(max_retries):
        try:
            sb = ScoreboardV2(
                game_date=date_str,
                league_id=LeagueID.default,
                timeout=timeout,
                headers=headers,
            )
            games = sb.get_data_frames()[0]
            games = games.rename(columns={"GAME_DATE_EST": "GAME_DATE"})
            games[["GAME_DATE_STR", "AWAY", "HOME"]] = (
                games["GAMECODE"].apply(lambda x: pd.Series(parse_gamecode(x)))
            )
            games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE_STR"], format="%Y%m%d")
            print(f"Successfully got game id for {date_str}")
            return games[['GAME_ID', 'GAME_DATE', 'GAMECODE', 'HOME', 'AWAY', 'SEASON', 'ARENA_NAME']]

        except Exception as e:
            wait = 2 ** attempt
            NBAStatsHTTP._session = None
            print(f"[Retry {attempt + 1}/{max_retries}] {date_str} – waiting {wait}s")
            time.sleep(wait)

    print(f"[FAILED – EMPTY RETURN] {date_str}")
    return pd.DataFrame(columns=["GAME_ID","GAME_DATE","GAMECODE","HOME","AWAY","SEASON","ARENA_NAME"])

def _load_error_log(error_log_path):
    """Load the error log CSV if it exists.

    Parameters
    ----------
    error_log_path : Path
        CSV path for the error log.

    Returns
    -------
    pd.DataFrame
        Error log with columns `date` and `error`.
    """
    if error_log_path.exists():
        return pd.read_csv(error_log_path, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "error"])


def _prepare_games_df(df, regular_only, min_season, max_season=None):
    """Filter and normalize raw games dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw games dataframe.
    regular_only : bool
        Whether to keep only regular-season games.
    min_season : int or None
        Minimum season threshold to keep.
    max_season : int or None
        Maximum season threshold to keep.

    Returns
    -------
    pd.DataFrame
        Cleaned games dataframe with normalized team codes and dates.
    """
    if regular_only:
        df = df[df["regular"] == True].copy()
    if min_season is not None:
        df = df[df["season"] >= min_season]
    if max_season is not None:
        df = df[df["season"] <= max_season]

    df = df.sort_values("date").reset_index(drop=True)
    df["home"] = df["home"].str.lower().map(TEAM_MAP)
    df["away"] = df["away"].str.lower().map(TEAM_MAP)
    df["date"] = df["date"].dt.normalize()
    return df


def run(config=None, inputs=None):
    """Run step 1 to fetch game IDs and scoreboard metadata.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `games_raw` when provided.

    Returns
    -------
    dict
        Dictionary with `games_with_ids` dataframe.
    """
    cfg = resolve_step_config(get_config(), "step1") if config is None else resolve_step_config(config, "step1")
    paths = cfg["paths"]
    filters = cfg["filters"]
    api_cfg = cfg["api"]
    cache_cfg = cfg.get("cache", {})

    output_path = Path(paths["output_file"])
    processed_fallback_path = Path(paths.get("processed_fallback", "")) if paths.get("processed_fallback") else None
    error_log_path = Path(paths["error_log"])
    retry_output_path = Path(paths["retry_output"])

    # Try to load from cache or fallback
    if not inputs:
        cached_df, from_cache = load_cached_output(str(output_path), cache_cfg)
        if from_cache:
            cached_df["date"] = pd.to_datetime(cached_df["date"])  # Ensure date is parsed
            return {"games_with_ids": cached_df}

        if processed_fallback_path:
            fallback_df, from_fallback = load_cached_output(str(processed_fallback_path), cache_cfg)
            if from_fallback:
                fallback_df["date"] = pd.to_datetime(fallback_df["date"])  # Ensure date is parsed
                return {"games_with_ids": fallback_df}

    if inputs and "games_raw" in inputs:
        df_raw = inputs["games_raw"].copy()
    else:
        df_raw = pd.read_csv(paths["input_file"], parse_dates=["date"])

    df = _prepare_games_df(
        df_raw,
        regular_only=filters.get("regular_only", True),
        min_season=filters.get("min_season"),
        max_season=filters.get("max_season"),
    )

    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    error_log = _load_error_log(error_log_path)

    df_games = df.copy()
    unique_dates = df_games["date"].unique()

    print(f"📅 Fetching scoreboard for {len(unique_dates)} dates")

    all_games = []
    failed_dates = []

    for d in unique_dates:
        daily_games = fetch_games_for_date(
            d,
            headers=api_cfg["headers"],
            timeout=api_cfg.get("timeout", 40),
            max_retries=api_cfg.get("max_retries", 5),
        )
        if not daily_games.empty:
            all_games.append(daily_games)
        else:
            failed_dates.append(pd.to_datetime(d))
        time.sleep(api_cfg.get("request_sleep", 0.25))

    if failed_dates:
        error_log = pd.concat(
            [
                error_log,
                pd.DataFrame(
                    {"date": failed_dates, "error": ["scoreboard_fetch_failed"] * len(failed_dates)}
                ),
            ],
            ignore_index=True,
        )
        error_log.drop_duplicates("date").to_csv(error_log_path, index=False)

    games_lookup = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame(
        columns=["GAME_ID", "GAME_DATE", "GAMECODE", "HOME", "AWAY", "SEASON", "ARENA_NAME"]
    )
    print(f"✅ Collected {len(games_lookup):,} games")
    print(f"⚠️ Failed dates logged: {error_log['date'].nunique()}")

    games_lookup["GAME_DATE"] = pd.to_datetime(games_lookup["GAME_DATE"]).dt.normalize()
    df_games = df_games.merge(
        games_lookup,
        left_on=["date", "home", "away"],
        right_on=["GAME_DATE", "HOME", "AWAY"],
        how="left",
    )

    missing = df_games["GAME_ID"].isna().sum()
    print(f"🔍 Missing GAME_IDs after merge: {missing}")

    final_cols = [
        "GAME_ID",
        "date",
        "season",
        "regular",
        "home",
        "away",
        "score_home",
        "score_away",
        "spread",
        "whos_favored",
        "ot_home",
        "ot_away",
        "ARENA_NAME",
        "id_spread",
    ]
    df_games = df_games[final_cols].copy().sort_values("date").reset_index(drop=True)

    # Save to cache if enabled
    save_output(df_games, str(output_path), cache_cfg)
    if missing > 0:
        print("🔍 Missing GAME_IDs found, retrying failed dates")
    else:
        print("💾 Saved games with GAME_ID")

    if missing > 0 and error_log_path.exists():
        failed_dates = pd.read_csv(error_log_path, parse_dates=["date"])["date"].unique()
        retry_games = []

        for d in failed_dates:
            daily_games = fetch_games_for_date(
                d,
                headers=api_cfg["headers"],
                timeout=api_cfg.get("timeout", 40),
                max_retries=api_cfg.get("max_retries", 5),
            )
            if not daily_games.empty:
                retry_games.append(daily_games)
            time.sleep(api_cfg.get("request_sleep", 0.25))

        if retry_games and cache_cfg.get("enabled") and cache_cfg.get("write_outputs"):
            retry_df = pd.concat(retry_games, ignore_index=True)
            save_output(retry_df, str(retry_output_path), cache_cfg)

    return {"games_with_ids": df_games}


if __name__ == "__main__":
    run(get_config())


