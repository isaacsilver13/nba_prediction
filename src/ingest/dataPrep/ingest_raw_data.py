"""
Ingestion Pipeline: Fetches raw game and player data from NBA API.

This pipeline focuses on collecting raw data:
- Step 1: Fetch game data (games, scores, spreads, etc)
- Step 3: Fetch player-level box score data by game
- Step 3a: Roll player data up to team level
"""

import importlib.util
import os
import sys

try:
    from .config import get_config
    from .steps import fetch_games, fetch_player_boxscores, aggregate_player_to_team
    PACKAGE_DIR = os.path.dirname(__file__)
except ImportError:  # pragma: no cover
    PACKAGE_DIR = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from ingest.dataPrep.config import get_config
    from ingest.dataPrep.steps import fetch_games, fetch_player_boxscores, aggregate_player_to_team


def run_ingest_pipeline(config=None):
    """Run ingestion steps (1, 3, 3a) to build raw and team-level data.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.

    Returns
    -------
    dict
        Dictionary with game-level data and aggregated box score outputs.
    """
    cfg = get_config() if config is None else config
    outputs = {}

    # Step 1: Fetch game data
    print("=" * 60)
    print("Step 1: Fetching game data from NBA API")
    print("=" * 60)
    outputs.update(fetch_games.run(cfg))

    # Step 3: Fetch player box score data
    print("\n" + "=" * 60)
    print("Step 3: Fetching player box score data")
    print("=" * 60)
    outputs.update(fetch_player_boxscores.run(cfg, outputs))

    # Step 3a: Roll up player data to team level
    print("\n" + "=" * 60)
    print("Step 3a: Rolling up player data to team level")
    print("=" * 60)
    outputs.update(aggregate_player_to_team.run(cfg, outputs))

    return outputs


if __name__ == "__main__":
    run_ingest_pipeline()
