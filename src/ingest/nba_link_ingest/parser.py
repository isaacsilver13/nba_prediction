"""NBA Link ingestion: boxscore -> appearances parser

This module provides a small, testable parser that uses `nba_api` to
fetch a game's boxscore and extract the list of players "listed in the
boxscore" for each team. The ingestion rule (per spec) is: if a player
is listed in the official box score for that game (including DNPs),
they count as an appearance and create teammate edges.

Functions:
- fetch_boxscore(game_id) -> raw payload from nba_api
- parse_appearances(boxscore_payload) -> List[dict] of appearances

The code is defensive: if `nba_api` is unavailable the functions raise
an ImportError with an actionable message.
"""
from typing import Any, Dict, List, Optional
import datetime
import json


def _ensure_nba_api():
    try:
        from nba_api.stats.endpoints import boxscoretraditionalv2
        return boxscoretraditionalv2
    except Exception as e:
        raise ImportError(
            "nba_api is required for ingestion. Install with: pip install nba_api"
        )


def fetch_boxscore(game_id: str) -> Dict[str, Any]:
    """Fetch a game's traditional boxscore payload using nba_api.

    Args:
        game_id: NBA game id string (e.g., '0022400001')

    Returns:
        Raw payload dict as returned by the endpoint's get_dict().
    """
    boxscoretraditionalv2 = _ensure_nba_api()
    endpoint = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    return endpoint.get_dict()


def parse_appearances(boxscore_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse appearances from a boxscore payload.

    The structure returned by `BoxScoreTraditionalV2.get_dict()` contains a
    `resultSets` list with rows for PlayerStats. We conservatively treat any
    player row present in the boxscore as "listed in boxscore". If a player
    has an empty or missing `MIN` field, we include them and set minutes=None
    and infer `dnp_reason` if available.

    Returns a list of appearances with these keys:
      - player_id (int)
      - player_name (str)
      - team_id (int)
      - team_abbrev (str)
      - minutes (str|None)
      - starter (bool)  # based on START_POSITION or other heuristic
      - listed_in_boxscore (bool)
      - dnp_reason (Optional[str])

    Note: different NBA API endpoints may vary; keep parsing defensive.
    """
    appearances: List[Dict[str, Any]] = []

    result_sets = boxscore_payload.get("resultSets") or []
    # Find the player stats result set (commonly named 'PlayerStats')
    player_rows = None
    headers = None
    for rs in result_sets:
        name = rs.get("name", "")
        if name.lower().startswith("player") or "PLAYER" in name.upper():
            headers = rs.get("headers", [])
            player_rows = rs.get("rowSet", [])
            break

    if player_rows is None or headers is None:
        # Fallback: try to inspect all resultSets for likely player rows
        for rs in result_sets:
            if rs.get("rowSet") and rs.get("headers") and len(rs.get("headers")) > 5:
                headers = rs.get("headers")
                player_rows = rs.get("rowSet")
                break

    if headers is None or player_rows is None:
        raise ValueError("Could not locate player rows in boxscore payload")

    # Build index map
    idx = {h: i for i, h in enumerate(headers)}

    def val(row, name):
        i = idx.get(name)
        if i is None:
            return None
        return row[i]

    for row in player_rows:
        try:
            player_id = val(row, "PLAYER_ID") or val(row, "PLAYER_ID")
            player_name = val(row, "PLAYER_NAME") or val(row, "PLAYER")
            team_id = val(row, "TEAM_ID") or val(row, "TEAM_ID")
            team_abbrev = val(row, "TEAM_ABBREVIATION") or val(row, "TEAM_ABBREVIATION")
            minutes = val(row, "MIN")
            starter = False
            # Some payloads expose START_POSITION or STARTED
            started = val(row, "START_POSITION") or val(row, "START_POSITION") or val(row, "STARTED")
            if started is not None:
                # START_POSITION is non-empty for starters (e.g. 'G', 'F')
                starter = bool(str(started).strip())

            dnp_reason = None
            # Some endpoints include DNP reason in COMMENT or DNP_REASON
            dnp_reason = val(row, "DNP") or val(row, "COMMENT") or val(row, "DNP_REASON")

            appearance = {
                "player_id": int(player_id) if player_id is not None else None,
                "player_name": player_name,
                "team_id": int(team_id) if team_id is not None else None,
                "team_abbrev": team_abbrev,
                "minutes": minutes if minutes is not None and minutes != "" else None,
                "starter": starter,
                "listed_in_boxscore": True,
                "dnp_reason": dnp_reason,
            }
            appearances.append(appearance)
        except Exception:
            # Keep parsing robust: skip problematic rows
            continue

    return appearances


def appearances_to_json(appearances: List[Dict[str, Any]]) -> str:
    return json.dumps(appearances, default=str, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse one game's boxscore into appearances")
    parser.add_argument("game_id", help="NBA game id (e.g., 0022400001)")
    args = parser.parse_args()

    try:
        payload = fetch_boxscore(args.game_id)
        ap = parse_appearances(payload)
        print(appearances_to_json(ap))
    except Exception as e:
        print("Error:", e)
        raise
