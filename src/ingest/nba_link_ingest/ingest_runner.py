"""Ingestion runner for NBA Link (dry-run friendly).

This script accepts one or more NBA `game_id`s and produces per-game
pairwise teammate events by grouping parsed appearances by team. It
supports a `--dry-run` mode that writes JSON files under
`data/staging/player_pair_events_<game_id>.json` instead of writing to a
database, allowing easy verification before wiring up DB upserts.

Usage examples:
  python src/ingest/nba_link_ingest/ingest_runner.py --game-ids 0022400001

"""
from __future__ import annotations
import json
import os
import itertools
import argparse
from typing import List, Dict, Any
import sys
from pathlib import Path
import logging
import sqlite3

# Ensure project root is on sys.path so `src` imports work when running as script
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.ingest.nba_link_ingest import parser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _ensure_db(db_path: str | Path = "data/nba_links.db") -> sqlite3.Connection:
    db_path = str(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_pair_events (
            game_id TEXT,
            team TEXT,
            player_a INTEGER,
            player_b INTEGER,
            PRIMARY KEY (game_id, player_a, player_b)
        )
        """
    )
    conn.commit()
    return conn

def build_pair_events(appearances: List[Dict[str, Any]], game_id: str) -> List[Dict[str, Any]]:
    """Build pair events for teammates listed in the boxscore.

    For each team in the game, make unordered pairs (player_a, player_b)
    where player_a_id < player_b_id to keep canonical ordering.
    """
    by_team: Dict[str, List[Dict[str, Any]]] = {}
    for a in appearances:
        team = a.get("team_abbrev") or str(a.get("team_id"))
        by_team.setdefault(team, []).append(a)

    events: List[Dict[str, Any]] = []
    for team, players in by_team.items():
        ids = [p["player_id"] for p in players if p.get("player_id") is not None]
        # create all unordered combinations
        for a_id, b_id in itertools.combinations(sorted(set(ids)), 2):
            events.append({
                "game_id": game_id,
                "team": team,
                "player_a": int(a_id),
                "player_b": int(b_id),
            })

    return events


def write_staging(game_id: str, events: List[Dict[str, Any]]) -> str:
    out_dir = os.path.join("data", "staging")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"player_pair_events_{game_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    return path


def run_for_game(game_id: str, dry_run: bool = True) -> Dict[str, Any]:
    payload = parser.fetch_boxscore(game_id)
    appearances = parser.parse_appearances(payload)
    events = build_pair_events(appearances, game_id)
    result = {"game_id": game_id, "num_players": len(appearances), "num_events": len(events)}
    if dry_run:
        path = write_staging(game_id, events)
        result["staging_path"] = path
        logging.info("Wrote staging for %s: %s events -> %s", game_id, len(events), path)
    else:
        # Write to sqlite DB with an upsert (INSERT OR IGNORE on primary key)
        db = _ensure_db()
        before = db.total_changes
        cur = db.cursor()
        rows = [(e["game_id"], e.get("team"), int(e["player_a"]), int(e["player_b"])) for e in events]
        cur.executemany(
            "INSERT OR IGNORE INTO player_pair_events (game_id, team, player_a, player_b) VALUES (?,?,?,?)",
            rows,
        )
        db.commit()
        after = db.total_changes
        inserted = after - before
        result["db_inserted"] = inserted
        logging.info("Upserted %d rows for %s into DB %s", inserted, game_id, db)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-ids", help="Comma-separated game ids to ingest")
    ap.add_argument("--game-list-file", help="Path to newline-delimited file with game ids")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Write staging JSON instead of DB (default: off)",
    )
    args = ap.parse_args()

    game_ids: List[str] = []
    if args.game_ids:
        game_ids.extend([g.strip() for g in args.game_ids.split(",") if g.strip()])
    if args.game_list_file:
        with open(args.game_list_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    game_ids.append(s)

    if not game_ids:
        print("No game ids provided. Use --game-ids or --game-list-file.")
        return

    for gid in game_ids:
        try:
            res = run_for_game(gid, dry_run=args.dry_run)
            print(f"Processed {gid}: players={res['num_players']} pairs={res['num_events']} staging={res.get('staging_path')}")
        except Exception as e:
            print(f"Failed {gid}: {e}")


if __name__ == "__main__":
    main()
