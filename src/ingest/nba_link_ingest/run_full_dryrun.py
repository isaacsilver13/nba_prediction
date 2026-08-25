"""Resumable dry-run ingestion across a date range.

Discover games by day using `ScoreboardV2`, collect game ids between
`start_date` and `end_date`, and run `ingest_runner` in dry-run mode to
produce staging files. Writes a manifest JSON with successes and
failures and supports resume by skipping already-processed game ids.

Usage:
  python src/ingest/nba_link_ingest/run_full_dryrun.py --start-date 2000-01-01 --end-date 2026-06-07 --limit 1000

Defaults:
  start_date=2000-01-01, end_date=today, limit=None (no limit)

Note: This script uses subprocess to call the existing `ingest_runner`.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time
import json
import sys
from subprocess import run, PIPE
import logging
import sqlite3
import os

# Ensure repo root importability
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from nba_api.stats.endpoints import scoreboardv2


def games_on_date(date_obj: datetime):
    date_str = date_obj.strftime("%m/%d/%Y")
    sb = scoreboardv2.ScoreboardV2(game_date=date_str)
    d = sb.get_dict()
    results = d.get("resultSets", [])
    for rs in results:
        name = rs.get("name", "")
        if "GameHeader" in name or name.lower().startswith("game"):
            rows = rs.get("rowSet", [])
            headers = rs.get("headers", [])
            if "GAME_ID" in headers:
                idx = headers.index("GAME_ID")
                return [r[idx] for r in rows]
    return []


def _game_has_staging(game_id: str) -> bool:
    p = Path("data") / "staging" / f"player_pair_events_{game_id}.json"
    return p.exists()


def _game_in_db(game_id: str) -> bool:
    db_path = Path("data") / "nba_links.db"
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM player_pair_events WHERE game_id = ? LIMIT 1", (game_id,))
        r = cur.fetchone()
        conn.close()
        return r is not None
    except Exception:
        return False


def collect_game_ids(start_date: datetime, end_date: datetime, limit: int | None = None, sleep: float = 0.6, dry_run: bool = True):
    ids = []
    d = start_date
    while d <= end_date:
        logging.info("Checking scoreboard for date %s", d.date())
        try:
            day_ids = games_on_date(d)
            if day_ids:
                # filter out game ids that are already staged or in DB depending on mode
                remaining = []
                for gid in day_ids:
                    if dry_run:
                        if _game_has_staging(gid):
                            logging.info("Skipping game %s (staging exists)", gid)
                            continue
                    else:
                        if _game_in_db(gid):
                            logging.info("Skipping game %s (already in DB)", gid)
                            continue
                    remaining.append(gid)

                if not remaining:
                    logging.info("All games for %s already processed; skipping date", d.date())
                else:
                    ids.extend(remaining)

            if limit and len(ids) >= limit:
                return ids[:limit]
        except Exception as e:
            logging.error("Scoreboard failure on %s: %s", d.date(), e)
        d = d + timedelta(days=1)
        time.sleep(sleep)
    return ids


def run_batch(game_ids, dry_run=True, batch_size=50, sleep_between=0.5):
    manifest = {"processed": [], "failed": []}
    staging_dir = Path("data") / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Skip game ids that already have staging files
    to_process = []
    for gid in game_ids:
        p = staging_dir / f"player_pair_events_{gid}.json"
        if p.exists():
            manifest["processed"].append({"game_id": gid, "status": "skipped_existing"})
        else:
            to_process.append(gid)

    # Process in batches to avoid very long command lines
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i : i + batch_size]
        gid_arg = ",".join(map(str, batch))
        cmd = [sys.executable, "src/ingest/nba_link_ingest/ingest_runner.py", "--game-ids", gid_arg]
        if dry_run:
            cmd.append("--dry-run")
        logging.info("Running ingest for batch: %s - %s", batch[0], batch[-1])
        res = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
        logging.info(res.stdout)
        if res.returncode != 0:
            # Mark each gid as failed with stderr
            for gid in batch:
                manifest["failed"].append({"game_id": gid, "error": res.stderr.strip()[:1000]})
        else:
            for gid in batch:
                manifest["processed"].append({"game_id": gid, "status": "processed"})
        time.sleep(sleep_between)

    return manifest


def save_manifest(manifest, out_dir: Path):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"ingest_manifest_{ts}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2000-01-01")
    ap.add_argument("--end-date", default=datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    logging.info("Collecting game ids from %s to %s", start.date(), end.date())
    ids = collect_game_ids(start, end, limit=args.limit, dry_run=args.dry_run)
    logging.info("Discovered %d game ids", len(ids))
    manifest = run_batch(ids, dry_run=args.dry_run, batch_size=args.batch_size)
    manifest_path = save_manifest(manifest, Path("data") / "staging")
    logging.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
