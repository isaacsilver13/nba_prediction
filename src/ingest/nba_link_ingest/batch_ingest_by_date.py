"""Batch ingest by date using nba_api ScoreboardV2 to discover game ids.

Collects game IDs starting from `--start-date` (YYYY-MM-DD) and proceeds
day-by-day until `--limit` games have been found. Calls the ingest_runner
in dry-run mode for the discovered game ids.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys
from subprocess import run

import json
import csv
import concurrent.futures
from typing import Iterable
from datetime import timezone
import time
from requests.exceptions import ReadTimeout, RequestException
from src.ingest.nba_link_ingest import parser as box_parser


def games_on_date(date_obj: datetime, allow_api: bool = True):
    # Try to use a local mapping of games -> ids if it exists
    # This avoids requiring nba_api to be installed.
    mapping = Path("data/processed/nba_games_with_game_id.csv")
    target_date = date_obj.strftime("%m/%d/%Y")
    if mapping.exists():
        ids = []
        with mapping.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # csv stores date as M/D/YYYY
                if row.get("date") == target_date:
                    gid = str(row.get("GAME_ID") or row.get("game_id") or "").strip()
                    if not gid:
                        continue
                    # convert to full 10-char game id used in cache filenames
                    full_gid = "00" + gid.zfill(8)
                    ids.append(full_gid)
        if ids:
            return ids
        # mapping exists but no ids for this date -> fall through to API fallback

    if not allow_api:
        return []

    # fallback: use nba_api if installed
    try:
        from nba_api.stats.endpoints import scoreboardv2
    except Exception as e:
        raise ImportError("nba_api not available and no local game id mapping found; install nba_api or provide data/processed/nba_games_with_game_id.csv") from e

    # nba_api expects M/D/YYYY or MM/DD/YYYY
    date_str = date_obj.strftime("%m/%d/%Y")
    # Retry with exponential backoff on transient request errors/timeouts
    max_attempts = 3
    backoff_base = 2
    d = None
    for attempt in range(1, max_attempts + 1):
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str)
            d = sb.get_dict()
            break
        except (ReadTimeout, RequestException) as e:
            if attempt == max_attempts:
                raise
            sleep = backoff_base ** attempt
            time.sleep(sleep)
    results = d.get("resultSets", [])
    # find game header
    for rs in results:
        name = rs.get("name", "")
        if "GameHeader" in name or "GameHeader".lower() in name.lower():
            rows = rs.get("rowSet", [])
            headers = rs.get("headers", [])
            if "GAME_ID" in headers:
                idx = headers.index("GAME_ID")
                return [r[idx] for r in rows]
    return []


def _ensure_dirs(paths: Iterable[Path]):
    for p in paths:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)


def _append_csv_row(path: Path, header: list[str], row: list):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def _read_failed_dates(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = list(r)
    # assume single-column header or no header
    dates = set()
    for r in rows:
        if not r:
            continue
        value = r[0].strip()
        if value == "date":
            continue
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            continue
        dates.add(value)
    return dates


def process_date(date_obj: datetime, out_dir: Path, dry_run: bool, workers: int = 8, allow_api: bool = True) -> tuple[bool,int,str]:
    """Process a single date. Returns (success, num_games, message)"""
    date_str = date_obj.strftime("%Y-%m-%d")
    out_path = out_dir / f"{date_str}.json"
    try:
        ids = games_on_date(date_obj, allow_api=allow_api)
    except Exception as e:
        return False, 0, f"ERROR_GAMEIDS:{e}"
    num_games = len(ids)
    if num_games == 0:
        return True, 0, "NO_GAMES"

    appearances_by_game = {}
    # helper: try to load cached per-game player CSVs
    def _load_cached_appearances(gid: str):
        # try several common cache locations
        candidates = [
            Path("data/cache/boxscores") / f"{gid}.csv",
            Path("data/cache/boxscores/traditional") / f"{gid}_player.csv",
            Path("data/cache/boxscores/traditional") / f"{gid}.csv",
        ]
        for c in candidates:
            if c.exists():
                # parse CSV into appearances
                apps = []
                with c.open("r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        try:
                            pid = row.get("PLAYER_ID") or row.get("player_id") or row.get("PLAYER_ID")
                            pname = row.get("PLAYER_NAME") or row.get("player_name") or row.get("name")
                            team_id = row.get("TEAM_ID") or row.get("team_id")
                            team_abbrev = row.get("TEAM_ABBREVIATION") or row.get("TEAM_ABBREV") or row.get("TEAM_ABBREV")
                            minutes = row.get("MIN") or row.get("minutes") or row.get("MIN")
                            started = row.get("START_POSITION") or row.get("START_POSITION") or row.get("START_POSITION")
                            comment = row.get("COMMENT") or row.get("comment") or row.get("DNP")
                            starter = bool(str(started).strip()) if started is not None else False
                            dnp_reason = comment if comment and "DNP" in str(comment).upper() else None
                            apps.append({
                                "player_id": int(pid) if pid else None,
                                "player_name": pname,
                                "team_id": int(team_id) if team_id else None,
                                "team_abbrev": team_abbrev,
                                "minutes": minutes if minutes and minutes != "" else None,
                                "starter": starter,
                                "listed_in_boxscore": True,
                                "dnp_reason": dnp_reason,
                            })
                        except Exception:
                            continue
                return apps
        return None

    # fetch per-game boxscores concurrently
    def _fetch_and_parse(gid: str):
        # prefer cached appearances
        cached = _load_cached_appearances(str(gid))
        if cached is not None:
            return str(gid), cached
        # cache miss
        if not allow_api:
            return str(gid), None
        payload = box_parser.fetch_boxscore(str(gid))
        ap = box_parser.parse_appearances(payload)
        return str(gid), ap

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch_and_parse, gid): gid for gid in ids}
            for fut in concurrent.futures.as_completed(futures):
                gid = futures[fut]
                try:
                    g, ap = fut.result()
                    if ap is not None:
                        appearances_by_game[g] = ap
                    else:
                        # cache miss and api disabled -> treat as missing
                        print(f"Cache miss for game {g} and API disabled; skipping")
                except Exception as e:
                    # if any game fails, record missing and continue
                    # mark this gid as missing by not adding to appearances_by_game
                    print(f"Warning: failed to fetch/parse game {gid}: {e}")
                    continue
    except Exception as e:
        return False, 0, f"ERROR:{e}"

    if dry_run:
        # report how many would be processed from cache/API
        processed = len(appearances_by_game)
        missing = num_games - processed
        status = "DRY_RUN"
        if missing > 0:
            status = f"DRY_RUN_PARTIAL_MISSING_{missing}"
        return True, processed, status

    # write JSON atomically
    # write JSON atomically
    processed = len(appearances_by_game)
    missing = num_games - processed
    if processed == 0:
        return False, 0, "NO_GAMES_PROCESSED"

    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"date": date_str, "games": appearances_by_game, "missing_games": missing}, f, indent=2, default=str)
    tmp.replace(out_path)
    if missing > 0:
        return True, processed, f"PARTIAL_MISSING_{missing}"
    return True, processed, "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=False)
    ap.add_argument("--end-date", required=False)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads for per-game fetches")
    ap.add_argument("--force", action="store_true", help="Reprocess even if date artifact exists")
    ap.add_argument("--dry-run", action="store_true", help="Do not write outputs; only simulate")
    ap.add_argument("--retry-failures", action="store_true", help="Process only dates in failed_lineup_dates.csv")
    ap.add_argument("--no-api", action="store_true", help="Do not call nba_api; enforce cache-only mode")
    args = ap.parse_args()

    # determine date range
    today = datetime.now().date()
    if args.retry_failures:
        # read failed dates and process those
        failed_csv = Path("data/processed/failed_lineup_dates.csv")
        retry_dates = sorted(_read_failed_dates(failed_csv))
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in retry_dates]
    else:
        if args.start_date:
            start = datetime.strptime(args.start_date, "%Y-%m-%d")
        else:
            # default window: last 7 days
            start = datetime.combine(today - timedelta(days=7), datetime.min.time())
        if args.end_date:
            end = datetime.strptime(args.end_date, "%Y-%m-%d")
        else:
            end = datetime.combine(today, datetime.min.time())

        # build list of dates
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d = d + timedelta(days=1)

    out_dir = Path("data/processed/lineups")
    log_dir = Path("data/logs")
    failed_path = Path("data/processed/failed_lineup_dates.csv")
    _ensure_dirs([out_dir, log_dir, failed_path.parent])

    # workers for concurrent fetches
    workers = getattr(args, "workers", 8)

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        out_path = out_dir / f"{date_str}.json"
        if out_path.exists() and not args.force and not args.dry_run:
            # skip
            print(f"Skipped {date_str}: artifact exists")
            _append_csv_row(log_dir / "lineup_ingest_log.csv", ["date","num_games","timestamp","status"], [date_str, 0, datetime.now(timezone.utc).isoformat(), "SKIPPED"])
            continue

        print(f"Processing {date_str}...")
        success, num_games, msg = process_date(d, out_dir, args.dry_run, workers=workers, allow_api=(not args.no_api))
        _append_csv_row(log_dir / "lineup_ingest_log.csv", ["date","num_games","timestamp","status"], [date_str, num_games, datetime.now(timezone.utc).isoformat(), msg])

        if not success:
            # record failed date (avoid duplicates)
            existing = _read_failed_dates(failed_path)
            if date_str not in existing:
                _append_csv_row(failed_path, ["date"], [date_str])
            print(f"Failed {date_str}: {msg}")
        else:
            # if success and the date was previously recorded as failed, remove it
            if failed_path.exists():
                existing = _read_failed_dates(failed_path)
                if date_str in existing:
                    remaining = sorted(existing - {date_str})
                    with failed_path.open("w", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        for r in remaining:
                            w.writerow([r])
        # if limit specified, decrement
        if args.limit:
            args.limit -= 1
            if args.limit <= 0:
                break


if __name__ == "__main__":
    main()
