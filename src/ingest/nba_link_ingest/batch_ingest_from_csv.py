"""Batch ingest helper: read game IDs from processed CSV and run ingest_runner.

This script reads `data/processed/nba_games_with_game_id_processed.csv`,
filters games from year 2000 onwards, and runs the ingestion runner in
dry-run for the first `--limit` games (default 10).
"""
from pathlib import Path
import csv
import argparse
from subprocess import run


def read_game_ids(csv_path: Path, start_year: int = 2000):
    ids = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = row.get("date")
                if not d:
                    continue
                year = int(d.split("-")[0])
                if year >= start_year:
                    gid = row.get("GAME_ID") or row.get("game_id")
                    if gid:
                        ids.append(gid)
            except Exception:
                continue
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--start-year", type=int, default=2000)
    args = ap.parse_args()

    csv_path = Path("data/processed/nba_games_with_game_id_processed.csv")
    if not csv_path.exists():
        print("Processed games CSV not found:", csv_path)
        return

    ids = read_game_ids(csv_path, start_year=args.start_year)
    ids = ids[: args.limit]
    if not ids:
        print("No game ids found")
        return

    gid_arg = ",".join(ids)
    # Call ingest_runner in the same Python environment
    cmd = ["python", "src/ingest/nba_link_ingest/ingest_runner.py", "--game-ids", gid_arg, "--dry-run"]
    print("Running:", " ".join(cmd))
    run(cmd)


if __name__ == "__main__":
    main()
