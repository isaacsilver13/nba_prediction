---
description: Run or troubleshoot date-based NBA Link lineup ingestion using existing artifacts, logs, and failed-date retries.
---

Operate the existing date-based lineup ingestion workflow in `src/ingest/nba_link_ingest/batch_ingest_by_date.py`.

1. Use the repository root as the working directory and select either an explicit `--start-date`/`--end-date` range or `--retry-failures`.
2. Treat `data/processed/lineups/YYYY-MM-DD.json` as the date artifact, `data/logs/lineup_ingest_log.csv` as the concise run log, and `data/processed/failed_lineup_dates.csv` as the retry queue.
3. Existing date artifacts should be skipped unless the user requests `--force`. Use `--dry-run` or `--no-api` when validating behavior without new network calls.
4. Use official boxscore appearances, including listed DNPs. Do not add player-stat endpoint calls or player-stat feature collection to this workflow.
5. Before editing, inspect whether the issue is date discovery, cache reconstruction, boxscore parsing, partial-game handling, logging, or failed-date persistence. Preserve atomic artifact writes and concurrent per-game fetching.
6. Validate a single known date or a temporary failure-list scenario first. Summarize processed, skipped, partial, and failed dates and link the artifact/log paths.
