# NBA Link ingestion notes

This document explains how the one-game parser works and how to run it
locally for testing.

Requirements
- Python 3.10+
- pip install nba_api

Run a single-game parse (example):

```bash
python src/ingest/nba_link_ingest/parser.py 0022400001
```

The script will print a JSON array of appearances. Each appearance will
include `player_id`, `player_name`, `team_id`, `team_abbrev`, `minutes`,
`starter`, `listed_in_boxscore`, and `dnp_reason` when available.

Next steps
- Wire parser into the ingestion pipeline: upsert `games`, `players`,
  `teams`, then insert `appearances`, then aggregate pairwise teammate
  events into `player_links`.
