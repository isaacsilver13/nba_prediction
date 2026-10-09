# NBA odds collector — setup and operations

Captures pre-game NBA odds from the SportsGameOdds (SGO) and The Odds API free tiers into a private
data repo. Design: `docs/superpowers/specs/2026-10-08-odds-collector-design.md`.

```
cron-job.org (every 5 min) -> GitHub workflow_dispatch -> Actions runs --run-due -> push to private data repo
GitHub schedule (every 30 min) = backup trigger
laptop: git pull the data repo into $NBA_DATA_DIR/raw/odds
```

## One-time setup (you do these; never paste keys into chat)

1. **SportsGameOdds key** — sign up at https://sportsgameodds.com/pricing (free "Amateur" plan) and copy the API key.
2. **The Odds API key** — sign up at https://the-odds-api.com (free 500-credit plan).
3. **Data repo** — create a **private** GitHub repo, e.g. `isaacsilver13/nba-odds-raw`, ticking
   "Add a README" (the workflow cannot check out an empty repo).
4. **Data token** — GitHub → Settings → Developer settings → Fine-grained tokens → Generate:
   repository access *Only select repositories* → the data repo; permissions → *Contents: Read and write*.
   Set expiry after 2027-07-01 (season end) and calendar a renewal.
5. **Repo secrets/variables** — in `nba_prediction` → Settings → Secrets and variables → Actions:
   - Secrets: `SGO_API_KEY`, `THE_ODDS_API_KEY`, `ODDS_DATA_TOKEN`
   - Variable: `ODDS_DATA_REPO` = `isaacsilver13/nba-odds-raw`

   The workflow is skipped until `ODDS_DATA_REPO` exists.
6. **Smoke test** (only after the PR is merged — GitHub dispatches workflows from the default branch):
   ```
   gh workflow run odds-collector.yml -f mode=dry-run
   gh workflow run odds-collector.yml -f mode=capture-oddsapi
   gh workflow run odds-collector.yml -f mode=capture-sgo
   gh workflow run odds-collector.yml -f mode=capture-sgo-settle
   ```
   `capture-sgo` asks for at most 5 upcoming events (≤ 5 of 2,500 monthly objects). Then pull the data repo
   and check: which `bookmaker_id`s appear, whether `period_id` includes `1h`/`1q`, whether player props appear,
   whether `open_odds`/`close_odds` are filled, and that no key appears anywhere (`git grep -i <first 6 chars>`).
7. **cron-job.org** — create a second fine-grained token: *Only select repositories* → `nba_prediction`;
   permissions → *Actions: Read and write*; expiry after 2027-07-01. Then on https://cron-job.org create a job:
   - URL: `https://api.github.com/repos/isaacsilver13/nba_prediction/actions/workflows/odds-collector.yml/dispatches`
   - Method `POST`; body `{"ref":"main"}`
   - Headers: `Accept: application/vnd.github+json`, `Authorization: Bearer <dispatch token>`,
     `X-GitHub-Api-Version: 2022-11-28`, `Content-Type: application/json`
   - Schedule: every 5 minutes, hours 14–23 and 0–4, time zone UTC (= 9 AM–11 PM Central all season)
   - Enable failure notifications. A successful dispatch returns HTTP 204.

## Laptop sync

```powershell
git clone https://github.com/isaacsilver13/nba-odds-raw "$env:NBA_DATA_DIR\raw\odds"   # once
git -C "$env:NBA_DATA_DIR\raw\odds" pull                                                 # any time
```

```python
import pandas as pd
sgo = pd.read_parquet(r"C:\...\data\raw\odds\sgo")          # every SGO snapshot; `date` column from folders
oddsapi = pd.read_parquet(r"C:\...\data\raw\odds\oddsapi")
```

## Operations

- `requests.csv` is the audit log: one row per HTTP attempt or slot outcome
  (`ok`, `empty` = provider returned no games, `error`, `missed` = window passed without a capture,
  `skipped_quota`). A day with no rows at all means no run happened (trigger outage).
- `python -m src.odds_collector --dry-run` (locally, after `git pull`) prints what is due/missed right now.
- A failed workflow run means an auth error (401/403: replace the secret) or a capture that was saved but flagged:
  `requests.csv` rows with `outcome=ok` and a non-empty `error` (`partial: ...` = a later SGO page failed,
  `flatten: ...` = parquet could not be built; fix `flatten.py` then `--rebuild-parquet`). Data is still pushed.
- Quota stops: SGO 2,300 objects per calendar month (counted from `requests.csv`); The Odds API 450 credits.
- If SGO renames fields: fix `src/odds_collector/flatten.py`, then locally
  `python -m src.odds_collector --rebuild-parquet` and commit/push the data repo.
- To pause collection: disable the cron-job.org job and the workflow (`gh workflow disable odds-collector.yml`).
