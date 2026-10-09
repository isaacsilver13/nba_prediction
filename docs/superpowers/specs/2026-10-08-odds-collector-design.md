# Forward-Only NBA Odds Collector — Design

Status: approved in conversation 2026-10-08, pending written-spec review
Scope: `repos/nba_prediction` — capture pre-game NBA odds snapshots from free API tiers, starting by 2026-27 opening night (~2026-10-20).

## Goal and context

The encompassing test (2026-10-08) showed the box-score spread model adds ~nothing beyond the closing spread, so research pivots to softer markets: player props, period (1H/quarter) lines, and moneylines. No free historical source exists for props or period lines, so lines not captured live in 2026-27 are lost permanently. This collector records them. It does not model, bet, or touch `experiment.py`.

The user bets from Illinois and line-shops across US books; per-bet fees differ by book (FanDuel $0.50 waived ≥ $25, DraftKings $0.50 waived for singles ≥ $50, Fanatics $0.25, others reportedly none), so **bookmaker identity is stored on every row**.

Supersedes the transport/storage parts of the untracked `docs/odds-api-free-collection-plan.md` and `docs/odds-api-free-implementation-plan.md` (main checkout). Decisions kept from them: key only from env, raw payload + request audit first, atomic writes, quota guard, `--dry-run`, never part of the experiment pipeline, no data in this repo. Changed: hosting (laptop → GitHub Actions), multiple providers/markets, parquet instead of CSV, adapter/game-matching deferred.

## Provider facts (verified from docs 2026-10-08)

**SportsGameOdds (SGO), free "Amateur" tier** — [pricing](https://sportsgameodds.com/pricing), [FAQ](https://sportsgameodds.com/docs/faq), [getEvents](https://sportsgameodds.com/docs/endpoints/getEvents)
- 2,500 objects/month, 10 requests/min, odds updated every ~10 min. An *object* is a top-level item returned — for `/events`, one **event** — regardless of how many markets/books it contains. Each response counts as at least 1 object.
- Includes player props and "Partials (1st half)". Since billing is per event, period markets cost nothing extra.
- Base `https://api.sportsgameodds.com/v2`, header `x-api-key`. oddID = `{statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}`, e.g. `points-PLAYER_ID-game-ou-over`, `points-home-1h-sp-home`, `points-all-game-ou-over`.
- Free-tier books (pricing page): FanDuel, DraftKings, BetMGM, Caesars, ESPN BET (now theScore Bet), Bovada, Unibet, PointsBet, William Hill. **Illinois-legal and live: FanDuel, DraftKings, BetMGM, Caesars, ESPN BET/theScore.** William Hill duplicates Caesars; Bovada is offshore; Unibet exited the US; PointsBet US became Fanatics. Actual returned books are confirmed in the smoke test.
- `includeOpenCloseOdds=true` adds `openOdds/closeOdds/openSpread/closeSpread/openOverUnder/closeOverUnder` per book "where available"; whether the free tier returns them, and whether `finalized=true` past events are queryable on it, is unknown → tested by the smoke capture and the `settle` slot.

**The Odds API, free tier** — [v4 guide](https://the-odds-api.com/liveapi/guides/v4/), [bookmakers](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html)
- 500 credits/month. Bulk `/v4/sports/basketball_nba/odds` cost = markets × regions; `h2h,spreads,totals` × `us` = **3 credits**. Props/period markets need the per-event endpoint (markets × regions per event) — unaffordable, excluded.
- `/v4/sports/basketball_nba/events` is quota-free and returns `x-requests-used/-remaining/-last` headers.
- `us` region adds two Illinois books SGO lacks: **BetRivers, Fanatics** (plus DK/FD/MGM/Caesars and offshore books).

Neither free tier carries bet365.

## Architecture

```
cron-job.org (every 5 min, 14:00–04:59 UTC = 9 AM–11 PM CT across DST)
   └─ POST GitHub workflow_dispatch ──▶ GitHub Actions job (public repo → free minutes)
                                          ├─ checkout code (this repo)
                                          ├─ sparse checkout data repo → $NBA_DATA_DIR/raw/odds
                                          ├─ python -m src.odds_collector --run-due
                                          └─ commit + push new files to data repo
GitHub `schedule:` cron every 30 min = backup trigger (idempotent; duplicate triggers find nothing due)
Laptop: git clone/pull the data repo into $NBA_DATA_DIR/raw/odds (already gitignored via /data/raw/)
```

- `concurrency: {group: odds-collector, cancel-in-progress: false}` prevents overlapping runs; no lock file.
- `workflow_dispatch`-triggered runs are not affected by GitHub's 60-day inactivity disabling of scheduled workflows; the backup cron may be disabled by it, which is acceptable.

### Code (this repo, stdlib + pyarrow only)

```
src/odds_collector/
  __init__.py
  schedule.py   # pure: (now, slate, request log rows) -> due slots + newly missed slots. No I/O.
  providers.py  # sgo_events(), oddsapi_events(), oddsapi_odds() via urllib; key from env; returns (status, headers, body)
  flatten.py    # pure: provider payload -> list[dict] tidy rows (SGO, Odds API)
  store.py      # atomic .json.gz + .parquet writes, request-log append/read, month-to-date usage
  __main__.py   # CLI: --dry-run | --run-due | --capture PROVIDER SLOT | --rebuild-parquet
tests/test_odds_collector.py
tests/fixtures/odds_collector/{sgo_events.json, oddsapi_events.json, oddsapi_odds.json}   # synthetic, no keys
.github/workflows/odds-collector.yml
docs/odds-collector.md   # setup + operations runbook
```

Not imported by `experiment.py`, `autorun.py`, or `src/ingest/dataPrep`. `pyarrow` is added to `requirements.txt` (already present in the local venv, 25.0.1); the workflow installs only `pyarrow`.

### Data repo layout (private repo, e.g. `isaacsilver13/nba-odds-raw`)

```
requests.csv
schedule/2026-10-21.json                                   # cached Odds API /events slate
sgo/date=2026-10-21/2026-10-21T15-02-11Z_open_<rid>.json.gz
sgo/date=2026-10-21/2026-10-21T15-02-11Z_open_<rid>.parquet
oddsapi/date=2026-10-21/2026-10-22T00-41-05Z_close@1900_<rid>.json.gz
oddsapi/date=2026-10-21/2026-10-22T00-41-05Z_close@1900_<rid>.parquet
```

`date=` is the Central slate date (Hive-style, so `pd.read_parquet("sgo/")` yields a `date` column). `<rid>` is a UUID4 generated before the request.

**Raw envelope** (`.json.gz`): `{request_id, provider, slot, slate_date, captured_at_utc, request: {endpoint, params (key removed)}, http_status, quota_headers, payload}` where `payload` is the exact provider body. Raw is the source of truth; parquet is derived.

**`requests.csv`** — one row per HTTP attempt or slot outcome:
`request_id, provider, slate_date, slot, attempted_at_utc, endpoint, http_status, outcome, error, objects, quota_used, quota_remaining, payload_path`
`outcome ∈ {ok, empty, error, missed, skipped_quota}`. This file is the audit trail, the SGO quota counter, and the slot-completion state.

**SGO parquet columns:** `request_id, slot, captured_at_utc, minutes_to_tip, event_id, starts_at_utc, home_team, away_team, odd_id, stat_id, stat_entity_id, period_id, bet_type_id, side_id, bookmaker_id, odds_american, line, available, last_updated_at, open_odds, open_line, close_odds, close_line`
(`line` = the book's `spread` or `overUnder`; oddID parsed as `rsplit('-', 3)` for period/bet/side, then the head split on the first `-` into stat/entity; `odd_id` is kept verbatim.)

**Odds API parquet columns:** `request_id, slot, captured_at_utc, minutes_to_tip, event_id, commence_time_utc, home_team, away_team, bookmaker_key, bookmaker_last_update, market_key, market_last_update, outcome_name, point, price_american`

Team names, player IDs, and book IDs are stored verbatim. No mirroring, defaulting, or de-vigging.

## Snapshot schedule

All times America/Chicago (`zoneinfo`). *Slate* = games whose start falls on a Central calendar date, from the Odds API `/events` (quota-free), cached at `schedule/<date>.json` and re-fetched when missing or older than 2 h.

A slot is **due** when `now` is inside its window and the log has no `ok` row for `(provider, slate_date, slot)` and fewer than 3 `error|empty` rows. When a window closes without `ok`, the next run appends a `missed` row. Late observations are never relabeled.

**SGO** — every request: `leagueID=NBA`, `includeOpenCloseOdds=true`, no alt lines, paginate via `cursor`.

| Slot | Window (CT) | Filter | Purpose |
|---|---|---|---|
| `open` | 10:00–12:00 | slate start range, `started=false` | early full-game + period lines |
| `props` | 15:00–17:00 | slate start range, `started=false` | props mostly posted by mid-afternoon |
| `close@HHMM` (one per distinct tip time) | tip−20 → tip−3 min (captured on the first tick, ≈ tip−20) | `startsAfter=tip−5m`, `startsBefore=tip+5m` | near-close, after most lineup news |
| `settle` | next day 10:00–12:00 | previous slate range, `finalized=true` | tests free-tier `closeOdds`; final results for prop settlement |

≈ 4 objects/game/day → ~850/month at peak. **Guard: no SGO request when month-to-date `objects` (UTC calendar month, from `requests.csv`) ≥ 2,300** → `skipped_quota`. A response with zero events is `empty` (retried).

**The Odds API** — `/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american`, filtered with `commenceTimeFrom/To` to the slate.

| Slot | Window (CT) |
|---|---|
| `open` | 10:00–12:00 |
| `close@HHMM` per tip *group* | first tip of group −20 → −3 min |

Tip groups: sorted tip times merged while within 30 min of the group's first tip. At most **3 close slots/day**; if more groups exist, keep the 3 with the most games, the rest are logged `missed`. Max 12 credits/day (~370/month). **Guard: no `/odds` request when the latest `x-requests-used` in `requests.csv` + 3 > 450** → `skipped_quota`. Every Odds API call logs that header, including the free `/events` refresh (≤ 2 h old), so the value is near-live and picks up the provider's monthly reset. `/odds` requests use `commenceTimeFrom = max(slate start, now)` so in-play games are never captured as pre-game.

The free SGO tier lags ≤ 10 min, so an SGO close is effectively ~20–30 min pre-tip; `captured_at_utc` and `minutes_to_tip` make this explicit.

## Error handling

- `urlopen` timeout 20 s; no in-run retries (the next tick retries).
- 401/403: logged `error`, and the process exits non-zero so the workflow fails and GitHub emails the user.
- 429/5xx/timeouts: logged `error`, retried next tick (max 3 per slot).
- Raw envelope is written (tmp file + `os.replace`) before flattening. If flattening raises, the capture stays `ok`, the error text goes in `error`, and `--rebuild-parquet` can regenerate later.
- Push conflict: one `git pull --rebase` and retry (concurrency group makes this unlikely).

## Security

- Keys only from env (`SGO_API_KEY`, `THE_ODDS_API_KEY`), injected from GitHub secrets. SGO key is sent as the `x-api-key` header; The Odds API key must be a query param and is stripped from every stored/logged param dict and error string.
- No `pull_request` trigger, so fork PRs never receive secrets. Workflow `permissions: contents: read`; data-repo writes only via `ODDS_DATA_TOKEN` (fine-grained, contents read/write on the data repo only).
- cron-job.org holds a separate fine-grained token with Actions read/write on this repo only (worst case if leaked: someone triggers or cancels runs).
- The user creates every account, key, token, secret, and the cron-job.org job; the agent never handles key values.

## Testing

pytest, synthetic fixtures, `tmp_path`; an autouse fixture makes `urllib.request.urlopen` raise so no test touches the network.

- schedule: window open/close, idempotency (an `ok` slot is never due again), retry cap, `missed` emission, tip grouping and 3-cap selection, DST-change dates, weekend early tips, `settle` targets the previous slate.
- providers/store: key never appears in stored params or errors; quota header parsing; atomic gzip round-trip; request-log append/read; SGO month-to-date object sum.
- flatten: SGO oddID parsing incl. player IDs and `points+rebounds`, missing open/close fields, multiple books; Odds API rows; `--rebuild-parquet` equals the original parquet.
- CLI: `--dry-run` reports due slots from cached files, writes nothing, calls no network.

CI (`.github/workflows/ci.yml`) is extended to run `pytest tests/test_odds_collector.py` with `pyarrow` installed.

## Rollout

1. Implement on branch, open PR.
2. User creates: SGO account/key, Odds API key, private data repo, `ODDS_DATA_TOKEN`, repo secrets, cron-job.org job + dispatch token (steps in `docs/odds-collector.md`).
3. Merge (GitHub only dispatches workflows that exist on the default branch; the job is skipped until the `ODDS_DATA_REPO` variable is set, so merging is inert). **After explicit user approval:** one manual `workflow_dispatch` capture per provider (`--capture sgo` is capped at `limit=5` events). Verify: returned bookmaker IDs, presence of `1h`/`1q` oddIDs and props, presence of `openOdds/closeOdds`, actual object usage vs `/account/usage`, raw + parquet + log rows in the data repo, key absent from all artifacts.
4. Enable cron-job.org. Preseason games (if listed by the providers) are the burn-in before opening night.

## Out of scope (follow-ups)

Kalshi (separate session), Illinois-book filtering, game-ID joins/adapters, de-vigging, alt lines, any modeling or `experiment.py` change.

## Risks

- Free tiers can change limits or books mid-season → quota guards + per-run log; the smoke test re-verifies before launch.
- GitHub Actions / cron-job.org outage → backup cron (every 30 min, so it can miss a 17-min close window); outages surface as `missed` rows rather than mislabeled data. On the first fetch of a day the schedule cache is seeded from yesterday's, so games that tipped during an outage still get `missed` rows.
- Data repo growth (~0.5–1 GB/season, gzip + parquet) — within GitHub's soft limit; sparse checkout keeps runs fast.
- SGO start times may differ from The Odds API's by a few minutes → ±5 min close filter; zero-event responses are `empty` and retried.
