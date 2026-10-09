# Odds Collector Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six minor review findings deferred from PR #6 so a flaky response, a bad cache file or a flatten bug can neither crash a run nor go unnoticed, and so the smoke test exercises the request shape production uses.

**Architecture:** Small, local changes to `providers.py`, `schedule.py` and `__main__.py`; no new modules. Failures that still leave usable data (a partial SGO pagination, a flatten error) keep the raw file, keep the slot `ok`, and now also exit non-zero so GitHub emails the owner.

**Tech Stack:** Python 3.12 stdlib, pyarrow, pytest (unchanged).

**Spec:** `docs/superpowers/specs/2026-10-08-odds-collector-design.md`. Base plan: `docs/superpowers/plans/2026-10-08-odds-collector.md`.

## The six deferred minors and where each is fixed

| # | Minor (from the PR body) | Task |
|---|---|---|
| 1 | `HTTPException` (e.g. `IncompleteRead`) is not caught | 1 |
| 2 | Non-dict SGO body; earlier pages lost when a later page fails | 1 |
| 3 | A malformed cached schedule event crashes every run | 2 |
| 4 | `captured_at` is the run's start time | 3 |
| 5 | Flatten failures are silent | 4 |
| 6 | Smoke test uses `limit=5`, scheduled runs use `limit=50` + pagination | 5 (needs your OK) |

## Global Constraints

- Everything in the base plan's Global Constraints still holds (stdlib + pyarrow only, keys only from env and redacted, no network in tests, `requests.csv` columns and outcome set unchanged, no data files committed).
- Do not touch `experiment.py`, `autorun.py`, `src/ingest/`, `tools/`.
- Raw `.json.gz` stays the source of truth: no change may make a capture that returned HTTP 200 lose its payload.
- No new `outcome` value. A partial or flatten-failed capture stays `ok` (so it is not re-billed); it is surfaced by `error` text plus exit code 1.
- `attempted_at_utc` keeps meaning "run start / slot evaluation time" (the monthly quota counts and `due()` use it). Only `captured_at_utc` and the file stem change meaning (Task 3).

## Rulings baked into this plan

- **Partial SGO pagination** is kept and logged `ok` with `error="partial: page N failed: ..."`, not retried. Retrying re-bills every page already fetched, and the close window is only 17 minutes. Cost if wrong: a close snapshot silently misses some games, but `error` is non-empty and the run fails loudly.
- **Exit code 1 for partial/flatten** means the workflow run goes red and GitHub emails the owner; the data-repo push step already runs under `!cancelled()` (`odds-collector.yml:54`), so the captured data is still pushed.
- **Corrupt schedule cache** is treated as empty, which makes `run_due` refetch `/events` (self-heals, seeded from yesterday's cache).

## Review Focus

1. An SGO page 2 times out after page 1 was billed → page 1 kept, slot `ok`, run exits 1. Test: `test_sgo_keeps_earlier_pages_when_a_later_page_fails` (Task 1), `test_partial_sgo_capture_is_kept_and_fails_the_run` (Task 4).
2. A provider answers 200 with a JSON array/object of the wrong shape → logged `error`, no traceback. Tests: `test_sgo_non_object_body_is_an_error`, `test_oddsapi_non_list_body_is_an_error` (Task 1).
3. `schedule/<date>.json` is truncated or hand-edited wrong → the next tick rebuilds it instead of crashing every tick until someone notices. Tests: `test_corrupt_schedule_cache_self_heals`, `test_slate_tips_skips_malformed_events` (Task 2).
4. `/events` returns an item with no `id` or a junk `commence_time` → dropped, other games still scheduled. Test: `test_malformed_event_from_events_endpoint_is_dropped` (Task 2).
5. A capture takes minutes (slow pagination) → `minutes_to_tip` reflects when the odds were received. Test: `test_captured_at_is_response_time_not_run_start` (Task 3).

**Running tests** (from the worktree root; the venv lives in the main checkout):

```bash
PY="C:/Users/justj/nba_prediction/repos/nba_prediction/.venv/Scripts/python.exe"
$PY -m pytest tests/test_odds_collector.py -q
```

Baseline before Task 1: 34 passed.

---

### Task 1: Providers return errors instead of raising or discarding pages

**Files:**
- Modify: `src/odds_collector/providers.py` (imports, `_get`, `sgo_events`, `oddsapi_events`, `oddsapi_odds`)
- Test: `tests/test_odds_collector.py` (next to the existing provider tests, after `test_missing_key_raises`; add `import http.client` to the imports)

**Interfaces:**
- Consumes: `providers.Response(status, headers, body, error)`.
- Produces: unchanged signatures. New behavior: `sgo_events` returns `Response(200, h, pages, "partial: page N failed: <err>")` when a later page fails; a non-dict page body or non-list Odds API body returns `body=None` with `error="unexpected body: <type>"` and the original status.

- [ ] **Step 1: Write the failing tests**

```python
def test_truncated_response_is_captured_not_raised(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "NETKEY")

    def fake(req, timeout):
        raise http.client.IncompleteRead(b"par")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.oddsapi_events()
    assert r.status is None and r.body is None and "IncompleteRead" in r.error


def test_sgo_keeps_earlier_pages_when_a_later_page_fails(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    answers = iter([FakeResp({"data": [{"eventID": "a"}], "nextCursor": "c2"}), TimeoutError("slow")])

    def fake(req, timeout):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.sgo_events({})
    assert r.status == 200 and [p["data"][0]["eventID"] for p in r.body] == ["a"]
    assert r.error.startswith("partial: page 2 failed: TimeoutError")


def test_sgo_non_object_body_is_an_error(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout: FakeResp([1, 2]))
    r = providers.sgo_events({})
    assert r.status == 200 and r.body is None and "list" in r.error


def test_oddsapi_non_list_body_is_an_error(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "ODDSKEY999")
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout: FakeResp({"message": "x"}))
    r = providers.oddsapi_odds({"regions": "us"})
    assert r.status == 200 and r.body is None and "dict" in r.error
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q -k "truncated or earlier_pages or non_object or non_list"`
Expected: 4 FAIL (`IncompleteRead` propagates; `AttributeError: 'list' object has no attribute 'get'`; partial test raises `StopIteration`/loses page; `dict` body returned as-is).

- [ ] **Step 3: Implement**

In `providers.py` add `import http.client` with the other imports, then:

```python
    except (OSError, ValueError, http.client.HTTPException) as e:   # URLError, timeouts, bad JSON, truncated reads
```

Add below `_get`:

```python
def _shape_error(r: Response, want: type) -> Response:
    """A 200 whose JSON is the wrong shape is an error, not a crash in whoever reads the body."""
    if r.body is None or isinstance(r.body, want):
        return r
    return r._replace(body=None, error=f"unexpected body: {type(r.body).__name__}")
```

Replace the body of the `sgo_events` loop:

```python
    while True:
        query = {**params, **({"cursor": cursor} if cursor else {})}
        r = _shape_error(_get(f"{SGO_URL}?{urllib.parse.urlencode(query)}", {"x-api-key": key}, key), dict)
        if r.body is None:
            if not pages:
                return r
            # earlier pages are already billed: keep them rather than discard
            return r._replace(status=200, body=pages, error=f"partial: page {len(pages) + 1} failed: {r.error}")
        pages.append(r.body)
        cursor = r.body.get("nextCursor")
        if not (paginate and cursor):
            return r._replace(body=pages)
```

and wrap the two Odds API returns: `return _shape_error(_get(...), list)` in `oddsapi_events` and `oddsapi_odds`.

- [ ] **Step 4: Run to verify everything passes**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: 38 passed.

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/providers.py tests/test_odds_collector.py
git commit -m "fix(odds): providers return errors for truncated/wrong-shape bodies and keep earlier SGO pages"
```

---

### Task 2: A bad schedule cache or event cannot crash a run

**Files:**
- Modify: `src/odds_collector/schedule.py` (`slate_tips`), `src/odds_collector/__main__.py` (`load_schedule`, `refresh_schedule`)
- Test: `tests/test_odds_collector.py` (schedule test after `test_slate_tips_filters_central_date_and_keeps_one_per_game`; CLI tests after `test_schedule_refresh_merges_by_event_id`)

**Interfaces:**
- Consumes: `cli.load_schedule(root, d) -> {"fetched_at_utc": str|None, "events": {id: event}}`, `schedule.slate_tips(events, d) -> list[datetime]`.
- Produces: same signatures; `load_schedule` never raises on a missing/corrupt file; `slate_tips` skips events without a timezone-aware parseable `commence_time`.

- [ ] **Step 1: Write the failing tests**

```python
def test_slate_tips_skips_malformed_events():
    good = ct(2026, 10, 21, 19)
    events = [{"id": "no-time"}, {"commence_time": "garbage"}, {"commence_time": None},
              {"commence_time": "2026-10-21T23:00:00"},   # naive: cannot be placed on the Central calendar
              {"commence_time": good.isoformat()}]
    assert schedule.slate_tips(events, D) == [good]
```

```python
def test_corrupt_schedule_cache_self_heals(tmp_path, monkeypatch):
    path = tmp_path / "schedule" / "2026-10-21.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"fetched_at_utc": "2026-10-21T2', encoding="utf-8")   # truncated mid-write
    monkeypatch.setattr(providers, "oddsapi_events",
                        lambda: providers.Response(200, {}, load_fixture("oddsapi_events.json"), ""))
    fake_providers(monkeypatch)
    assert cli.run_due(tmp_path, NOW) == 0
    assert set(cli.load_schedule(tmp_path, date(2026, 10, 21))["events"]) == {"OA1", "OA2"}


def test_malformed_event_from_events_endpoint_is_dropped(tmp_path, monkeypatch):
    good = {"id": "OA2", "commence_time": "2026-10-22T23:30:00Z", "home_team": "A", "away_team": "B"}
    monkeypatch.setattr(providers, "oddsapi_events",
                        lambda: providers.Response(200, {}, [good, {"commence_time": "x"}, "junk", None], ""))
    sched, rc = cli.refresh_schedule(tmp_path, date(2026, 10, 21), NOW)
    assert rc == 0 and set(sched["events"]) == {"OA2"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q -k "malformed or corrupt"`
Expected: 3 FAIL (`KeyError`/`ValueError` in `slate_tips`; `JSONDecodeError` in `load_schedule`; `TypeError`/`KeyError: 'id'` in `refresh_schedule`).

- [ ] **Step 3: Implement**

`schedule.py`:

```python
def slate_tips(events, d: date) -> list[datetime]:
    """One tip per game (duplicates kept so group sizes count games), sorted, for Central date d.

    Events without a parseable timezone-aware commence_time are skipped: one bad cached event must not stop every run.
    """
    tips = []
    for e in events:
        try:
            t = datetime.fromisoformat(e["commence_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if t.tzinfo is not None and t.astimezone(CT).date() == d:
            tips.append(t)
    return sorted(tips)
```

`__main__.py`:

```python
def load_schedule(root: Path, d: date) -> dict:
    try:
        sched = json.loads(_schedule_path(root, d).read_text(encoding="utf-8"))
        if isinstance(sched["events"], dict):
            return sched
    except (OSError, ValueError, KeyError, TypeError):   # missing or corrupt: refetch /events and rebuild
        pass
    return {"fetched_at_utc": None, "events": {}}
```

and in `refresh_schedule` change the dict comprehension to
`{e["id"]: e for e in r.body if isinstance(e, dict) and e.get("id")}`.

- [ ] **Step 4: Run to verify everything passes**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: 41 passed.

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/schedule.py src/odds_collector/__main__.py tests/test_odds_collector.py
git commit -m "fix(odds): corrupt schedule cache or malformed events are skipped, not fatal"
```

---

### Task 3: `captured_at` is when the response arrived

**Files:**
- Modify: `src/odds_collector/__main__.py` (add `utcnow`, use in `capture` and `main`)
- Test: `tests/test_odds_collector.py` (add an autouse `fixed_clock` fixture under `no_network`; new test in the CLI section)

**Interfaces:**
- Consumes: `capture(root, slot, now, log, paginate, limit)`.
- Produces: `cli.utcnow() -> datetime` (module-level so tests can pin it). In `capture`, `env["captured_at_utc"]` and the file stem use `utcnow()` taken right after the provider call; `attempted_at_utc` and quota logic still use `now`.

- [ ] **Step 1: Write the failing test and the fixture**

```python
@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(cli, "utcnow", lambda: NOW)   # NOW is defined in the CLI section; looked up at call time
```

`NOW` is a module global defined further down the file, which is fine because the lambda resolves it when called.

```python
def test_captured_at_is_response_time_not_run_start(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch)
    later = NOW + timedelta(seconds=90)
    monkeypatch.setattr(cli, "utcnow", lambda: later)
    cli.run_due(tmp_path, NOW)
    oks = [r for r in store.read_log(tmp_path) if r["outcome"] == "ok"]
    assert len(oks) == 2
    for r in oks:
        assert r["attempted_at_utc"] == cli.iso(NOW)
        assert store.read_raw(tmp_path / r["payload_path"])["captured_at_utc"] == cli.iso(later)
        assert cli.iso(later).replace(":", "-") in r["payload_path"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_odds_collector.py -q -k "captured_at or fixed_clock"`
Expected: FAIL with `AttributeError: ... has no attribute 'utcnow'` (monkeypatch of a missing attribute).

- [ ] **Step 3: Implement**

In `__main__.py`, below `iso`:

```python
def utcnow() -> datetime:
    return datetime.now(UTC)
```

In `capture`, directly after the `r = providers...` call in each branch is awkward; instead take the timestamp once after the if/else, before `row.update(...)`:

```python
    got = utcnow()   # when the odds were received; `now` is when the run started
```

then use `got` in the `stem` (`f"{got.astimezone(UTC):%Y-%m-%dT%H-%M-%SZ}_..."`) and `"captured_at_utc": iso(got)`. In `main`, replace `datetime.now(UTC)` with `utcnow()`.

- [ ] **Step 4: Run to verify everything passes**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: 42 passed (existing tests unaffected: the autouse fixture pins the clock to `NOW`).

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/__main__.py tests/test_odds_collector.py
git commit -m "fix(odds): captured_at is the response time, not the run start"
```

---

### Task 4: Partial and flatten-failed captures fail the run

**Files:**
- Modify: `src/odds_collector/__main__.py` (last line of `capture`; docstring of `_rc`), `docs/odds-collector.md` (Operations bullet about failed runs)
- Test: `tests/test_odds_collector.py` (modify `test_flatten_failure_keeps_raw_and_logs_ok`, add a partial test after it)

**Interfaces:**
- Consumes: `Response.error` text from Task 1 (`"partial: ..."`); `row["error"]` already holds `"flatten: ..."`.
- Produces: `capture` returns `1` when the `ok` row carries a non-empty `error`, else `0`.

- [ ] **Step 1: Write the failing tests**

In `test_flatten_failure_keeps_raw_and_logs_ok` change `cli.run_due(tmp_path, NOW)` to `assert cli.run_due(tmp_path, NOW) == 1`. Add:

```python
def test_partial_sgo_capture_is_kept_and_fails_the_run(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    page = load_fixture("sgo_events.json")
    monkeypatch.setattr(providers, "sgo_events", lambda params, paginate=True: providers.Response(
        200, {}, [page], "partial: page 2 failed: TimeoutError: slow"))
    monkeypatch.setattr(providers, "oddsapi_odds", lambda params: providers.Response(200, {}, [], ""))
    assert cli.run_due(tmp_path, NOW) == 1
    sgo = [r for r in store.read_log(tmp_path) if r["provider"] == "sgo" and r["outcome"] == "ok"]
    assert len(sgo) == 1 and sgo[0]["error"].startswith("partial:") and (tmp_path / sgo[0]["payload_path"]).exists()
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q -k "flatten_failure or partial_sgo"`
Expected: 2 FAIL (`assert 0 == 1`).

- [ ] **Step 3: Implement**

Last two lines of `capture`:

```python
    store.append_log(root, {**row, "outcome": "ok", "payload_path": f"{stem}.json.gz"})
    return 1 if row["error"] else 0   # data is saved, but a partial page set or flatten bug must reach the owner
```

Update `_rc`'s docstring to mention it is only for HTTP auth failures, and in `docs/odds-collector.md` Operations replace the "A failed workflow run means an auth error..." bullet with:

```
- A failed workflow run means an auth error (401/403: replace the secret) or a capture that was saved but flagged:
  `requests.csv` rows with `outcome=ok` and a non-empty `error` (`partial: ...` = a later SGO page failed,
  `flatten: ...` = parquet could not be built; fix `flatten.py` then `--rebuild-parquet`). Data is still pushed.
```

Confirm in `.github/workflows/odds-collector.yml` that the push step is still `if: ${{ !cancelled() }}` (it is, line 54).

- [ ] **Step 4: Run to verify everything passes**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: 43 passed.

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/__main__.py docs/odds-collector.md tests/test_odds_collector.py
git commit -m "fix(odds): partial and flatten-failed captures exit non-zero so the owner is emailed"
```

---

### Task 5: Smoke capture uses the production request shape (needs your approval first)

**Why this needs a decision:** the approved spec says the SGO smoke capture is capped at `limit=5` (≤ 5 of 2,500 monthly objects). Production sends `limit=50` with pagination, which the smoke test never exercises; if the free tier rejects `limit=50`, opening night's `open` slot fails three times and is logged missed. Running the smoke test with the real shape costs up to one night's games per call (≈ 5–15 objects), so `capture-sgo` + `capture-sgo-settle` ≈ 30 of 2,500 objects (~1%). **Do not start this task until the user says yes.** If they decline, skip it and instead add to `docs/odds-collector.md` step 6: "after the first scheduled day, confirm the `open` rows in `requests.csv` are `ok` and not `error`."

**Files:**
- Modify: `src/odds_collector/__main__.py` (`SGO_PAGE_LIMIT` constant, `capture` default, `manual_slot`, `main`, `--capture` help), `docs/odds-collector.md` (step 6 sentence), `docs/superpowers/specs/2026-10-08-odds-collector-design.md:148`
- Test: `tests/test_odds_collector.py` (replace `test_manual_capture_caps_sgo_objects`)

**Interfaces:**
- Consumes: `capture(..., paginate=True, limit=50)`, `manual_slot(kind, now)`.
- Produces: `SGO_PAGE_LIMIT = 50`; manual `sgo` slot covers `[now, now + 1 day)` instead of 7 days; `main` no longer passes `paginate=False, limit=5`.

- [ ] **Step 1: Write the failing test (replaces the old one)**

```python
def test_manual_sgo_capture_uses_the_production_request_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    calls = fake_providers(monkeypatch)
    assert cli.main(["--capture", "sgo"]) == 0
    params = calls[0][1]
    assert params["limit"] == "50" and params["_paginate"] is True
    assert params["startsAfter"] == "2026-10-21T23:15:00Z" and params["startsBefore"] == "2026-10-22T23:15:00Z"
    assert store.read_log(tmp_path / "raw" / "odds")[0]["slot"] == "manual"
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_odds_collector.py -q -k manual_sgo`
Expected: FAIL (`'5' != '50'`).

- [ ] **Step 3: Implement**

`__main__.py`: add `SGO_PAGE_LIMIT = 50` next to the quota constants; `capture(..., paginate: bool = True, limit: int = SGO_PAGE_LIMIT)`; in `manual_slot` the final line becomes
`return schedule.Slot(kind, today, "manual", now, now, now, now + timedelta(days=1 if kind == "sgo" else 7))`;
`main`'s capture call becomes `return capture(root, manual_slot(a.capture, now), now, store.read_log(root))`; the `--capture` help text drops "SGO capped at 5 events" and says "SGO: games in the next 24h, same request shape as scheduled runs".

Docs: in `docs/odds-collector.md` step 6 replace the `capture-sgo asks for at most 5 upcoming events (≤ 5 of 2,500 monthly objects)` sentence with "`capture-sgo` requests the next 24 hours of games exactly as a scheduled run does (about one night's games, roughly 5-15 of 2,500 monthly objects; `capture-sgo-settle` costs about the same)". Edit the spec line 148 the same way (replace "`--capture sgo` is capped at `limit=5` events" with "`--capture sgo` requests the next 24 h with the production `limit=50` + pagination").

- [ ] **Step 4: Run to verify everything passes**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: 43 passed.

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/__main__.py docs/odds-collector.md docs/superpowers/specs/2026-10-08-odds-collector-design.md tests/test_odds_collector.py
git commit -m "feat(odds): smoke capture uses the production SGO request shape"
```

---

### Task 6: Verify and update the PR

- [ ] **Step 1:** Run the whole suite: `$PY -m pytest tests/test_odds_collector.py -q` → 43 passed. Then `$PY -m src.odds_collector --dry-run` against an empty `NBA_DATA_DIR` temp dir → prints "schedule stale" and exits 0 with no traceback.
- [ ] **Step 2:** `git grep -n "SGO_API_KEY\|THE_ODDS_API_KEY"` shows no literal key values; `git status` is clean (no data files).
- [ ] **Step 3:** With the user's go-ahead, push the branch (`git push`) so PR #6 picks up the commits, and move the "Deferred minors" list in the PR body to "Resolved" (Task 5 listed as resolved or declined).
