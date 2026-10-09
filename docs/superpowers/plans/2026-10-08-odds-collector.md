# Forward-Only NBA Odds Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture pre-game NBA odds snapshots (props, period lines, full-game lines) from the SportsGameOdds and The Odds API free tiers, every day from 2026-27 opening night, as raw `.json.gz` plus tidy parquet in a private data repo.

**Architecture:** A stdlib+pyarrow package `src/odds_collector` (pure schedule policy, pure flatteners, an HTTP boundary, a storage layer, a CLI). GitHub Actions runs `--run-due` whenever cron-job.org dispatches the workflow (every 5 min) or the 30-min backup cron fires; the run captures only due slots and pushes new files to a private data repo checked out at `$NBA_DATA_DIR/raw/odds`.

**Tech Stack:** Python 3.12 stdlib (`urllib`, `zoneinfo`, `csv`, `gzip`), `pyarrow`, pytest, GitHub Actions, cron-job.org.

**Spec:** `docs/superpowers/specs/2026-10-08-odds-collector-design.md`

## Global Constraints

- Python 3.12; runtime deps: stdlib + `pyarrow` only (no `requests`, no pandas at capture time).
- Keys only from env: `SGO_API_KEY`, `THE_ODDS_API_KEY`. SGO key sent as `x-api-key` header; Odds API key appears only in the request URL and is redacted from every stored/logged string.
- Never import from or modify `experiment.py`, `autorun.py`, `src/ingest/`. No data files committed to this repo.
- Data root: `Path(os.environ.get("NBA_DATA_DIR", <repo>/data)).resolve() / "raw" / "odds"` (matches `experiment.py:102`).
- Slate/window times are America/Chicago wall clock; stored timestamps are UTC ISO `YYYY-MM-DDTHH:MM:SSZ`.
- Quota stops: SGO 2,300 objects per UTC month (from `requests.csv`); Odds API latest logged `x-requests-used` + 3 > 450.
- Slot windows: SGO `open` 10:00–12:00, `props` 15:00–17:00, `close@HHMM` tip−20→tip−3 (request tip±5 min; final-review fix, was tip−40), `settle` next day 10:00–12:00 `finalized=true`; Odds API `open` 10:00–12:00, `close@HHMM` per 30-min tip group, max 3/day (largest groups). Max 3 failures per slot.
- `requests.csv` columns, in order: `request_id, provider, slate_date, slot, attempted_at_utc, endpoint, http_status, outcome, error, objects, quota_used, quota_remaining, payload_path`; `outcome ∈ {ok, empty, error, missed, skipped_quota}`.
- No test may touch the network (autouse fixture makes `urlopen` raise).

## Review Focus

1. A provider drops a started game from `/events`, or moves a tip time → the cached slate must keep the game and take the new time (merge by event id). Test: `test_schedule_refresh_merges_by_event_id` (Task 5).
2. A capture window passes with no run at all (cron-job.org/Actions outage) → exactly one `missed` row, not one per later tick. Tests: `test_missed_reports_closed_windows_once` (Task 2), `test_run_due_captures_logs_and_is_idempotent` (Task 5).
3. A live SGO payload with missing/renamed/garbage fields → null columns, never a crash; and if flattening does crash, raw is still saved and the slot is `ok`. Tests: `EVT2`/`EVT3` in `test_sgo_rows` (Task 4), `test_flatten_failure_keeps_raw_and_logs_ok` (Task 5).
4. A provider error body that echoes the API key → `***` in `requests.csv`. Test: `test_http_error_text_is_redacted` (Task 3).
5. UTC month rollover → SGO quota counts only this month's objects. Test: `test_sgo_objects_counts_only_sgo_this_utc_month` (Task 1).

**Running tests** (from the worktree root; the venv lives in the main checkout):

```bash
PY="C:/Users/justj/nba_prediction/repos/nba_prediction/.venv/Scripts/python.exe"
$PY -m pytest tests/test_odds_collector.py -q
```

---

### Task 1: Storage layer

**Files:**
- Create: `src/odds_collector/__init__.py`
- Create: `src/odds_collector/store.py`
- Create: `tests/test_odds_collector.py`
- Modify: `requirements.txt` (append `pyarrow`)

**Interfaces:**
- Produces: `store.LOG_FIELDS: list[str]`, `store.data_root() -> Path`, `store.atomic_write(path: Path, data: bytes) -> None`, `store.write_raw(path: Path, envelope: dict) -> None`, `store.read_raw(path: Path) -> dict`, `store.write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None`, `store.read_log(root: Path) -> list[dict]` (all values `str`), `store.append_log(root: Path, row: dict) -> None`, `store.sgo_objects_this_month(rows: list[dict], now: datetime) -> int`, `store.oddsapi_used(rows: list[dict]) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_odds_collector.py`:

```python
"""Tests for src.odds_collector — synthetic fixtures only; no test may touch the network."""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.odds_collector import store

FIXTURES = Path(__file__).parent / "fixtures" / "odds_collector"
UTC = timezone.utc


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a network call")
    monkeypatch.setattr("urllib.request.urlopen", refuse)


# --- store -----------------------------------------------------------------

def test_data_root_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    assert store.data_root() == tmp_path.resolve() / "raw" / "odds"


def test_raw_roundtrip_is_deterministic_and_leaves_no_tmp(tmp_path):
    env = {"request_id": "r1", "payload": [{"a": 1}]}
    p = tmp_path / "sgo" / "date=2026-10-21" / "x.json.gz"
    store.write_raw(p, env)
    first = p.read_bytes()
    store.write_raw(p, env)
    assert p.read_bytes() == first
    assert store.read_raw(p) == env
    assert [f.name for f in p.parent.iterdir()] == ["x.json.gz"]


def test_log_append_and_read(tmp_path):
    store.append_log(tmp_path, {"request_id": "a", "provider": "sgo", "outcome": "ok", "objects": 3})
    store.append_log(tmp_path, {"request_id": "b", "provider": "oddsapi", "outcome": "error", "error": None})
    rows = store.read_log(tmp_path)
    assert [r["request_id"] for r in rows] == ["a", "b"]
    assert rows[0]["objects"] == "3" and rows[1]["error"] == ""
    assert list(rows[0]) == store.LOG_FIELDS


def test_read_log_missing_file_is_empty(tmp_path):
    assert store.read_log(tmp_path / "nope") == []


def test_sgo_objects_counts_only_sgo_this_utc_month():
    rows = [
        {"provider": "sgo", "attempted_at_utc": "2026-10-31T23:59:00Z", "objects": "100"},
        {"provider": "sgo", "attempted_at_utc": "2026-11-01T00:05:00Z", "objects": "7"},
        {"provider": "sgo", "attempted_at_utc": "2026-11-02T15:00:00Z", "objects": ""},
        {"provider": "oddsapi", "attempted_at_utc": "2026-11-02T15:00:00Z", "objects": "9"},
    ]
    assert store.sgo_objects_this_month(rows, datetime(2026, 11, 2, 16, tzinfo=UTC)) == 7


def test_oddsapi_used_takes_latest_logged_header():
    rows = [{"provider": "oddsapi", "quota_used": "40"}, {"provider": "sgo", "quota_used": ""},
            {"provider": "oddsapi", "quota_used": "43"}, {"provider": "oddsapi", "quota_used": ""}]
    assert store.oddsapi_used(rows) == 43
    assert store.oddsapi_used([]) == 0


def test_parquet_with_no_rows_keeps_schema(tmp_path):
    schema = pa.schema([("a", pa.string()), ("b", pa.float64())])
    store.write_parquet(tmp_path / "x.parquet", [], schema)
    assert pq.read_table(tmp_path / "x.parquet").schema.equals(schema)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'src.odds_collector'`.

- [ ] **Step 3: Implement**

`src/odds_collector/__init__.py`:

```python
"""Forward-only NBA odds collector (SportsGameOdds + The Odds API free tiers). See docs/odds-collector.md."""
```

`src/odds_collector/store.py`:

```python
"""Storage for the odds collector: atomic raw/parquet/json writes and the requests.csv log."""
import csv
import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOG_FIELDS = ["request_id", "provider", "slate_date", "slot", "attempted_at_utc", "endpoint", "http_status",
              "outcome", "error", "objects", "quota_used", "quota_remaining", "payload_path"]


def data_root() -> Path:
    base = Path(os.environ.get("NBA_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))).resolve()
    return base / "raw" / "odds"


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_raw(path: Path, envelope: dict) -> None:
    atomic_write(path, gzip.compress(json.dumps(envelope, sort_keys=True).encode(), mtime=0))


def read_raw(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
    atomic_write(path, sink.getvalue().to_pybytes())


def read_log(root: Path) -> list[dict]:
    p = root / "requests.csv"
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_log(root: Path, row: dict) -> None:
    # ponytail: plain append, no lock; the workflow's concurrency group guarantees a single writer.
    p = root / "requests.csv"
    root.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in LOG_FIELDS})


def sgo_objects_this_month(rows: list[dict], now: datetime) -> int:
    month = now.astimezone(timezone.utc).strftime("%Y-%m")
    return sum(int(float(r.get("objects") or 0)) for r in rows
               if r.get("provider") == "sgo" and (r.get("attempted_at_utc") or "").startswith(month))


def oddsapi_used(rows: list[dict]) -> int:
    for r in reversed(rows):
        if r.get("provider") == "oddsapi" and r.get("quota_used"):
            return int(float(r["quota_used"]))
    return 0
```

Append one line to `requirements.txt`:

```
pyarrow
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/__init__.py src/odds_collector/store.py tests/test_odds_collector.py requirements.txt
git commit -m "feat(odds): storage layer — atomic raw/parquet writes, request log, quota counters"
```

---

### Task 2: Schedule policy

**Files:**
- Create: `src/odds_collector/schedule.py`
- Modify: `tests/test_odds_collector.py` (change import to `from src.odds_collector import schedule, store`; add `from src.odds_collector.schedule import CT`; append tests)

**Interfaces:**
- Produces: `schedule.CT: ZoneInfo`, `schedule.Slot` (frozen dataclass: `provider: str, slate_date: date, name: str, opens: datetime, closes: datetime, starts_after: datetime, starts_before: datetime, finalized: bool = False, capped: bool = False`, property `key -> (provider, slate_date.isoformat(), name)`), `schedule.ct(d: date, hour: int, minute: int = 0) -> datetime`, `schedule.slate_tips(events: Iterable[dict], d: date) -> list[datetime]` (events need `commence_time`), `schedule.group_tips(tips) -> list[list[datetime]]`, `schedule.slots_for(d: date, tips: list[datetime]) -> list[Slot]`, `schedule.due(slots, log, now) -> list[Slot]`, `schedule.missed(slots, log, now) -> list[Slot]`.

- [ ] **Step 1: Write the failing tests**

Update the imports at the top of `tests/test_odds_collector.py`:

```python
from src.odds_collector import schedule, store
from src.odds_collector.schedule import CT
```

Append:

```python
# --- schedule --------------------------------------------------------------

def ct(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=CT)


D = date(2026, 10, 21)
TIPS = [ct(2026, 10, 21, 18), ct(2026, 10, 21, 18), ct(2026, 10, 21, 18, 30), ct(2026, 10, 21, 21)]
CAPPED_TIPS = ([ct(2026, 10, 21, 12)] + [ct(2026, 10, 21, 15)] * 2 + [ct(2026, 10, 21, 18)] * 3
               + [ct(2026, 10, 21, 21)] * 2)


def names(slots, provider=None):
    return sorted(s.name for s in slots if provider in (None, s.provider))


def test_slate_tips_filters_central_date_and_keeps_one_per_game():
    events = [{"commence_time": "2026-10-21T23:00:00Z"}, {"commence_time": "2026-10-21T23:00:00Z"},
              {"commence_time": "2026-10-22T02:00:00Z"},   # 21:00 CT, still the 21st's slate
              {"commence_time": "2026-10-22T23:00:00Z"}]   # next slate
    assert schedule.slate_tips(events, D) == [ct(2026, 10, 21, 18), ct(2026, 10, 21, 18), ct(2026, 10, 21, 21)]


def test_slots_for_a_normal_night():
    slots = schedule.slots_for(D, TIPS)
    assert names(slots, "sgo") == ["close@1800", "close@1830", "close@2100", "open", "props", "settle"]
    assert names(slots, "oddsapi") == ["close@1800", "close@2100", "open"]   # 18:00+18:30 share a call
    c = next(s for s in slots if s.key == ("sgo", "2026-10-21", "close@1830"))
    assert (c.opens, c.closes) == (ct(2026, 10, 21, 17, 50), ct(2026, 10, 21, 18, 27))
    assert (c.starts_after, c.starts_before) == (ct(2026, 10, 21, 18, 25), ct(2026, 10, 21, 18, 35))
    settle = next(s for s in slots if s.name == "settle")
    assert settle.finalized and settle.opens == ct(2026, 10, 22, 10)
    assert (settle.starts_after, settle.starts_before) == (ct(2026, 10, 21, 0), ct(2026, 10, 22, 0))


def test_no_games_no_slots():
    assert schedule.slots_for(D, []) == []


def test_props_slot_skipped_when_every_game_tips_before_three():
    assert "props" not in names(schedule.slots_for(D, [ct(2026, 10, 21, 11), ct(2026, 10, 21, 14, 30)]))


def test_oddsapi_closes_capped_at_three_largest_groups():
    closes = [s for s in schedule.slots_for(D, CAPPED_TIPS) if s.provider == "oddsapi" and s.name.startswith("close")]
    assert {s.name: s.capped for s in closes} == {
        "close@1200": True, "close@1500": False, "close@1800": False, "close@2100": False}


def test_due_respects_window_done_and_failure_cap():
    slots = schedule.slots_for(D, TIPS)
    now = ct(2026, 10, 21, 17, 45)
    assert names(schedule.due(slots, [], now)) == ["close@1800", "close@1800"]   # sgo + oddsapi
    log = [{"provider": "sgo", "slate_date": "2026-10-21", "slot": "close@1800", "outcome": "ok"}]
    assert [s.key for s in schedule.due(slots, log, now)] == [("oddsapi", "2026-10-21", "close@1800")]
    log = [{"provider": "oddsapi", "slate_date": "2026-10-21", "slot": "close@1800", "outcome": o}
           for o in ("error", "empty", "skipped_quota")]
    assert [s.provider for s in schedule.due(slots, log, now)] == ["sgo"]


def test_missed_reports_closed_windows_once():
    slots = schedule.slots_for(D, TIPS)
    now = ct(2026, 10, 21, 13)
    assert sorted(s.key for s in schedule.missed(slots, [], now)) == [
        ("oddsapi", "2026-10-21", "open"), ("sgo", "2026-10-21", "open")]
    log = [{"provider": p, "slate_date": "2026-10-21", "slot": "open", "outcome": "missed"} for p in ("sgo", "oddsapi")]
    assert schedule.missed(slots, log, now) == []


def test_capped_slot_never_due_but_reported_missed():
    slots = schedule.slots_for(D, CAPPED_TIPS)
    capped = next(s for s in slots if s.capped)
    assert capped not in schedule.due(slots, [], ct(2026, 10, 21, 11, 40))
    assert capped in schedule.missed(slots, [], ct(2026, 10, 21, 12))


def test_dst_end_day_uses_local_wall_clock():
    d = date(2026, 11, 1)   # US DST ends 02:00 this morning
    tips = schedule.slate_tips([{"commence_time": "2026-11-02T01:00:00Z"}], d)   # 19:00 CST
    assert tips == [ct(2026, 11, 1, 19)]
    slots = schedule.slots_for(d, tips)
    sgo_open = next(s for s in slots if s.key == ("sgo", "2026-11-01", "open"))
    assert sgo_open.opens.astimezone(UTC) == datetime(2026, 11, 1, 16, tzinfo=UTC)
    close = next(s for s in slots if s.key == ("sgo", "2026-11-01", "close@1900"))
    assert close.closes.astimezone(UTC) == datetime(2026, 11, 2, 0, 57, tzinfo=UTC)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: collection error `ImportError: cannot import name 'schedule'`.

- [ ] **Step 3: Implement**

`src/odds_collector/schedule.py`:

```python
"""Pure snapshot-schedule policy: which capture slots a slate has, which are due now, which were missed.

No I/O. Datetimes are timezone-aware; slate dates and windows are America/Chicago wall clock.
"""
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

CT = ZoneInfo("America/Chicago")
CLOSE_OPENS = timedelta(minutes=40)   # close window opens this long before tip...
CLOSE_ENDS = timedelta(minutes=3)     # ...and ends this long before tip
SGO_TIP_PAD = timedelta(minutes=5)    # an SGO close requests games starting within tip +/- this
GROUP_SPAN = timedelta(minutes=30)    # Odds API: tips this close to a group's first tip share one call
MAX_ODDSAPI_CLOSES = 3
MAX_FAILURES = 3
DONE = {"ok", "missed"}
FAILED = {"error", "empty", "skipped_quota"}


@dataclass(frozen=True)
class Slot:
    provider: str            # "sgo" | "oddsapi"
    slate_date: date         # Central date of the games
    name: str                # open | props | settle | close@HHMM | manual | manual-settle
    opens: datetime          # capture window is [opens, closes)
    closes: datetime
    starts_after: datetime   # request filter on game start time
    starts_before: datetime
    finalized: bool = False  # SGO settle: ask for finished games instead of unstarted ones
    capped: bool = False     # over the Odds API daily close cap: never due, reported missed

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.slate_date.isoformat(), self.name)


def ct(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(d, time(hour, minute), CT)


def slate_tips(events, d: date) -> list[datetime]:
    """One tip per game (duplicates kept so group sizes count games), sorted, for Central date d."""
    tips = (datetime.fromisoformat(e["commence_time"]) for e in events)
    return sorted(t for t in tips if t.astimezone(CT).date() == d)


def group_tips(tips: list[datetime]) -> list[list[datetime]]:
    groups: list[list[datetime]] = []
    for t in tips:
        if groups and t - groups[-1][0] <= GROUP_SPAN:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def _close_name(tip: datetime) -> str:
    return f"close@{tip.astimezone(CT):%H%M}"


def slots_for(d: date, tips: list[datetime]) -> list[Slot]:
    if not tips:
        return []
    nxt = d + timedelta(days=1)
    lo, hi = ct(d, 0), ct(nxt, 0)
    out = [Slot("sgo", d, "open", ct(d, 10), ct(d, 12), lo, hi),
           Slot("oddsapi", d, "open", ct(d, 10), ct(d, 12), lo, hi),
           Slot("sgo", d, "settle", ct(nxt, 10), ct(nxt, 12), lo, hi, finalized=True)]
    if tips[-1] > ct(d, 15):
        out.append(Slot("sgo", d, "props", ct(d, 15), ct(d, 17), lo, hi))
    for t in sorted(set(tips)):
        out.append(Slot("sgo", d, _close_name(t), t - CLOSE_OPENS, t - CLOSE_ENDS, t - SGO_TIP_PAD, t + SGO_TIP_PAD))
    groups = group_tips(tips)
    keep = sorted(groups, key=len, reverse=True)[:MAX_ODDSAPI_CLOSES]   # stable: ties keep the earlier group
    for g in groups:
        out.append(Slot("oddsapi", d, _close_name(g[0]), g[0] - CLOSE_OPENS, g[0] - CLOSE_ENDS, lo, hi,
                        capped=g not in keep))
    return out


def _status(log: list[dict]) -> tuple[set, Counter]:
    done, fails = set(), Counter()
    for r in log:
        k = (r.get("provider"), r.get("slate_date"), r.get("slot"))
        if r.get("outcome") in DONE:
            done.add(k)
        elif r.get("outcome") in FAILED:
            fails[k] += 1
    return done, fails


def due(slots: list[Slot], log: list[dict], now: datetime) -> list[Slot]:
    done, fails = _status(log)
    return [s for s in slots if not s.capped and s.opens <= now < s.closes
            and s.key not in done and fails[s.key] < MAX_FAILURES]


def missed(slots: list[Slot], log: list[dict], now: datetime) -> list[Slot]:
    done, _ = _status(log)
    return [s for s in slots if now >= s.closes and s.key not in done]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: `16 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/schedule.py tests/test_odds_collector.py
git commit -m "feat(odds): pure snapshot schedule — slots, due, missed, DST-safe"
```

---

### Task 3: Provider HTTP boundary

**Files:**
- Create: `src/odds_collector/providers.py`
- Modify: `tests/test_odds_collector.py` (import becomes `from src.odds_collector import providers, schedule, store`; add `import io`, `import urllib.error`; append tests)

**Interfaces:**
- Produces: `providers.Response` (NamedTuple `status: int | None, headers: dict[str, str]` (lower-case keys), `body: object` (None on failure), `error: str` (redacted)), `providers.sgo_events(params: dict, paginate: bool = True) -> Response` (success body = `list` of SGO page dicts), `providers.oddsapi_events() -> Response` (body = list of events), `providers.oddsapi_odds(params: dict) -> Response` (body = list of events with bookmakers). Missing key → `RuntimeError("<NAME> is not set")`.

- [ ] **Step 1: Write the failing tests**

Update imports at the top of the test file:

```python
import io
import urllib.error
```

```python
from src.odds_collector import providers, schedule, store
```

Append:

```python
# --- providers -------------------------------------------------------------

class FakeResp:
    def __init__(self, body, status=200, headers=None):
        self.status, self.headers, self._body = status, headers or {}, json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_sgo_sends_key_in_header_and_paginates(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    seen = []
    pages = iter([FakeResp({"data": [{"eventID": "a"}], "nextCursor": "c2"}),
                  FakeResp({"data": [{"eventID": "b"}], "nextCursor": None})])

    def fake(req, timeout):
        seen.append(req)
        return next(pages)
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.sgo_events({"leagueID": "NBA"})
    assert r.status == 200 and [p["data"][0]["eventID"] for p in r.body] == ["a", "b"]
    assert all("SGOKEY123" not in q.full_url for q in seen)
    assert seen[0].get_header("X-api-key") == "SGOKEY123"
    assert "cursor=c2" in seen[1].full_url


def test_sgo_without_pagination_stops_after_first_page(monkeypatch):
    monkeypatch.setenv("SGO_API_KEY", "SGOKEY123")
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp({"data": [{"eventID": "a"}], "nextCursor": "c2"}))
    assert len(providers.sgo_events({}, paginate=False).body) == 1


def test_http_error_text_is_redacted(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "ODDSKEY999")

    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {"X-Requests-Used": "5"},
                                     io.BytesIO(b"invalid key ODDSKEY999"))
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.oddsapi_odds({"regions": "us"})
    assert r.status == 401 and r.body is None
    assert "ODDSKEY999" not in r.error and "***" in r.error
    assert r.headers["x-requests-used"] == "5"


def test_network_error_is_captured_not_raised(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "NETKEY")

    def fake(req, timeout):
        raise TimeoutError("timed out")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    r = providers.oddsapi_events()
    assert r.status is None and r.body is None and "TimeoutError" in r.error


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SGO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SGO_API_KEY"):
        providers.sgo_events({})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: collection error `ImportError: cannot import name 'providers'`.

- [ ] **Step 3: Implement**

`src/odds_collector/providers.py`:

```python
"""HTTP boundary for the odds providers. Keys come only from env and never leave this module unredacted."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

SGO_URL = "https://api.sportsgameodds.com/v2/events"
ODDSAPI_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba"
TIMEOUT = 20
USER_AGENT = {"User-Agent": "nba-odds-collector/1.0"}   # default Python-urllib UA is often bot-blocked


class Response(NamedTuple):
    status: int | None
    headers: dict
    body: object
    error: str


def api_key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***").replace(urllib.parse.quote(secret), "***")


def _lower(headers) -> dict:
    return {k.lower(): v for k, v in (headers or {}).items()}


def _get(url: str, headers: dict, secret: str) -> Response:
    try:
        req = urllib.request.Request(url, headers={**USER_AGENT, **headers})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return Response(r.status, _lower(r.headers), json.loads(r.read()), "")
    except urllib.error.HTTPError as e:   # before OSError: HTTPError is a URLError is an OSError
        body = e.read()[:300].decode("utf-8", "replace")
        return Response(e.code, _lower(e.headers), None, _redact(f"HTTP {e.code}: {body}", secret))
    except (OSError, ValueError) as e:    # URLError, timeouts, malformed JSON
        return Response(None, {}, None, _redact(f"{type(e).__name__}: {e}", secret))


def sgo_events(params: dict, paginate: bool = True) -> Response:
    """GET /v2/events, following nextCursor. Success body is the list of page bodies."""
    key = api_key("SGO_API_KEY")
    pages, cursor = [], None
    while True:
        query = {**params, **({"cursor": cursor} if cursor else {})}
        r = _get(f"{SGO_URL}?{urllib.parse.urlencode(query)}", {"x-api-key": key}, key)
        if r.body is None:
            return r
        pages.append(r.body)
        cursor = r.body.get("nextCursor")
        if not (paginate and cursor):
            return r._replace(body=pages)


def oddsapi_events() -> Response:
    """Quota-free schedule call; its headers still carry x-requests-used."""
    key = api_key("THE_ODDS_API_KEY")
    return _get(f"{ODDSAPI_URL}/events?{urllib.parse.urlencode({'apiKey': key})}", {}, key)


def oddsapi_odds(params: dict) -> Response:
    key = api_key("THE_ODDS_API_KEY")
    return _get(f"{ODDSAPI_URL}/odds?{urllib.parse.urlencode({**params, 'apiKey': key})}", {}, key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: `21 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/odds_collector/providers.py tests/test_odds_collector.py
git commit -m "feat(odds): provider HTTP boundary — header auth for SGO, key redaction, pagination"
```

---

### Task 4: Flatteners and fixtures

**Files:**
- Create: `src/odds_collector/flatten.py`
- Create: `tests/fixtures/odds_collector/sgo_events.json`
- Create: `tests/fixtures/odds_collector/oddsapi_odds.json`
- Create: `tests/fixtures/odds_collector/oddsapi_events.json`
- Modify: `tests/test_odds_collector.py` (import becomes `from src.odds_collector import flatten, providers, schedule, store`; append tests)

**Interfaces:**
- Consumes: `store.write_parquet` (Task 1).
- Produces: `flatten.SGO_SCHEMA`, `flatten.ODDSAPI_SCHEMA` (`pa.Schema`), `flatten.parse_odd_id(odd_id: str) -> tuple[str|None, ...]` (5-tuple), `flatten.flatten(envelope: dict) -> tuple[list[dict], pa.Schema]`. Envelope keys used: `provider, request_id, slot, captured_at_utc, payload` (SGO payload = list of page dicts; Odds API payload = list of events).

- [ ] **Step 1: Write the fixtures**

`tests/fixtures/odds_collector/sgo_events.json` (synthetic; field names per SGO docs, verified at smoke test):

```json
{
  "success": true,
  "nextCursor": null,
  "data": [
    {
      "eventID": "EVT1",
      "leagueID": "NBA",
      "teams": {
        "home": {"teamID": "LOS_ANGELES_LAKERS_NBA", "names": {"long": "Los Angeles Lakers"}},
        "away": {"teamID": "BOSTON_CELTICS_NBA", "names": {"long": "Boston Celtics"}}
      },
      "status": {"startsAt": "2026-10-22T02:00:00.000Z"},
      "odds": {
        "points-home-game-sp-home": {"byBookmaker": {
          "fanduel": {"odds": "-110", "spread": "-3.5", "available": true, "lastUpdatedAt": "2026-10-21T15:00:00.000Z",
                      "openOdds": "-108", "openSpread": "-2.5", "closeOdds": "-112", "closeSpread": "-4"},
          "draftkings": {"odds": "+100", "spread": "-3", "available": true, "lastUpdatedAt": "2026-10-21T14:55:00.000Z"}
        }},
        "points-all-1h-ou-over": {"byBookmaker": {
          "betmgm": {"odds": "-115", "overUnder": "112.5", "available": true, "lastUpdatedAt": "2026-10-21T14:00:00.000Z"}
        }},
        "points+rebounds-LEBRON_JAMES_1_NBA-game-ou-under": {"byBookmaker": {
          "caesars": {"odds": "-120", "overUnder": "33.5", "available": false, "lastUpdatedAt": "2026-10-21T13:00:00.000Z"}
        }}
      }
    },
    {"eventID": "EVT2"},
    {
      "eventID": "EVT3",
      "teams": null,
      "status": {"startsAt": "not-a-date"},
      "odds": {"weird": {"byBookmaker": {"fanduel": {"odds": "N/A"}}}}
    }
  ]
}
```

`tests/fixtures/odds_collector/oddsapi_odds.json`:

```json
[
  {
    "id": "OA1", "sport_key": "basketball_nba", "commence_time": "2026-10-22T02:00:00Z",
    "home_team": "Los Angeles Lakers", "away_team": "Boston Celtics",
    "bookmakers": [
      {"key": "betrivers", "title": "BetRivers", "last_update": "2026-10-21T14:59:00Z", "markets": [
        {"key": "h2h", "last_update": "2026-10-21T14:59:00Z", "outcomes": [
          {"name": "Los Angeles Lakers", "price": -150}, {"name": "Boston Celtics", "price": 130}]},
        {"key": "spreads", "last_update": "2026-10-21T14:58:00Z", "outcomes": [
          {"name": "Los Angeles Lakers", "price": -110, "point": -3.5},
          {"name": "Boston Celtics", "price": -110, "point": 3.5}]}
      ]},
      {"key": "fanatics", "title": "Fanatics", "last_update": "2026-10-21T14:57:00Z", "markets": [
        {"key": "totals", "last_update": "2026-10-21T14:57:00Z", "outcomes": [
          {"name": "Over", "price": -112, "point": 224.5}, {"name": "Under", "price": -108, "point": 224.5}]}
      ]}
    ]
  }
]
```

`tests/fixtures/odds_collector/oddsapi_events.json` (OA2 tips 18:30 CT, OA1 21:00 CT on 2026-10-21):

```json
[
  {"id": "OA1", "sport_key": "basketball_nba", "commence_time": "2026-10-22T02:00:00Z",
   "home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"},
  {"id": "OA2", "sport_key": "basketball_nba", "commence_time": "2026-10-21T23:30:00Z",
   "home_team": "New York Knicks", "away_team": "Cleveland Cavaliers"}
]
```

- [ ] **Step 2: Write the failing tests**

Update the import line to `from src.odds_collector import flatten, providers, schedule, store`, then append:

```python
# --- flatten ---------------------------------------------------------------

def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def sgo_env():
    return {"request_id": "rid1", "provider": "sgo", "slot": "open", "captured_at_utc": "2026-10-21T15:00:00Z",
            "payload": [load_fixture("sgo_events.json")]}


def test_parse_odd_id_handles_players_and_combo_stats():
    assert flatten.parse_odd_id("points+rebounds-LEBRON_JAMES_1_NBA-game-ou-under") == (
        "points+rebounds", "LEBRON_JAMES_1_NBA", "game", "ou", "under")
    assert flatten.parse_odd_id("points-all-1h-ou-over") == ("points", "all", "1h", "ou", "over")
    assert flatten.parse_odd_id("garbage") == (None,) * 5


def test_sgo_rows():
    rows, schema = flatten.flatten(sgo_env())
    assert schema is flatten.SGO_SCHEMA and len(rows) == 5   # EVT1: 4, EVT2 (no odds): 0, EVT3 (junk): 1
    fd = next(r for r in rows if r["bookmaker_id"] == "fanduel" and r["event_id"] == "EVT1")
    assert fd == {
        "request_id": "rid1", "slot": "open", "captured_at_utc": "2026-10-21T15:00:00Z", "minutes_to_tip": 660.0,
        "event_id": "EVT1", "starts_at_utc": "2026-10-22T02:00:00.000Z",
        "home_team": "Los Angeles Lakers", "away_team": "Boston Celtics",
        "odd_id": "points-home-game-sp-home", "stat_id": "points", "stat_entity_id": "home", "period_id": "game",
        "bet_type_id": "sp", "side_id": "home", "bookmaker_id": "fanduel",
        "odds_american": -110.0, "line": -3.5, "available": True, "last_updated_at": "2026-10-21T15:00:00.000Z",
        "open_odds": -108.0, "open_line": -2.5, "close_odds": -112.0, "close_line": -4.0}
    dk = next(r for r in rows if r["bookmaker_id"] == "draftkings")
    assert dk["odds_american"] == 100.0 and dk["open_odds"] is None and dk["close_line"] is None
    half = next(r for r in rows if r["bookmaker_id"] == "betmgm")
    assert (half["period_id"], half["stat_entity_id"], half["line"]) == ("1h", "all", 112.5)
    prop = next(r for r in rows if r["bookmaker_id"] == "caesars")
    assert (prop["stat_id"], prop["stat_entity_id"], prop["line"], prop["available"]) == (
        "points+rebounds", "LEBRON_JAMES_1_NBA", 33.5, False)
    junk = next(r for r in rows if r["event_id"] == "EVT3")
    assert junk["stat_id"] is None and junk["odds_american"] is None
    assert junk["minutes_to_tip"] is None and junk["home_team"] is None


def test_oddsapi_rows_preserve_signs_and_books():
    env = {"request_id": "rid2", "provider": "oddsapi", "slot": "close@1830",
           "captured_at_utc": "2026-10-22T01:40:00Z", "payload": load_fixture("oddsapi_odds.json")}
    rows, schema = flatten.flatten(env)
    assert schema is flatten.ODDSAPI_SCHEMA and len(rows) == 6
    spreads = {r["outcome_name"]: (r["point"], r["price_american"]) for r in rows if r["market_key"] == "spreads"}
    assert spreads == {"Los Angeles Lakers": (-3.5, -110.0), "Boston Celtics": (3.5, -110.0)}
    assert {r["bookmaker_key"] for r in rows} == {"betrivers", "fanatics"}
    assert rows[0]["minutes_to_tip"] == 20.0 and rows[0]["point"] is None   # h2h has no point


def test_flattened_rows_write_to_parquet(tmp_path):
    rows, schema = flatten.flatten(sgo_env())
    store.write_parquet(tmp_path / "x.parquet", rows, schema)
    table = pq.read_table(tmp_path / "x.parquet")
    assert table.num_rows == 5 and table.schema.equals(flatten.SGO_SCHEMA)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: collection error `ImportError: cannot import name 'flatten'`.

- [ ] **Step 4: Implement**

`src/odds_collector/flatten.py`:

```python
"""Pure provider payload -> tidy rows (one per snapshot x event x market x side x book).

Field names follow the provider docs as of 2026-10-08. If the live smoke capture shows different names,
fix them here and run `python -m src.odds_collector --rebuild-parquet`; the raw files are untouched.
"""
from datetime import datetime

import pyarrow as pa

S, F, B = pa.string(), pa.float64(), pa.bool_()
_META = [("request_id", S), ("slot", S), ("captured_at_utc", S), ("minutes_to_tip", F), ("event_id", S)]
SGO_SCHEMA = pa.schema(_META + [
    ("starts_at_utc", S), ("home_team", S), ("away_team", S), ("odd_id", S), ("stat_id", S),
    ("stat_entity_id", S), ("period_id", S), ("bet_type_id", S), ("side_id", S), ("bookmaker_id", S),
    ("odds_american", F), ("line", F), ("available", B), ("last_updated_at", S),
    ("open_odds", F), ("open_line", F), ("close_odds", F), ("close_line", F)])
ODDSAPI_SCHEMA = pa.schema(_META + [
    ("commence_time_utc", S), ("home_team", S), ("away_team", S), ("bookmaker_key", S),
    ("bookmaker_last_update", S), ("market_key", S), ("market_last_update", S),
    ("outcome_name", S), ("point", F), ("price_american", F)])


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _minutes_to_tip(start, captured):
    try:
        return (datetime.fromisoformat(start) - datetime.fromisoformat(captured)).total_seconds() / 60
    except (TypeError, ValueError):
        return None


def _first(d: dict, *keys):
    return next((d[k] for k in keys if d.get(k) is not None), None)


def parse_odd_id(odd_id):
    """statID-statEntityID-periodID-betTypeID-sideID; entity IDs (players) contain no '-'."""
    try:
        head, period, bet, side = odd_id.rsplit("-", 3)
    except (AttributeError, ValueError):
        return (None,) * 5
    stat, _, entity = head.partition("-")
    return stat, entity or None, period, bet, side


def _sgo_team(teams, side):
    return (((teams or {}).get(side) or {}).get("names") or {}).get("long")


def sgo_rows(env: dict) -> list[dict]:
    meta = {k: env[k] for k in ("request_id", "slot", "captured_at_utc")}
    rows = []
    for page in env["payload"]:
        for ev in page.get("data") or []:
            start = (ev.get("status") or {}).get("startsAt")
            base = {**meta, "event_id": ev.get("eventID"), "starts_at_utc": start,
                    "minutes_to_tip": _minutes_to_tip(start, env["captured_at_utc"]),
                    "home_team": _sgo_team(ev.get("teams"), "home"),
                    "away_team": _sgo_team(ev.get("teams"), "away")}
            for odd_id, odd in (ev.get("odds") or {}).items():
                stat, entity, period, bet, side = parse_odd_id(odd_id)
                for book, b in ((odd or {}).get("byBookmaker") or {}).items():
                    b = b or {}
                    rows.append({**base, "odd_id": odd_id, "stat_id": stat, "stat_entity_id": entity,
                                 "period_id": period, "bet_type_id": bet, "side_id": side, "bookmaker_id": book,
                                 "odds_american": _num(b.get("odds")),
                                 "line": _num(_first(b, "spread", "overUnder")),
                                 "available": b["available"] if isinstance(b.get("available"), bool) else None,
                                 "last_updated_at": b.get("lastUpdatedAt"),
                                 "open_odds": _num(b.get("openOdds")),
                                 "open_line": _num(_first(b, "openSpread", "openOverUnder")),
                                 "close_odds": _num(b.get("closeOdds")),
                                 "close_line": _num(_first(b, "closeSpread", "closeOverUnder"))})
    return rows


def oddsapi_rows(env: dict) -> list[dict]:
    meta = {k: env[k] for k in ("request_id", "slot", "captured_at_utc")}
    rows = []
    for ev in env["payload"] or []:
        start = ev.get("commence_time")
        base = {**meta, "event_id": ev.get("id"), "commence_time_utc": start,
                "minutes_to_tip": _minutes_to_tip(start, env["captured_at_utc"]),
                "home_team": ev.get("home_team"), "away_team": ev.get("away_team")}
        for bk in ev.get("bookmakers") or []:
            for m in bk.get("markets") or []:
                for o in m.get("outcomes") or []:
                    rows.append({**base, "bookmaker_key": bk.get("key"), "bookmaker_last_update": bk.get("last_update"),
                                 "market_key": m.get("key"), "market_last_update": m.get("last_update"),
                                 "outcome_name": o.get("name"), "point": _num(o.get("point")),
                                 "price_american": _num(o.get("price"))})
    return rows


def flatten(env: dict) -> tuple[list[dict], pa.Schema]:
    if env["provider"] == "sgo":
        return sgo_rows(env), SGO_SCHEMA
    return oddsapi_rows(env), ODDSAPI_SCHEMA
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: `25 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/odds_collector/flatten.py tests/fixtures/odds_collector tests/test_odds_collector.py
git commit -m "feat(odds): flatten SGO and Odds API payloads to typed parquet rows"
```

---

### Task 5: CLI — run-due, dry-run, manual capture, rebuild

**Files:**
- Create: `src/odds_collector/__main__.py`
- Modify: `tests/test_odds_collector.py` (import becomes `from src.odds_collector import __main__ as cli, flatten, providers, schedule, store`; append tests)

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `cli.iso(dt) -> str`, `cli.load_schedule(root, d) -> {"fetched_at_utc": str|None, "events": {id: event}}`, `cli.refresh_schedule(root, d, now) -> (schedule dict, rc int)`, `cli.run_due(root, now, dry_run=False) -> int`, `cli.capture(root, slot, now, log, paginate=True, limit=50) -> int`, `cli.manual_slot(kind, now) -> Slot`, `cli.rebuild_parquet(root) -> int`, `cli.main(argv=None) -> int`. Return code 1 only for HTTP 401/403.

- [ ] **Step 1: Write the failing tests**

Update the import line to `from src.odds_collector import __main__ as cli, flatten, providers, schedule, store`, then append:

```python
# --- CLI -------------------------------------------------------------------

NOW = datetime(2026, 10, 21, 23, 15, tzinfo=UTC)   # 18:15 CT: OA2 tips 18:30, OA1 tips 21:00


def seed_schedule(root, fetched=NOW):
    events = load_fixture("oddsapi_events.json")
    store.atomic_write(root / "schedule" / "2026-10-21.json",
                       json.dumps({"fetched_at_utc": cli.iso(fetched), "events": {e["id"]: e for e in events}}).encode())


def fake_providers(monkeypatch, status=200):
    page, odds, calls = load_fixture("sgo_events.json"), load_fixture("oddsapi_odds.json"), []
    ok = status == 200

    def sgo(params, paginate=True):
        calls.append(("sgo", {**params, "_paginate": paginate}))
        return providers.Response(status, {}, [page] if ok else None, "" if ok else f"HTTP {status}")

    def oddsapi(params):
        calls.append(("oddsapi", params))
        return providers.Response(status, {"x-requests-used": "12", "x-requests-remaining": "488"},
                                  odds if ok else None, "" if ok else f"HTTP {status}")
    monkeypatch.setattr(providers, "sgo_events", sgo)
    monkeypatch.setattr(providers, "oddsapi_odds", oddsapi)
    return calls


def test_run_due_captures_logs_and_is_idempotent(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    calls = fake_providers(monkeypatch)
    assert cli.run_due(tmp_path, NOW) == 0
    log = store.read_log(tmp_path)
    assert sorted((r["provider"], r["slot"], r["outcome"]) for r in log) == [
        ("oddsapi", "close@1830", "ok"), ("oddsapi", "open", "missed"),
        ("sgo", "close@1830", "ok"), ("sgo", "open", "missed"), ("sgo", "props", "missed")]
    sgo_row = next(r for r in log if r["provider"] == "sgo" and r["outcome"] == "ok")
    assert sgo_row["objects"] == "3" and sgo_row["error"] == ""
    raw = tmp_path / sgo_row["payload_path"]
    env = store.read_raw(raw)
    assert env["request"]["params"]["startsAfter"] == "2026-10-21T23:25:00Z"
    assert env["request"]["params"]["started"] == "false"
    assert pq.read_table(raw.with_name(raw.name.replace(".json.gz", ".parquet"))).num_rows == 5
    oddsapi_params = next(p for who, p in calls if who == "oddsapi")
    assert oddsapi_params["commenceTimeFrom"] == "2026-10-21T23:15:00Z"   # never in-play games
    n = len(calls)
    assert cli.run_due(tmp_path, NOW + timedelta(minutes=5)) == 0
    assert len(calls) == n and len(store.read_log(tmp_path)) == 5


def test_dry_run_reports_without_network_or_writes(tmp_path, capsys):
    seed_schedule(tmp_path, fetched=NOW - timedelta(hours=5))   # stale: a real run would refetch
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    assert cli.run_due(tmp_path, NOW, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "due sgo 2026-10-21 close@1830" in out and "missed sgo 2026-10-21 open" in out and "stale" in out
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == before


def test_auth_failure_exits_nonzero_and_logs_error(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch, status=401)
    assert cli.run_due(tmp_path, NOW) == 1
    errors = [r for r in store.read_log(tmp_path) if r["outcome"] == "error"]
    assert {r["provider"] for r in errors} == {"sgo", "oddsapi"} and all(r["http_status"] == "401" for r in errors)
    assert not list(tmp_path.glob("*/date=*/*"))


def test_flatten_failure_keeps_raw_and_logs_ok(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch)

    def boom(env):
        raise KeyError("payload")
    monkeypatch.setattr(flatten, "flatten", boom)
    cli.run_due(tmp_path, NOW)
    ok = [r for r in store.read_log(tmp_path) if r["outcome"] == "ok"]
    assert len(ok) == 2 and all(r["error"].startswith("flatten: KeyError") for r in ok)
    assert all((tmp_path / r["payload_path"]).exists() for r in ok)


def test_quota_guards_skip_requests(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    calls = fake_providers(monkeypatch)
    store.append_log(tmp_path, {"provider": "sgo", "attempted_at_utc": "2026-10-01T15:00:00Z", "objects": 2300,
                                "slot": "old", "outcome": "ok"})
    store.append_log(tmp_path, {"provider": "oddsapi", "attempted_at_utc": "2026-10-21T20:00:00Z", "quota_used": 448,
                                "slot": "schedule", "outcome": "ok"})
    cli.run_due(tmp_path, NOW)
    assert calls == []
    assert sorted(r["provider"] for r in store.read_log(tmp_path) if r["outcome"] == "skipped_quota") == [
        "oddsapi", "sgo"]


def test_schedule_refresh_merges_by_event_id(tmp_path, monkeypatch):
    seed_schedule(tmp_path, fetched=NOW - timedelta(hours=3))
    moved = {"id": "OA2", "commence_time": "2026-10-22T00:00:00Z",
             "home_team": "New York Knicks", "away_team": "Cleveland Cavaliers"}
    monkeypatch.setattr(providers, "oddsapi_events",
                        lambda: providers.Response(200, {"x-requests-used": "20"}, [moved], ""))
    sched, rc = cli.refresh_schedule(tmp_path, date(2026, 10, 21), NOW)
    assert rc == 0 and set(sched["events"]) == {"OA1", "OA2"}            # OA1 dropped by provider, kept
    assert sched["events"]["OA2"]["commence_time"] == "2026-10-22T00:00:00Z"   # tip change picked up
    assert cli.load_schedule(tmp_path, date(2026, 10, 21)) == sched
    assert store.read_log(tmp_path)[-1]["quota_used"] == "20"


def test_manual_capture_caps_sgo_objects(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    calls = fake_providers(monkeypatch)
    assert cli.main(["--capture", "sgo"]) == 0
    assert calls[0][1]["limit"] == "5" and calls[0][1]["_paginate"] is False
    assert store.read_log(tmp_path / "raw" / "odds")[0]["slot"] == "manual"


def test_rebuild_parquet_matches_original(tmp_path, monkeypatch):
    seed_schedule(tmp_path)
    fake_providers(monkeypatch)
    cli.run_due(tmp_path, NOW)
    paths = sorted(tmp_path.glob("*/date=*/*.parquet"))
    before = [pq.read_table(p) for p in paths]
    for p in paths:
        p.unlink()
    assert cli.rebuild_parquet(tmp_path) == 0
    assert len(paths) == 2 and all(pq.read_table(p).equals(t) for p, t in zip(paths, before))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: collection error `ImportError: cannot import name '__main__'`.

- [ ] **Step 3: Implement**

`src/odds_collector/__main__.py`:

```python
"""Odds collector CLI.

python -m src.odds_collector --dry-run          report missed/due slots from cached files (no network, no writes)
python -m src.odds_collector --run-due          capture every due slot (the scheduled entry point)
python -m src.odds_collector --capture sgo      one manual smoke capture: sgo | sgo-settle | oddsapi
python -m src.odds_collector --rebuild-parquet  regenerate every .parquet from its raw .json.gz
"""
import argparse
import json
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import flatten, providers, schedule, store

UTC = timezone.utc
SGO_OBJECT_STOP = 2300
ODDSAPI_CREDIT_STOP = 450
ODDSAPI_ODDS_COST = 3   # h2h,spreads,totals x region us
SCHEDULE_MAX_AGE = timedelta(hours=2)
EVENTS_ENDPOINT = "/v4/sports/basketball_nba/events"


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _schedule_path(root: Path, d: date) -> Path:
    return root / "schedule" / f"{d.isoformat()}.json"


def load_schedule(root: Path, d: date) -> dict:
    p = _schedule_path(root, d)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"fetched_at_utc": None, "events": {}}


def _quota(r: providers.Response) -> dict:
    return {"quota_used": r.headers.get("x-requests-used", ""),
            "quota_remaining": r.headers.get("x-requests-remaining", "")}


def _rc(r: providers.Response) -> int:
    """Auth failures fail the workflow run so GitHub emails the owner; everything else retries next tick."""
    return 1 if r.status in (401, 403) else 0


def _row(slot: schedule.Slot, now: datetime, **kw) -> dict:
    return {"provider": slot.provider, "slate_date": slot.slate_date.isoformat(), "slot": slot.name,
            "attempted_at_utc": iso(now), **kw}


def refresh_schedule(root: Path, d: date, now: datetime) -> tuple[dict, int]:
    """Fetch the quota-free /events list, merge it by event id into the day's cache, and log the call."""
    sched = load_schedule(root, d)
    r = providers.oddsapi_events()
    ok = r.status == 200 and isinstance(r.body, list)
    if ok:
        # Merge, don't replace: started games drop out of /events but their slots must stay evaluable.
        sched = {"fetched_at_utc": iso(now), "events": {**sched["events"], **{e["id"]: e for e in r.body}}}
        store.atomic_write(_schedule_path(root, d), json.dumps(sched, indent=1, sort_keys=True).encode())
    store.append_log(root, {"request_id": uuid.uuid4().hex, "provider": "oddsapi", "slate_date": d.isoformat(),
                            "slot": "schedule", "attempted_at_utc": iso(now), "endpoint": EVENTS_ENDPOINT,
                            "http_status": r.status, "outcome": "ok" if ok else "error", "error": r.error,
                            "objects": len(r.body) if ok else 0, **_quota(r)})
    return sched, _rc(r)


def capture(root: Path, slot: schedule.Slot, now: datetime, log: list[dict],
            paginate: bool = True, limit: int = 50) -> int:
    rid = uuid.uuid4().hex
    row = _row(slot, now, request_id=rid)
    after = slot.starts_after if slot.finalized else max(slot.starts_after, now)   # never in-play games
    if slot.provider == "sgo":
        if store.sgo_objects_this_month(log, now) >= SGO_OBJECT_STOP:
            store.append_log(root, {**row, "outcome": "skipped_quota"})
            return 0
        endpoint = "/v2/events"
        params = {"leagueID": "NBA", "includeOpenCloseOdds": "true", "limit": str(limit),
                  "startsAfter": iso(after), "startsBefore": iso(slot.starts_before),
                  **({"finalized": "true"} if slot.finalized else {"started": "false"})}
        r = providers.sgo_events(params, paginate)
        objects = sum(len(page.get("data") or []) for page in r.body) if r.status == 200 else 0
    else:
        if store.oddsapi_used(log) + ODDSAPI_ODDS_COST > ODDSAPI_CREDIT_STOP:
            store.append_log(root, {**row, "outcome": "skipped_quota"})
            return 0
        endpoint = "/v4/sports/basketball_nba/odds"
        params = {"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american",
                  "commenceTimeFrom": iso(after), "commenceTimeTo": iso(slot.starts_before)}
        r = providers.oddsapi_odds(params)
        objects = len(r.body) if r.status == 200 else 0
    row.update(endpoint=endpoint, http_status=r.status, error=r.error, objects=objects, **_quota(r))
    if r.status != 200 or r.body is None:
        store.append_log(root, {**row, "outcome": "error"})
        return _rc(r)
    if objects == 0:   # SGO bills every response as at least one object
        store.append_log(root, {**row, "outcome": "empty", "objects": 1 if slot.provider == "sgo" else 0})
        return 0
    stem = (f"{slot.provider}/date={slot.slate_date.isoformat()}/"
            f"{now.astimezone(UTC):%Y-%m-%dT%H-%M-%SZ}_{slot.name}_{rid}")
    env = {"request_id": rid, "provider": slot.provider, "slot": slot.name,
           "slate_date": slot.slate_date.isoformat(), "captured_at_utc": iso(now),
           "request": {"endpoint": endpoint, "params": params}, "http_status": r.status,
           "headers": r.headers, "payload": r.body}
    store.write_raw(root / f"{stem}.json.gz", env)
    try:
        rows, schema = flatten.flatten(env)
        store.write_parquet(root / f"{stem}.parquet", rows, schema)
    except Exception as e:   # raw is already safe; --rebuild-parquet can redo this after a fix
        row["error"] = f"flatten: {type(e).__name__}: {e}"
    store.append_log(root, {**row, "outcome": "ok", "payload_path": f"{stem}.json.gz"})
    return 0


def run_due(root: Path, now: datetime, dry_run: bool = False) -> int:
    today = now.astimezone(schedule.CT).date()
    yday = today - timedelta(days=1)
    sched, rc = load_schedule(root, today), 0
    fetched = sched["fetched_at_utc"]
    if fetched is None or now - datetime.fromisoformat(fetched) > SCHEDULE_MAX_AGE:
        if dry_run:
            print(f"schedule stale (fetched {fetched}); a real run would refresh /events")
        else:
            sched, rc = refresh_schedule(root, today, now)
    slots = (schedule.slots_for(yday, schedule.slate_tips(load_schedule(root, yday)["events"].values(), yday))
             + schedule.slots_for(today, schedule.slate_tips(sched["events"].values(), today)))
    log = store.read_log(root)
    for s in schedule.missed(slots, log, now):
        if dry_run:
            print("missed", *s.key)
        else:
            store.append_log(root, _row(s, now, request_id=uuid.uuid4().hex, outcome="missed"))
    for s in schedule.due(slots, log, now):
        if dry_run:
            print("due", *s.key)
        else:
            rc |= capture(root, s, now, store.read_log(root))
    return rc


def manual_slot(kind: str, now: datetime) -> schedule.Slot:
    today = now.astimezone(schedule.CT).date()
    if kind == "sgo-settle":
        yday = today - timedelta(days=1)
        return schedule.Slot("sgo", yday, "manual-settle", now, now, schedule.ct(yday, 0), schedule.ct(today, 0),
                             finalized=True)
    return schedule.Slot(kind, today, "manual", now, now, now, now + timedelta(days=7))


def rebuild_parquet(root: Path) -> int:
    raws = sorted(root.glob("*/date=*/*.json.gz"))
    for raw in raws:
        env = store.read_raw(raw)
        rows, schema = flatten.flatten(env)
        store.write_parquet(raw.with_name(raw.name.removesuffix(".json.gz") + ".parquet"), rows, schema)
    print(f"rebuilt {len(raws)} parquet files")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.odds_collector", description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="report missed/due slots; no network, no writes")
    g.add_argument("--run-due", action="store_true", help="capture every due slot (scheduled entry point)")
    g.add_argument("--capture", choices=["sgo", "sgo-settle", "oddsapi"],
                   help="one manual smoke capture; ignores windows, keeps quota guards; SGO capped at 5 events")
    g.add_argument("--rebuild-parquet", action="store_true", help="regenerate every .parquet from raw")
    a = p.parse_args(argv)
    root, now = store.data_root(), datetime.now(UTC)
    if a.rebuild_parquet:
        return rebuild_parquet(root)
    if a.capture:
        return capture(root, manual_slot(a.capture, now), now, store.read_log(root), paginate=False, limit=5)
    return run_due(root, now, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_odds_collector.py -q`
Expected: `33 passed`.

- [ ] **Step 5: Manual dry-run check (no keys, no network)**

Run: `NBA_DATA_DIR="$(mktemp -d)" $PY -m src.odds_collector --dry-run`
Expected: prints `schedule stale (fetched None); a real run would refresh /events`, exit 0, nothing written.
Run: `$PY -m src.odds_collector --help` — shows the four modes.

- [ ] **Step 6: Commit**

```bash
git add src/odds_collector/__main__.py tests/test_odds_collector.py
git commit -m "feat(odds): CLI — run-due, dry-run, manual capture, parquet rebuild"
```

---

### Task 6: Workflow, CI, and runbook

**Files:**
- Create: `.github/workflows/odds-collector.yml`
- Modify: `.github/workflows/ci.yml` (add test step)
- Create: `docs/odds-collector.md`
- Modify: `.env.example` (add key names)
- Modify: `README.md` (one pointer line)

**Interfaces:**
- Consumes: `python -m src.odds_collector --run-due | --dry-run | --capture KIND`; `store.data_root()` resolves to `<checkout>/data/raw/odds` when `NBA_DATA_DIR` is unset.

- [ ] **Step 1: Write the workflow**

`.github/workflows/odds-collector.yml`:

```yaml
name: Odds collector

# Triggered every 5 min by cron-job.org via workflow_dispatch (precise timing); the schedule below is
# only a backup. Every run is idempotent: it captures the slots that are due and nothing else.
# Setup and operations: docs/odds-collector.md
on:
  workflow_dispatch:
    inputs:
      mode:
        description: What to run
        type: choice
        default: run-due
        options: [run-due, dry-run, capture-sgo, capture-sgo-settle, capture-oddsapi]
  schedule:
    - cron: "17,47 14-23,0-4 * * *"

concurrency:
  group: odds-collector
  cancel-in-progress: false

permissions:
  contents: read

jobs:
  collect:
    # Inert until the data repo is configured (Settings -> Secrets and variables -> Actions -> Variables).
    if: vars.ODDS_DATA_REPO != ''
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - name: Check out data repo (root files + schedule/ only)
        uses: actions/checkout@v4
        with:
          repository: ${{ vars.ODDS_DATA_REPO }}
          token: ${{ secrets.ODDS_DATA_TOKEN }}
          path: data/raw/odds
          sparse-checkout: schedule
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install pyarrow
      - name: Collect
        env:
          MODE: ${{ inputs.mode || 'run-due' }}
          SGO_API_KEY: ${{ secrets.SGO_API_KEY }}
          THE_ODDS_API_KEY: ${{ secrets.THE_ODDS_API_KEY }}
        run: |
          case "$MODE" in
            capture-*) python -m src.odds_collector --capture "${MODE#capture-}" ;;
            *) python -m src.odds_collector "--$MODE" ;;
          esac
      - name: Push captures to data repo
        if: ${{ !cancelled() }}
        working-directory: data/raw/odds
        run: |
          git config user.name "odds-collector"
          git config user.email "odds-collector@users.noreply.github.com"
          git add --sparse -A
          git diff --cached --quiet && exit 0
          git commit -q -m "capture $(date -u +%Y-%m-%dT%H:%M:%SZ)"
          git push -q || { git pull -q --rebase && git push -q; }
```

- [ ] **Step 2: Add the test step to CI**

In `.github/workflows/ci.yml`, append after the "Check autoresearch paths" step:

```yaml
      - name: Odds collector tests
        run: |
          pip install pyarrow pytest
          python -m pytest tests/test_odds_collector.py -q
```

- [ ] **Step 3: Write the runbook**

`docs/odds-collector.md`:

````markdown
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
- A failed workflow run means an auth error (401/403): a key was revoked or expired — replace the secret.
- Quota stops: SGO 2,300 objects per calendar month (counted from `requests.csv`); The Odds API 450 credits.
- If SGO renames fields: fix `src/odds_collector/flatten.py`, then locally
  `python -m src.odds_collector --rebuild-parquet` and commit/push the data repo.
- To pause collection: disable the cron-job.org job and the workflow (`gh workflow disable odds-collector.yml`).
````

- [ ] **Step 4: `.env.example` and README**

Append to `.env.example`:

```
# Odds collector (normally set as GitHub Actions secrets; locally only for manual --capture runs).
SGO_API_KEY=
THE_ODDS_API_KEY=
```

Append to `README.md` (end of file):

```markdown

## Odds collection

Forward-only NBA odds snapshots (props, 1H/quarter lines, moneylines) are collected by
`src/odds_collector` on GitHub Actions — setup and operations in [docs/odds-collector.md](docs/odds-collector.md).
```

- [ ] **Step 5: Verify**

Run: `$PY -c "import yaml, sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/odds-collector.yml .github/workflows/ci.yml` → no output. (If PyYAML is not installed, skip this check; GitHub validates the YAML when the branch is pushed in Task 7.)
Run: `$PY -m compileall -q src/odds_collector` → no output.
Run: `$PY -m pytest tests/test_odds_collector.py -q` → `33 passed`.
Run: `git status --short` → only the files listed in this task.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/odds-collector.yml .github/workflows/ci.yml docs/odds-collector.md .env.example README.md
git commit -m "ci(odds): collector workflow (dispatch + backup cron), CI tests, setup runbook"
```

---

### Task 7: Review, PR, and gated rollout

**Files:** none (process).

- [ ] **Step 1: Whole-branch review** — run a fresh code review of `main...HEAD` focused on the Review Focus list, key handling, and the workflow's secret exposure. Fix confirmed findings with a test each.
- [ ] **Step 2: Push and open PR** — `git push -u origin HEAD`, then `gh pr create --base main` with a summary, the test count, and the rollout checklist (setup steps 1–7 of `docs/odds-collector.md`). End the body with the Claude Code attribution line.
- [ ] **Step 3: Hand off to the user** — they merge after review, then do setup steps 1–5 themselves.
- [ ] **Step 4: Smoke capture — ONLY after the user explicitly approves the network calls in chat** — dispatch `dry-run`, `capture-oddsapi`, `capture-sgo`, `capture-sgo-settle` (`gh workflow run ...`), pull the data repo, and report: books returned, period/prop presence, open/close presence, object/credit usage, key absent. If field names differ from the fixture, fix `flatten.py` + fixture + tests in a follow-up PR and rebuild parquet.
- [ ] **Step 5: User enables cron-job.org** (setup step 7) before 2026-10-20.
