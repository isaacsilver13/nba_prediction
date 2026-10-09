"""Tests for src.odds_collector — synthetic fixtures only; no test may touch the network."""
import io
import json
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.odds_collector import providers, schedule, store
from src.odds_collector.schedule import CT

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
