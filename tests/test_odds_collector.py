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
