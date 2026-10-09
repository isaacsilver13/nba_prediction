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
