"""HTTP boundary for the odds providers. Keys come only from env and never leave this module unredacted."""
import http.client
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
    except (OSError, ValueError, http.client.HTTPException) as e:   # URLError, timeouts, bad JSON, truncated reads
        return Response(None, {}, None, _redact(f"{type(e).__name__}: {e}", secret))


def _shape_error(r: Response, want: type) -> Response:
    """A 200 whose JSON is the wrong shape is an error, not a crash in whoever reads the body."""
    if r.body is None or isinstance(r.body, want):
        return r
    return r._replace(body=None, error=f"unexpected body: {type(r.body).__name__}")


def sgo_events(params: dict, paginate: bool = True) -> Response:
    """GET /v2/events, following nextCursor. Success body is the list of page bodies."""
    key = api_key("SGO_API_KEY")
    pages, cursor = [], None
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


def oddsapi_events() -> Response:
    """Quota-free schedule call; its headers still carry x-requests-used."""
    key = api_key("THE_ODDS_API_KEY")
    return _shape_error(_get(f"{ODDSAPI_URL}/events?{urllib.parse.urlencode({'apiKey': key})}", {}, key), list)


def oddsapi_odds(params: dict) -> Response:
    key = api_key("THE_ODDS_API_KEY")
    return _shape_error(_get(f"{ODDSAPI_URL}/odds?{urllib.parse.urlencode({**params, 'apiKey': key})}", {}, key), list)
