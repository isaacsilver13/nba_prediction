"""Pull free Kalshi settled-market history (no API key) for NBA series into $NBA_DATA_DIR/kalshi/.

    python tools/kalshi_history.py KXNBAPTS KXNBAREB              # market lists -> <SERIES>_markets.jsonl
    python tools/kalshi_history.py KXNBAPTS --candles             # + hourly candles before tip -> <SERIES>_pregame.csv

Markets settled before GET /historical/cutoff live only under /historical/*. Tip time is taken as
expected_expiration_time - 3h (Kalshi sets NBA game props to expire 3h after scheduled tip; implied tips cluster at
7:00/7:30/8:00pm ET). Candle output is resumable: re-run the same command to continue an interrupted pull.
"""
import argparse
import csv
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(os.environ.get("NBA_DATA_DIR", Path(__file__).resolve().parents[1] / "data")) / "kalshi"
RATE = 1 / 8  # seconds between requests; unthrottled bursts get 429s
COLS = ["ticker", "tip_ts", "end_ts", "yes_bid", "yes_ask", "price_close", "volume", "open_interest"]

_lock, _next = threading.Lock(), [0.0]


def fetch(url):
    """GET JSON with a global rate limit; retries 429/5xx/network errors, returns None on other 4xx or give-up."""
    for _ in range(6):
        with _lock:
            t = _next[0] = max(time.time(), _next[0] + RATE)
        time.sleep(max(0.0, t - time.time()))
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    return None


def pull_markets(series):
    path = OUT / f"{series}_markets.jsonl"
    n, cursor = 0, ""
    with open(path, "w") as f:
        while True:
            r = fetch(f"{BASE}/historical/markets?series_ticker={series}&limit=1000" + (f"&cursor={cursor}" if cursor else ""))
            if r is None:
                raise RuntimeError(f"market list fetch failed for {series} after {n} markets")
            for m in r["markets"]:
                f.write(json.dumps(m) + "\n")
            n += len(r["markets"])
            cursor = r.get("cursor")
            if not cursor or not r["markets"]:
                break
    print(f"{series}: {n} markets -> {path}")


def tip_ts(m):
    return int(datetime.fromisoformat(m["expected_expiration_time"].replace("Z", "+00:00")).timestamp()) - 3 * 3600


def candle_rows(m, hours):
    tip = tip_ts(m)
    r = fetch(f"{BASE}/historical/markets/{m['ticker']}/candlesticks?start_ts={tip - hours * 3600}&end_ts={tip}&period_interval=60")
    if r is None:
        return None  # not written, so a re-run retries it
    rows = [[m["ticker"], tip, c["end_period_ts"], c["yes_bid"]["close"], c["yes_ask"]["close"],
             (c.get("price") or {}).get("close"), c.get("volume"), c.get("open_interest")] for c in r["candlesticks"]]
    return rows or [[m["ticker"], tip] + [""] * 6]  # marker row: fetched, no candles


def pull_candles(series, hours, name="pregame"):
    markets = [json.loads(line) for line in open(OUT / f"{series}_markets.jsonl")]
    path = OUT / f"{series}_{name}.csv"
    done = set()
    if path.exists():
        seen = list(dict.fromkeys(r["ticker"] for r in csv.DictReader(open(path))))
        done = set(seen[:-50])  # the tail may be partially written by an interrupted run; refetch it (readers dedupe)
    todo = [m for m in markets if m["ticker"] not in done]
    print(f"{series}: {len(todo)} markets to fetch candles for")
    with open(path, "a", newline="") as f, ThreadPoolExecutor(8) as ex:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(COLS)
        for i, rows in enumerate(ex.map(lambda m: candle_rows(m, hours), todo), 1):
            if rows:
                w.writerows(rows)
            if i % 1000 == 0:
                f.flush()
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"{series}: candles -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("series", nargs="+")
    ap.add_argument("--candles", action="store_true", help="also pull hourly candles before tip")
    ap.add_argument("--hours", type=int, default=6, help="hours of candles before tip (default 6)")
    ap.add_argument("--name", default="pregame", help="candle file is <SERIES>_<name>.csv (use e.g. 'early48' with --hours 48)")
    ap.add_argument("--skip-markets", action="store_true", help="reuse existing <SERIES>_markets.jsonl")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.series:
        if not (a.skip_markets and (OUT / f"{s}_markets.jsonl").exists()):
            pull_markets(s)
        if a.candles:
            pull_candles(s, a.hours, a.name)
