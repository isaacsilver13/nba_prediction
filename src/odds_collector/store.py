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
