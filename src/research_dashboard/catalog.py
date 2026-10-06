"""Discover run bundles under <outputs>/research_runs and describe legacy (unbundled) files."""

import os
from datetime import datetime
from pathlib import Path

from .manifest import Run, load_manifest

LEGACY_FILES = [
    "results.tsv",
    "outputs/kelly_bets_summary.csv",
    "outputs/copilot_betting_summary.csv",
    "outputs/copilot_model_rmse.csv",
]


def outputs_dir(base: Path) -> Path:
    return Path(os.environ.get("NBA_OUTPUTS_DIR", base / "outputs"))


def list_runs(outputs: Path) -> list:
    """Newest first by created_at (failed manifests sort last)."""
    runs = [load_manifest(m.parent) for m in sorted(Path(outputs).glob("research_runs/*/manifest.json"))]
    return sorted(runs, key=lambda r: r.data.get("created_at", ""), reverse=True)


def legacy_entry(base: Path) -> dict:
    """Read-only description of legacy files: they share no run ID, so never treat them as one run."""
    files = []
    for rel in LEGACY_FILES:
        p = base / rel
        files.append({
            "file": rel,
            "exists": p.is_file(),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if p.is_file() else None,
        })
    return {
        "label": "Legacy / unbundled",
        "files": files,
        "notes": ["No shared run ID across these files.",
                  "No per-bet odds-quality tier; only aggregate join-audit counts exist."],
    }
