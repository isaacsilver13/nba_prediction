# NBA Prediction Copilot Instructions

## Project boundary

This repository contains the NBA data-ingestion, feature-engineering, modeling, betting simulation, dashboard, and autoresearch workflows. Work from `repos/nba_prediction` unless the user explicitly names another project.

## Source of truth

- `README.md` describes setup and repository-relative entrypoints.
- `src/ingest/dataPrep/config.py` owns shared DataPrep defaults and per-step overrides.
- `src/ingest/dataPrep/pipeline.py` owns DataPrep orchestration.
- `program.md` and `experiment.py` define autoresearch constraints. Treat the `FROZEN` section and results format as protected.
- `src/ingest/nba_link_ingest/` owns lineup appearance ingestion. Use its date artifacts and logs instead of inventing a second cache format.

## Data and cache rules

- Run Python commands from this repository root so `src` imports and relative data paths resolve consistently.
- Prefer read-through caches and complete artifact reconstruction over partial-cache assumptions.
- Preserve existing cache paths and schemas unless a migration is explicit. Check cache completeness, file dates, and column names before fetching again.
- Do not commit raw or processed datasets, generated outputs, model artifacts, runtime logs, or credentials. Small reproducible fixtures belong under `data/sample/`.
- For lineup ingestion, official boxscore-listed players count as appearances, including DNPs. Do not add separate player-stat downloads to that workflow.

## Modeling rules

- Keep walk-forward splits chronological and fit preprocessing, feature selection, and transforms on the training fold only.
- Treat leakage filters and target construction as correctness boundaries, not tuning knobs.
- Report RMSE, calibration, odds-match quality, ROI, drawdown, and Kelly behavior separately. Do not treat a single profitable or unprofitable run as proof of a model change.
- Preserve join audits and identify fallback or default odds rows before interpreting betting results.
- Make one focused experiment change at a time and record the exact configuration and artifact paths.

## Working method

Inspect the owning module, neighboring tests, and current artifact schema before editing. Make the smallest change that addresses the request. Run the narrowest import, compile, test, dry-run, or diagnostic check after each focused edit. Do not claim a historical fix still works without current evidence.
