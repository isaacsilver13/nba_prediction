# Data Layout

Large datasets should live outside this repository in a dedicated shared-data location.

## Recommended structure

- Shared data root: `NBA_DATA_DIR`
- Required files under `NBA_DATA_DIR`:
  - `processed/df_model_3.csv`
  - `processed/nba_games_with_game_id_processed.csv`
  - `odds/nba_2008-2025.csv`

## Local development

- Keep only minimal sample files in `data/sample/` for CI and smoke tests.
- If `NBA_DATA_DIR` is not set, code falls back to this repo's `data/` directory.
