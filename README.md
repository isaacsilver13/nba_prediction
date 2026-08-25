# nba_prediction

This repository contains the NBA model training, backtesting, and dashboard workflow.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure data locations (optional):

```bash
set NBA_DATA_DIR=C:\\path\\to\\shared-data\\nba
set NBA_OUTPUTS_DIR=C:\\path\\to\\shared-data\\nba\\outputs
```

If these are not set, defaults are `./data` and `./outputs` under this repo.

4. Run an experiment:

```bash
python experiment.py
```

5. Launch dashboard:

```bash
streamlit run dashboard.py
```

6. Run the autoresearch loop from any working directory:

```bash
python /path/to/nba_prediction/autorun.py --dry-run
python /path/to/nba_prediction/autorun.py --groq --max-iters 20
```

`autorun.py` resolves `experiment.py`, `program.md`, `results.tsv`, and
`experiments/` relative to its own repository, so callers do not need to
change directories first. Do not commit raw or processed datasets, generated
outputs, model artifacts, or API credentials; use `data/sample/` for small
reproducible fixtures.
