# NBA Prediction - Data Preparation Pipeline

The dataPrep pipeline has been split into two separate pipelines for better modularity and flexibility.

## Pipeline Modes

### 1. **Full Pipeline** (pipeline.py)
Runs all steps in sequence. Useful when you need the complete workflow.

```bash
# Run full pipeline
python -m src.ingest.dataPrep.pipeline

# Or with explicit mode
python -m src.ingest.dataPrep.pipeline --mode full
```

**Steps executed:** 1 → 2 → 3 → 3a → 4 → 5 → 6 → 6a

---

### 2. **Ingest Raw Data** (ingest_raw_data.py)
Fetches raw data from NBA API and prepares it for feature calculation.

```bash
python -m src.ingest.dataPrep.ingest_raw_data
```

**Steps executed:**
- **Step 1:** Fetch game data (games, scores, spreads, etc.)
- **Step 3:** Fetch player-level box score data
- **Step 3a:** Roll player data up to team level

**Output:** Raw game and player data ready for feature engineering

---

### 3. **Calculate Features** (calculate_features.py)
Computes derived features from ingested raw data.

```bash
python -m src.ingest.dataPrep.calculate_features
```

**Steps executed:**
- **Step 2:** Elo, rest days, back-to-back games, rolling team stats
- **Step 4:** Team/player features, injury flags, rest status
- **Step 5:** Travel distance and fatigue features
- **Step 6:** Additional feature calculations
- **Step 6a:** Final feature transformations

**Input:** Raw data from ingest pipeline
**Output:** Feature-engineered dataset ready for modeling

---

## Usage Patterns

### Pattern 1: Development & Iteration
1. Run ingest once to cache raw data
2. Iterate on features quickly without re-fetching data

```bash
# First time - ingest data
python -m src.ingest.dataPrep.ingest_raw_data

# Then rapidly test feature changes
python -m src.ingest.dataPrep.calculate_features
python -m src.ingest.dataPrep.calculate_features
python -m src.ingest.dataPrep.calculate_features
```

### Pattern 2: Full Refresh
When you need completely fresh data:

```bash
python -m src.ingest.dataPrep.pipeline --mode full
```

### Pattern 3: Feature-Only Pipeline
When raw data is already cached and you only want to recalculate features:

```bash
python -m src.ingest.dataPrep.pipeline --mode features
```

---

## Pipeline Architecture

```
pipeline.py (main entry point)
├── ingest_raw_data.py (can run independently)
│   ├── step1.run()    - Fetch games
│   ├── step3.run()    - Fetch player box scores
│   └── step3a.run()   - Roll up to team level
└── calculate_features.py (can run independently)
    ├── step2.run()    - Elo, rest, B2B, rolling stats
    ├── step4.run()    - Features, injury, rest
    ├── step5.run()    - Travel, fatigue
    ├── step6.run()    - Additional features
    └── step6a.run()   - Final transformations
```

---

## Benefits of This Split

✅ **Faster Iteration:** Recalculate features without waiting for API calls
✅ **Modularity:** Each pipeline can be used independently
✅ **Flexibility:** Mix and match - use cached data with updated feature logic
✅ **Debugging:** Easier to isolate issues in ingestion vs. feature calculation
✅ **Scalability:** Can parallelize ingestion and feature calculation if needed
