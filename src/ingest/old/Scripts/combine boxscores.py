import os
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
BOXSCORES_DIR = Path("data/cache/boxscores")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "all_boxscores.csv"

# =========================
# VALIDATION
# =========================
if not BOXSCORES_DIR.exists():
    raise FileNotFoundError(f"Boxscores directory not found: {BOXSCORES_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD + CONCAT FILES
# =========================
dfs = []
failed_files = []

for file in sorted(BOXSCORES_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(file)

        if df.empty:
            print(f"⚠️ Skipping empty file: {file.name}")
            continue

        df["source_file"] = file.name  # optional traceability
        print(f"Added {file}")
        dfs.append(df)

    except Exception as e:
        print(f"❌ Failed to read {file.name}: {e}")
        failed_files.append(file.name)

# =========================
# FINAL CONCAT
# =========================
if not dfs:
    raise RuntimeError("No valid boxscore files were loaded.")

final_df = pd.concat(dfs, ignore_index=True)

# =========================
# SAVE OUTPUT
# =========================
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved {len(final_df):,} rows to {OUTPUT_FILE}")
print(f"📄 Files processed: {len(dfs)}")
print(f"⚠️ Files failed: {len(failed_files)}")

if failed_files:
    print("Failed files:")
    for f in failed_files:
        print(f" - {f}")
