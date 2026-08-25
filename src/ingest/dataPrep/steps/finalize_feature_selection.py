"""Finalize feature selection and write the model-ready table."""

import os
import pandas as pd

try:
    # When imported as a package module
    from ..config import get_config, resolve_step_config
    from ..helpers import load_cached_output, save_output
except ImportError:
    # When loaded dynamically or as direct import
    try:
        from ingest.dataPrep.config import get_config, resolve_step_config
        from ingest.dataPrep.helpers import load_cached_output, save_output
    except ImportError:
        # Fallback for different import contexts
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config import get_config, resolve_step_config
        from helpers import load_cached_output, save_output


def run(config=None, inputs=None):
    """Run step 6a to select final model features.

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior step outputs; uses `df_model_2` when provided.

    Returns
    -------
    dict
        Dictionary with `df_model_3` dataframe.
    """
    cfg = resolve_step_config(get_config(), "step6a") if config is None else resolve_step_config(config, "step6a")
    paths = cfg["paths"]
    columns_cfg = cfg["columns"]
    cache_cfg = cfg.get("cache", {})

    output_path = paths["output_model"]

    # Try to load from cache
    df_cached, from_cache = load_cached_output(output_path, cache_cfg)
    if from_cache and not inputs:
        print(f"[step6a] Cache hit: loaded final model table from {output_path}")
        return {"df_model_3": df_cached}

    if inputs and "df_model_2" in inputs:
        df = inputs["df_model_2"].copy()
    else:
        df = pd.read_csv(paths["input_model"])

    base_cols_to_keep = columns_cfg["keep"]

    dynamic_prefixes = (
        "home_team_adv_",
        "away_team_adv_",
        "home_team_ff_",
        "away_team_ff_",
        "home_team_sc_",
        "away_team_sc_",
        "home_team_usg_",
        "away_team_usg_",
        "home_team_misc_",
        "away_team_misc_",
        "home_team_ptrk_",
        "away_team_ptrk_",
    )
    dynamic_suffixes = ("_r5", "_r10", "_r20", "_ewm5", "_ewm10", "_ewm20", "_s2d")

    dynamic_cols = [
        c
        for c in df.columns
        if c.startswith(dynamic_prefixes) and c.endswith(dynamic_suffixes)
    ]

    cols_to_keep = [c for c in base_cols_to_keep if c in df.columns] + dynamic_cols
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    print(f"[step6a] Base columns kept: {len([c for c in base_cols_to_keep if c in df.columns]):,}")
    print(f"[step6a] Dynamic V3 columns kept: {len(dynamic_cols):,}")
    df = df.loc[:, cols_to_keep]

    # Save to cache if enabled
    save_output(df, output_path, cache_cfg)
    print(f"[step6a] Saved final feature table to {output_path} | rows={len(df):,}, cols={df.shape[1]:,}")

    return {"df_model_3": df}


if __name__ == "__main__":
    run(get_config())
