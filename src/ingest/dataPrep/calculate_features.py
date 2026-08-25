"""
Feature Calculation Pipeline: Computes derived features from raw data.

This pipeline focuses on calculating features from ingested data:
- Step 2: Calculate Elo, rest, back-to-back, and rolling team stats
- Step 4: Calculate features including rolling stats for team/player data, injury & rest
- Step 5: Merge player data with game data and add travel distance features
- Step 6: Calculate additional features
- Step 6a: Additional feature calculations
"""

import importlib.util
import os
import sys

try:
    from .config import get_config
    from .steps import calculate_elo_and_rest, engineer_advanced_features
    PACKAGE_DIR = os.path.dirname(__file__)
except ImportError:  # pragma: no cover
    PACKAGE_DIR = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from ingest.dataPrep.config import get_config
    from ingest.dataPrep.steps import calculate_elo_and_rest, engineer_advanced_features


def _load_module(module_name, filename):
    """Dynamically import a pipeline step module by filename.

    Parameters
    ----------
    module_name : str
        Logical name to assign to the loaded module.
    filename : str
        Python filename under the steps directory.

    Returns
    -------
    module
        Imported module object with a callable `run` function.
    """
    file_path = os.path.join(PACKAGE_DIR, "steps", filename)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Dynamically load modules that may have special names
step5 = _load_module("calculate_travel_fatigue", "calculate_travel_fatigue.py")
step6 = _load_module("engineer_rolling_features", "engineer_rolling_features.py")
step6a = _load_module("finalize_feature_selection", "finalize_feature_selection.py")


def run_features_pipeline(config=None, inputs=None):
    """Run feature engineering steps (2, 4, 5, 6, 6a).

    Parameters
    ----------
    config : dict, optional
        Pipeline configuration; defaults to `get_config()` when None.
    inputs : dict, optional
        Prior pipeline outputs, used to avoid re-reading intermediates.

    Returns
    -------
    dict
        Dictionary of step outputs keyed by step names.
    """
    cfg = get_config() if config is None else config
    outputs = {}

    # Step 2: Calculate Elo, rest, B2B, and rolling team stats
    print("=" * 60)
    print("Step 2: Calculating Elo, rest, back-to-back, and rolling stats")
    print("=" * 60)
    outputs.update(calculate_elo_and_rest.run(cfg, inputs or outputs))

    # Step 4: Calculate features including rolling, injury, rest
    print("\n" + "=" * 60)
    print("Step 4: Calculating team/player features and injury/rest flags")
    print("=" * 60)
    outputs.update(engineer_advanced_features.run(cfg, inputs or outputs))

    # Step 5: Add travel distance features
    print("\n" + "=" * 60)
    print("Step 5: Adding travel distance and fatigue features")
    print("=" * 60)
    outputs.update(step5.run(cfg, inputs or outputs))

    # Step 6: Calculate additional features
    print("\n" + "=" * 60)
    print("Step 6: Calculating additional features")
    print("=" * 60)
    outputs.update(step6.run(cfg, inputs or outputs))

    # Step 6a: Additional feature calculations
    print("\n" + "=" * 60)
    print("Step 6a: Final feature transformations")
    print("=" * 60)
    outputs.update(step6a.run(cfg, inputs or outputs))

    return outputs


if __name__ == "__main__":
    run_features_pipeline()
