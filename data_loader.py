"""
data_loader.py
Generic loader for whatever Everesteer hands you tonight.
Edit COLUMN CONFIG at the top once you see the real schema in the Discord walkthrough.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# EDIT THESE ONCE YOU SEE THE REAL DATA (from Discord walkthrough)
# ============================================================
TARGET_COL = "target"          # the thing you're predicting
TIME_COL = "timestamp"         # datetime / tick column, if any
ID_COL = "id"                  # row identifier used in submission
DROP_COLS = []                 # any columns to exclude from features (leaks, ids, etc.)
# ============================================================


def load_data(path: str) -> pd.DataFrame:
    """Loads csv, parquet, or json depending on extension. Adjust if the format differs."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unrecognized file type: {path.suffix}")

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


def quick_profile(df: pd.DataFrame, n=5):
    """Fast sanity check the moment you get real data. Run this first, always."""
    print("shape:", df.shape)
    print("\ndtypes:\n", df.dtypes)
    print("\nmissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nhead:\n", df.head(n))
    if TARGET_COL in df.columns:
        print(f"\ntarget ({TARGET_COL}) summary:\n", df[TARGET_COL].describe())
    return df.describe(include="all").T


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = load_data(sys.argv[1])
        quick_profile(df)
    else:
        print("Usage: python data_loader.py <path_to_data_file>")
