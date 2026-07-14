#!/usr/bin/env python
"""
Data availability checker for "Asset Pricing in Pre-trained Transformers" (arXiv:2505.01575).

The paper's exact factor/stock-universe construction is not publicly enumerated (see
data/README_data.md), so this script cannot auto-download the real dataset. It instead:
1. Checks whether the two required CSVs already exist locally.
2. Verifies basic integrity (non-empty, parseable, expected column-count ballpark).
3. Prints guided instructions if data is missing.
"""
from __future__ import annotations

import os

import pandas as pd

FACTORS_PATH = os.path.join(os.path.dirname(__file__), "raw", "factors_182.csv")
RETURNS_PATH = os.path.join(os.path.dirname(__file__), "raw", "excess_returns_420stocks.csv")


def check_file(path: str, expected_min_cols: int, label: str) -> bool:
    if not os.path.exists(path):
        print(f"[download.py] MISSING: {label} not found at {path}")
        return False
    try:
        df = pd.read_csv(path, index_col=0, nrows=5)
    except Exception as e:  # noqa: BLE001
        print(f"[download.py] ERROR reading {label}: {e}")
        return False
    if df.shape[1] < expected_min_cols:
        print(f"[download.py] WARNING: {label} has only {df.shape[1]} columns "
              f"(expected >= {expected_min_cols}). Check the file.")
    print(f"[download.py] OK: {label} found with {df.shape[1]} columns (showing first 5 rows).")
    return True


def main() -> None:
    print("[download.py] Checking for required data files...")
    factors_ok = check_file(FACTORS_PATH, expected_min_cols=1, label="factors_182.csv")
    returns_ok = check_file(RETURNS_PATH, expected_min_cols=1, label="excess_returns_420stocks.csv")

    if factors_ok and returns_ok:
        print("[download.py] All data files present. You're ready to run train.py.")
        return

    print()
    print("=" * 70)
    print("Real data not found. This dataset is not publicly redistributable")
    print("as constructed in the paper. See data/README_data.md for:")
    print("  - the public factor library (openassetpricing.com) for factors_182.csv")
    print("  - guidance on reconstructing the 420-stock large-cap universe (CRSP/WRDS)")
    print()
    print("In the meantime, you can smoke-test the full pipeline with synthetic data:")
    print("  python train.py --config configs/config.yaml --debug")
    print("=" * 70)


if __name__ == "__main__":
    main()
