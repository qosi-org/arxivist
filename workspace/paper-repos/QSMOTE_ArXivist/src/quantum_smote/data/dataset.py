"""Dataset utilities for Quantum-SMOTE.

Provides TelcoChurnDataset for loading the raw CSV dataset.
"""
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TelcoChurnDataset:
    """Helper to locate and load the Telco Customer Churn CSV.

    The loader is forgiving: if `data_path` does not exist it will attempt
    to locate common Telco dataset filenames in the repository root.

    Usage:
        df = TelcoChurnDataset.load('data/telco_customer_churn.csv')
    """

    COMMON_NAMES = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_customer_churn.csv",
        "Telco-Customer-Churn.csv",
        "telco-customer-churn.csv",
    ]

    @staticmethod
    def _search_repo_for_telco() -> Path:
        # Search current working directory and its direct children for likely filenames
        cwd = Path.cwd()
        # check common names first
        for name in TelcoChurnDataset.COMMON_NAMES:
            p = cwd / name
            if p.exists():
                return p

        # fallback: any csv with 'telco' in the filename
        for p in cwd.glob("**/*.csv"):
            if "telco" in p.name.lower():
                return p

        return None

    @staticmethod
    def load(data_path: str) -> pd.DataFrame:
        p = Path(data_path)
        if not p.exists():
            logger.warning("Configured dataset path '%s' not found; searching repo for Telco CSV...", data_path)
            found = TelcoChurnDataset._search_repo_for_telco()
            if found is None:
                raise FileNotFoundError(f"Dataset not found at path: {data_path} and no telco CSV found in repo")
            logger.info("Found dataset at %s (falling back to discovered file)", found)
            p = found

        # Read CSV with sensible defaults
        df = pd.read_csv(p)

        # Basic validation
        if df.empty:
            raise ValueError(f"Loaded dataset is empty: {p}")

        return df
