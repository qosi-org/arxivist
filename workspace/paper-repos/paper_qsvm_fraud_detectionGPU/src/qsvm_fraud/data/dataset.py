"""
data/dataset.py — Kaggle Credit Card Fraud Detection dataset loader.

Handles loading, feature selection (KBest), and train/test splitting.

Paper reference: Section IV-A — Data Preparation.
  - Dataset: Kaggle Credit Card Fraud Detection (284,807 transactions, 492 fraud)
  - Features: V1–V28 (PCA-derived), Time, Amount → 30 total
  - Feature selection: SelectKBest, top 10 features (paper states "top 10 above V19")
  - Class imbalance: fraud = 0.172% of all transactions

Assumptions (from SIR):
  - score_func: f_classif (ASSUMED; confidence 0.60)
  - test_size: 0.20 (ASSUMED 80/20 split; confidence 0.55)
  - Selected features are the top-10 by KBest score (exact set logged at runtime)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# KBest scoring functions available
SCORE_FUNCS = {
    "f_classif": f_classif,
    "chi2": chi2,
    "mutual_info_classif": mutual_info_classif,
}


class FraudDataset:
    """
    Loader for the Kaggle Credit Card Fraud Detection dataset.

    Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Size: 284,807 transactions over 2 days; 492 (0.172%) are fraudulent.
    Features: V1–V28 (PCA), Time, Amount. Target: Class (0=legit, 1=fraud).

    Paper: Section IV-A.

    Args:
        n_features:    Number of top features to select via KBest. Paper: 10.
        score_func:    KBest scoring function name. ASSUMED='f_classif' (confidence 0.60).
        test_size:     Fraction for test split. ASSUMED=0.20 (confidence 0.55).
        random_state:  Reproducibility seed.
        max_samples:   If set, subsample to this many rows (debug/speed use only).
    """

    TARGET_COL = "Class"
    ALL_FEATURE_COLS = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    def __init__(
        self,
        n_features: int = 10,               # Paper: top-10 KBest (confidence: 0.90)
        score_func: str = "f_classif",      # ASSUMED (confidence: 0.60)
        test_size: float = 0.20,            # ASSUMED (confidence: 0.55)
        random_state: int = 42,
        max_samples: Optional[int] = None,  # Debug only
    ) -> None:
        if score_func not in SCORE_FUNCS:
            raise ValueError(
                f"score_func must be one of {list(SCORE_FUNCS)}; got {score_func!r}"
            )
        self.n_features = n_features
        self.score_func = score_func
        self.test_size = test_size
        self.random_state = random_state
        self.max_samples = max_samples

        self._selector: Optional[SelectKBest] = None
        self.selected_feature_names: list[str] = []

    def load(self, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the raw creditcard.csv dataset.

        Args:
            csv_path: Path to creditcard.csv.

        Returns:
            X: [N, 30] float64 — all features.
            y: [N] int32 — labels {0, 1}.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}.\n"
                "Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "Or run: python scripts/download_data.py --output-dir data/raw/"
            )

        logger.info("Loading dataset from %s...", csv_path)
        df = pd.read_csv(csv_path)

        # Validate expected columns
        missing = [c for c in self.ALL_FEATURE_COLS + [self.TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing expected columns: {missing}")

        if self.max_samples is not None:
            df = df.sample(
                n=min(self.max_samples, len(df)),
                random_state=self.random_state,
            ).reset_index(drop=True)
            logger.warning("DEBUG: subsampled to %d rows (max_samples=%d)", len(df), self.max_samples)

        X = df[self.ALL_FEATURE_COLS].values.astype(np.float64)
        y = df[self.TARGET_COL].values.astype(np.int32)

        n_fraud = int(y.sum())
        logger.info(
            "Dataset loaded: %d samples, %d features, %d fraud (%.3f%%)",
            len(X), X.shape[1], n_fraud, 100 * n_fraud / len(X),
        )
        return X, y

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reuse_fitted: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Select top-k features using SelectKBest.

        Paper: Section IV-A — "utilizes KBest to select features based on their
        scores … selects the top 10 features above V19."

        ASSUMPTION: score_func=f_classif (ASSUMED; confidence 0.60).
        The selected feature names are logged so you can verify against the paper's
        Figure 3 (Feature Scores Rank).

        Args:
            X:             [N, 30] full feature matrix.
            y:             [N] labels.
            reuse_fitted:  If True, apply a previously fitted selector (for test data).

        Returns:
            X_reduced: [N, n_features] with top-k columns only.
            feature_names: List of selected column names.
        """
        if reuse_fitted and self._selector is not None:
            X_reduced = self._selector.transform(X)
            return X_reduced, self.selected_feature_names

        # Fit KBest on training data
        selector = SelectKBest(
            score_func=SCORE_FUNCS[self.score_func],
            k=self.n_features,
        )
        X_reduced = selector.fit_transform(X, y)
        self._selector = selector

        # Record which features were selected
        mask = selector.get_support()
        self.selected_feature_names = [
            col for col, selected in zip(self.ALL_FEATURE_COLS, mask) if selected
        ]
        logger.info(
            "KBest feature selection (%s, k=%d): selected %s",
            self.score_func, self.n_features, self.selected_feature_names,
        )
        return X_reduced, self.selected_feature_names

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stratified train/test split.

        ASSUMPTION: 80/20 split (confidence: 0.55). The paper does not state the split ratio.

        Args:
            X: [N, D] feature matrix.
            y: [N] labels.

        Returns:
            X_train, X_test, y_train, y_test.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,         # ASSUMED: 0.20 (confidence: 0.55)
            stratify=y,
            random_state=self.random_state,
        )
        logger.info(
            "Train/test split: train=%d (fraud=%d), test=%d (fraud=%d)",
            len(X_train), int(y_train.sum()),
            len(X_test), int(y_test.sum()),
        )
        return X_train, X_test, y_train, y_test

    def get_class_distribution(self, y: np.ndarray) -> dict[int, int]:
        """Return class label → count mapping."""
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.debug("Class distribution: %s", dist)
        return dist
