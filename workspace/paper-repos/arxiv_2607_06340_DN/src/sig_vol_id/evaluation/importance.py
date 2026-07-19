"""
Feature importance analysis: XGBoost built-in (gain) importance vs.
permutation importance (Section 6.4, Figure 6.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class ImportanceAnalyzer:
    """Computes both importance measures for a trained SignatureXGBClassifier."""

    def builtin_importance(self, clf, feature_names: list[str] | None = None) -> pd.Series:
        """XGBoost's built-in gain-based feature importance.

        Args:
            clf: A trained SignatureXGBClassifier (uses clf.model internally).
            feature_names: Optional names (defaults to sig_0, sig_1, ...).
        """
        importances = clf.model.feature_importances_
        if feature_names is None:
            feature_names = [f"sig_{i}" for i in range(len(importances))]
        s = pd.Series(importances, index=feature_names, name="builtin_importance")
        return s.sort_values(ascending=False)

    def permutation_importance(
        self,
        clf,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str] | None = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.Series:
        """Scale-invariant permutation importance (Breiman 2001): decrease in
        accuracy when each feature is randomly shuffled.
        """
        result = permutation_importance(
            clf.model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, scoring="accuracy"
        )
        if feature_names is None:
            feature_names = [f"sig_{i}" for i in range(X_test.shape[1])]
        s = pd.Series(result.importances_mean, index=feature_names, name="permutation_importance")
        return s.sort_values(ascending=False)

    @staticmethod
    def truncate_at_cumulative(s: pd.Series, threshold: float = 0.9) -> pd.Series:
        """Keep only the top features accounting for `threshold` of total importance
        (matches the paper's Figure 6.4: 'features truncated at 90% cumulative importance')."""
        normalized = s / s.sum()
        cumsum = normalized.cumsum()
        n_keep = (cumsum <= threshold).sum() + 1
        return s.iloc[:n_keep]
