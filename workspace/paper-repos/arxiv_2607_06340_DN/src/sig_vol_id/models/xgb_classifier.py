"""
XGBoost multi-class classifier for signature-based volatility model
identification (Section 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from xgboost import XGBClassifier


class SignatureXGBClassifier:
    """Wraps XGBClassifier with the paper's exact hyperparameters.

    Args:
        n_classes: Number of model classes for this experiment.
        learning_rate: Paper value: 0.05.
        max_depth: Paper value: 6.
        n_estimators: Paper value: 500.
        tree_method: "hist" (CPU) or "gpu_hist" / device="cuda" (GPU).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_classes: int,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        n_estimators: int = 500,
        tree_method: str = "hist",
        random_state: int = 42,
    ):
        self.n_classes = n_classes
        # NOTE: we intentionally do NOT pass objective="multi:softprob" or
        # num_class explicitly. Doing so causes XGBClassifier.predict() to
        # return raw probability arrays instead of class labels in recent
        # xgboost versions (verified: xgboost==3.3.0) -- a real bug found
        # during testing, not a paper-fidelity choice. Letting XGBClassifier
        # auto-detect the objective and num_class from `y` at fit() time
        # gives identical underlying boosting behavior (same loss function,
        # same hyperparameters) while keeping predict() well-behaved.
        self.model = XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            tree_method=tree_method,
            random_state=random_state,
            eval_metric="mlogloss",
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the classifier. No separate validation set is used, per the paper."""
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def confusion_matrix(
        self, X: np.ndarray, y: np.ndarray, class_names: list[str], as_percentage: bool = False
    ) -> pd.DataFrame:
        """Compute the confusion matrix as a labeled DataFrame.

        Args:
            X: Test features.
            y: True labels.
            class_names: Names for each class index, in order.
            as_percentage: If True, normalize each row to percentages
                (matches the paper's Section 6 reporting convention; Section 5
                reports absolute counts, i.e. as_percentage=False).
        """
        preds = self.predict(X)
        cm = sk_confusion_matrix(y, preds, labels=list(range(self.n_classes)))
        if as_percentage:
            cm = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100
        return pd.DataFrame(cm, index=class_names, columns=class_names)
