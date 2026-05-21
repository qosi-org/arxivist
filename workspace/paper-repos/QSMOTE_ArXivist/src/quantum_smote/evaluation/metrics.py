"""Classification metrics for Quantum-SMOTE evaluation.

Computes the paper's evaluation set:
- Accuracy
- F1
- PR-AUC
- ROC-AUC
- Confusion matrix
"""
from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    auc,
)


class ClassificationMetrics:
    """Compute, print, and tabulate classification metrics."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Compute metrics used in the paper.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels.
        y_pred : np.ndarray
            Predicted class labels.
        y_proba : np.ndarray
            Predicted probabilities or scores for the positive class.

        Returns
        -------
        dict
            Dictionary containing accuracy, f1, pr_auc, roc_auc, confusion_matrix.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        y_proba = np.asarray(y_proba).ravel()

        accuracy = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = float(auc(recall, precision))

        roc_auc = float(roc_auc_score(y_true, y_proba))

        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }

    @staticmethod
    def print_report(metrics: Dict[str, Any]) -> None:
        """Pretty-print the metrics dict."""
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

    @staticmethod
    def to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of metrics dicts into a dataframe suitable for tables."""
        rows = []
        for item in results:
            row = {
                k: v
                for k, v in item.items()
                if k in {"accuracy", "f1", "pr_auc", "roc_auc"}
            }
            if "confusion_matrix" in item:
                row["confusion_matrix"] = item["confusion_matrix"]
            if "model" in item:
                row["model"] = item["model"]
            if "condition" in item:
                row["condition"] = item["condition"]
            rows.append(row)
        return pd.DataFrame(rows)
