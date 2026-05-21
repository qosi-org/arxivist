"""
evaluation/metrics.py — Evaluation metrics for QSVM fraud detection.

Computes all four metrics reported in the paper (Table I & II):
  - Accuracy
  - F1-score (macro or binary weighted)
  - Recall
  - AUC (ROC-AUC)

Also generates:
  - Confusion matrix heatmap (paper Figure 4)
  - ROC curve

Paper reference: Section IV-B, Tables I and II, Figure 4.

Target metrics (primary config — QSVM-10qubit + Quantum-SMOTE):
  accuracy = 98.8%,  F1 = 0.962,  recall = 0.945,  AUC = 0.992
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available. Plots will be skipped.")


class FraudMetrics:
    """
    Computes and reports fraud detection evaluation metrics.

    Matches the exact metrics reported in Table I and Table II of the paper.

    Usage:
        >>> metrics = FraudMetrics()
        >>> results = metrics.compute(y_true, y_pred, y_score)
        >>> metrics.print_report(results)
        >>> metrics.save_confusion_matrix_plot(y_true, y_pred, "results/cm.png")
    """

    # Paper target values for primary config (QSVM-10qubit + Quantum-SMOTE)
    PAPER_TARGETS = {
        "accuracy": 98.8,
        "f1": 0.962,
        "recall": 0.945,
        "auc": 0.992,
    }

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        label: str = "QSVM",
    ) -> dict:
        """
        Compute all evaluation metrics from paper Tables I & II.

        Args:
            y_true:  Ground-truth labels [N] int {0, 1}.
            y_pred:  Predicted labels [N] int {0, 1}.
            y_score: Predicted fraud probability [N] float — required for AUC.
                     Use QSVM.predict_proba()[:, 1] or decision_function() output.
            label:   Model name for reporting.

        Returns:
            Dict with keys: accuracy, f1, recall, precision, auc, confusion_matrix,
                            classification_report.
        """
        assert len(y_true) == len(y_pred), "y_true and y_pred length mismatch"

        acc = accuracy_score(y_true, y_pred) * 100  # paper reports as percentage
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )

        auc = None
        if y_score is not None:
            score = y_score[:, 1] if y_score.ndim == 2 else y_score
            auc = roc_auc_score(y_true, score)

        results = {
            "label": label,
            "accuracy": round(acc, 1),
            "f1": round(f1, 3),
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "auc": round(auc, 3) if auc is not None else None,
            "confusion_matrix": cm,
            "classification_report": report,
            "n_test": len(y_true),
            "n_fraud_true": int(y_true.sum()),
            "n_fraud_pred": int(y_pred.sum()),
        }

        logger.info(
            "[%s] accuracy=%.1f%%  F1=%.3f  recall=%.3f  precision=%.3f  AUC=%s",
            label, acc, f1, recall, precision,
            f"{auc:.3f}" if auc is not None else "N/A",
        )
        return results

    def print_report(self, results: dict) -> None:
        """Print formatted metric table to stdout, compared against paper targets."""
        label = results.get("label", "Model")
        print(f"\n{'='*60}")
        print(f"  Results: {label}")
        print(f"{'='*60}")
        print(f"  {'Metric':<15} {'Computed':>10}  {'Paper Target':>14}")
        print(f"  {'-'*45}")

        rows = [
            ("Accuracy (%)", "accuracy", self.PAPER_TARGETS["accuracy"], "{:.1f}"),
            ("F1-score",     "f1",       self.PAPER_TARGETS["f1"],       "{:.3f}"),
            ("Recall",       "recall",   self.PAPER_TARGETS["recall"],   "{:.3f}"),
            ("AUC",          "auc",      self.PAPER_TARGETS["auc"],      "{:.3f}"),
        ]
        for display_name, key, target, fmt in rows:
            val = results.get(key)
            val_str = fmt.format(val) if val is not None else "N/A"
            target_str = fmt.format(target)
            delta = ""
            if val is not None:
                diff = val - target
                delta = f"  (Δ {diff:+.3f})"
            print(f"  {display_name:<15} {val_str:>10}  {target_str:>14}{delta}")

        print(f"\n  n_test={results['n_test']}, "
              f"n_fraud_true={results['n_fraud_true']}, "
              f"n_fraud_pred={results['n_fraud_pred']}")
        print(f"\n{results['classification_report']}")
        print(f"{'='*60}\n")

    def save_confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str,
        title: str = "Confusion Matrix",
    ) -> None:
        """
        Save confusion matrix heatmap (reproduces paper Figure 4).

        Args:
            y_true:    Ground-truth labels.
            y_pred:    Predicted labels.
            save_path: Output PNG path.
            title:     Plot title.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available — skipping confusion matrix plot.")
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)

    def save_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        save_path: str,
        title: str = "ROC Curve",
    ) -> None:
        """
        Save ROC curve plot.

        Args:
            y_true:    Ground-truth labels.
            y_score:   Fraud probability scores [N] or [N, 2].
            save_path: Output PNG path.
            title:     Plot title.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available — skipping ROC curve plot.")
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        score = y_score[:, 1] if y_score.ndim == 2 else y_score

        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, score, ax=ax, name=title)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("ROC curve saved to %s", save_path)

    def compare_table(self, results_list: list[dict]) -> str:
        """
        Format multiple results as a comparison table (reproduces Tables I & II).

        Args:
            results_list: List of results dicts from compute().

        Returns:
            Formatted table string.
        """
        header = f"\n{'Label':<35} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'AUC':>8}"
        sep = "-" * 75
        rows = [header, sep]
        for r in results_list:
            auc_str = f"{r['auc']:.3f}" if r.get("auc") is not None else " N/A"
            rows.append(
                f"{r['label']:<35} {r['accuracy']:>9.1f}% "
                f"{r['f1']:>8.3f} {r['recall']:>8.3f} {auc_str:>8}"
            )
        rows.append(sep)
        return "\n".join(rows)
