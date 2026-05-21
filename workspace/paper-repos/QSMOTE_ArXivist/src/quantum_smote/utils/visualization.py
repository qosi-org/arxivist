"""Visualization helpers for reproducing the paper's figures.

The methods mirror the architecture plan and create common evaluation plots
for synthetic-vs-real comparisons, distributions, confusion matrices, ROC,
and PR curves.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc


class Visualizer:
    """Plot utilities for Quantum-SMOTE results."""

    @staticmethod
    def plot_scatter(X, y, syn_X, feature_names, title) -> None:
        """Plot a 2D scatter using the first two features.

        Parameters follow the architecture plan. If more than two dimensions are
        present, only the first two are shown.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        syn_X = np.asarray(syn_X) if syn_X is not None else np.empty((0, X.shape[1] if X.ndim == 2 else 0))

        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must be a 2D array with at least two features for scatter plotting")

        plt.figure(figsize=(9, 6))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c="#4c78a8", s=12, alpha=0.7, label="Class 0")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#f58518", s=12, alpha=0.7, label="Class 1")

        if syn_X.size > 0:
            plt.scatter(syn_X[:, 0], syn_X[:, 1], c="#54a24b", s=14, alpha=0.65, label="Synthetic")

        xlabel = feature_names[0] if feature_names and len(feature_names) > 0 else "Feature 0"
        ylabel = feature_names[1] if feature_names and len(feature_names) > 1 else "Feature 1"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(frameon=False)
        plt.tight_layout()

    @staticmethod
    def plot_confusion_matrix(cm, title) -> None:
        """Plot a confusion matrix heatmap."""
        cm = np.asarray(cm)
        if cm.shape != (2, 2):
            raise ValueError("Confusion matrix must be 2x2")

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()

    @staticmethod
    def plot_roc_curve(y_true, y_proba, title) -> None:
        """Plot the ROC curve."""
        y_true = np.asarray(y_true).ravel()
        y_proba = np.asarray(y_proba).ravel()
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#4c78a8", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(frameon=False)
        plt.tight_layout()

    @staticmethod
    def plot_pr_curve(y_true, y_proba, title) -> None:
        """Plot the Precision-Recall curve."""
        y_true = np.asarray(y_true).ravel()
        y_proba = np.asarray(y_proba).ravel()
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="#f58518", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(frameon=False)
        plt.tight_layout()

    @staticmethod
    def plot_distribution(X, feature_idx, title) -> None:
        """Plot a feature distribution histogram."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D for distribution plots")
        if not (0 <= int(feature_idx) < X.shape[1]):
            raise IndexError("feature_idx out of range")

        plt.figure(figsize=(7, 5))
        sns.histplot(X[:, int(feature_idx)], bins=30, kde=True, color="#4c78a8")
        plt.xlabel(f"Feature {int(feature_idx)}")
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
