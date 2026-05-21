"""Classical classifier factory for Quantum-SMOTE evaluation.

Builds Random Forest or Logistic Regression models and trains/evaluates them
using the metrics required by the paper.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from quantum_smote.evaluation.metrics import ClassificationMetrics


class ClassifierFactory:
    """Instantiate and evaluate classical classifiers."""

    @staticmethod
    def build(model_name: str, config: dict) -> BaseEstimator:
        """Build a classifier from config.

        Supported model names:
        - 'rf' / 'random_forest'
        - 'lr' / 'logistic_regression'
        """
        model_name = model_name.lower().strip()
        cfg = config or {}

        if model_name in {"rf", "random_forest"}:
            rf_cfg = cfg.get("random_forest", cfg)
            return RandomForestClassifier(
                n_estimators=int(rf_cfg.get("n_estimators", 100)),
                random_state=rf_cfg.get("random_state", 42),
                n_jobs=rf_cfg.get("n_jobs", -1),
            )

        if model_name in {"lr", "logistic_regression"}:
            lr_cfg = cfg.get("logistic_regression", cfg)
            return LogisticRegression(
                solver=lr_cfg.get("solver", "lbfgs"),
                max_iter=int(lr_cfg.get("max_iter", 1000)),
                C=float(lr_cfg.get("C", 1.0)),
                random_state=lr_cfg.get("random_state", 42),
            )

        raise ValueError(f"Unsupported model_name: {model_name}")

    @staticmethod
    def train_evaluate(model: BaseEstimator, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train the classifier and compute evaluation metrics."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).ravel()
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test).ravel()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # convert raw scores to [0, 1] via logistic transform for metrics
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_proba = y_pred.astype(float)

        metrics = ClassificationMetrics.compute(y_test, y_pred, y_proba)
        metrics["y_pred"] = y_pred
        metrics["y_proba"] = y_proba
        return metrics
