"""
training/trainer.py — Full QSVM fraud detection training pipeline.

Orchestrates all pipeline stages:
  1. Load & validate dataset
  2. Feature selection (SelectKBest, k=10)
  3. Train/test split (stratified 80/20)
  4. Preprocessing (StandardScaler)
  5. Quantum-SMOTE oversampling (or undersampling baseline)
  6. QSVM kernel computation + SVM fitting
  7. Evaluation & metric logging
  8. Model + result checkpoint save

Also supports classical SVM baseline training for ablation studies
(reproducing Tables I & II).

Paper: Section IV — Experimental Simulation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from qsvm_fraud.data.dataset import FraudDataset
from qsvm_fraud.data.transforms import FraudPreprocessor
from qsvm_fraud.models.quantum_smote import build_smote
from qsvm_fraud.models.qsvm import QSVM
from qsvm_fraud.evaluation.metrics import FraudMetrics

logger = logging.getLogger(__name__)


class QSVMTrainer:
    """
    End-to-end trainer for the QSVM fraud detection pipeline.

    Reads from a config dict (loaded from YAML) and runs the full
    pipeline described in Section IV of the paper.

    Args:
        config: Full config dict (from Config.load()).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_cfg = config["model"]
        self.smote_cfg = config["quantum_smote"]
        self.data_cfg = config["data"]
        self.eval_cfg = config["evaluation"]
        self.hw_cfg = config["hardware"]

        self.results_dir = Path(self.eval_cfg.get("results_dir", "results/"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._metrics = FraudMetrics()
        self._all_results: list[dict] = []

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full training + evaluation pipeline.

        Returns:
            Dict with paths to saved model and metrics file.
        """
        logger.info("=" * 60)
        logger.info("QSVM Fraud Detection — Training Run")
        logger.info("Config: n_qubits=%d, C=%.2f, smote_enabled=%s",
                    self.model_cfg["n_qubits"],
                    self.model_cfg.get("C", 1.0),
                    self.smote_cfg.get("enabled", True))
        logger.info("=" * 60)

        # 1. Load data
        X_raw, y_raw = self._load_data()

        # 2. Feature selection (KBest)
        dataset = FraudDataset(
            n_features=self.data_cfg["n_features"],
            score_func=self.data_cfg.get("score_func", "f_classif"),
            test_size=self.data_cfg.get("test_size", 0.2),
            random_state=self.hw_cfg.get("random_seed", 42),
            max_samples=self.data_cfg.get("max_train_samples"),
        )
        X_sel, feature_names = dataset.select_features(X_raw, y_raw)

        # Save selected features for reproducibility audit
        features_path = self.results_dir / "selected_features.json"
        features_path.write_text(json.dumps({"selected_features": feature_names}, indent=2))
        logger.info("Selected features saved to %s", features_path)

        # 3. Train/test split
        X_train, X_test, y_train, y_test = dataset.split(X_sel, y_raw)

        # 4. Preprocessing (StandardScaler fit on train only)
        preprocessor = FraudPreprocessor(
            scaler_type=self.data_cfg.get("scaler_type", "standard")
        )
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        # 5. Balance training data via Quantum-SMOTE (or undersampling)
        X_balanced, y_balanced = self.preprocess(X_train_scaled, y_train)

        # 6. Train QSVM
        model = self.train_qsvm(X_balanced, y_balanced)

        # 7. Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        y_score = None
        if self.model_cfg.get("probability", True):
            try:
                y_score = model.predict_proba(X_test_scaled)
            except Exception as e:
                logger.warning("predict_proba failed: %s — AUC will be None", e)

        n_qubits = self.model_cfg["n_qubits"]
        smote_label = "Quantum-SMOTE" if self.smote_cfg.get("enabled", True) else "Undersampling"
        result = self._metrics.compute(
            y_test, y_pred, y_score,
            label=f"QSVM-{n_qubits}qubit-{smote_label}",
        )
        self._metrics.print_report(result)
        self._all_results.append(result)

        # 8. Classical SVM baseline (for comparison)
        if self.eval_cfg.get("run_classical_baseline", True):
            self._run_classical_baseline(X_train_scaled, y_train, X_test_scaled, y_test)

        # 9. Save model and results
        model_path = self._save_model(model, n_qubits)
        metrics_path = self._save_metrics()

        # 10. Save plots
        if self.eval_cfg.get("save_confusion_matrix", True):
            self._metrics.save_confusion_matrix_plot(
                y_test, y_pred,
                str(self.results_dir / f"confusion_matrix_{n_qubits}qubit.png"),
                title=f"QSVM {n_qubits}-qubit Confusion Matrix",
            )
        if self.eval_cfg.get("save_roc_curve", True) and y_score is not None:
            self._metrics.save_roc_curve(
                y_test, y_score,
                str(self.results_dir / f"roc_curve_{n_qubits}qubit.png"),
                title=f"QSVM {n_qubits}-qubit ROC Curve",
            )

        logger.info("Training complete. Model: %s | Metrics: %s", model_path, metrics_path)
        return {"model_path": str(model_path), "metrics_path": str(metrics_path)}

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load raw CSV and validate."""
        csv_path = self.data_cfg["csv_path"]
        dataset = FraudDataset()
        return dataset.load(csv_path)

    def preprocess(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply Quantum-SMOTE oversampling to training data.

        Paper: Section II — "minority class (fraud) samples are generated by
        Quantum-SMOTE to increase its proportion."

        If smote is disabled, applies the undersampling strategy described in
        Section IV-B (first 15,000 samples; 293 fraud + 300 non-fraud control).

        Args:
            X_train: [N_train, D] scaled training features.
            y_train: [N_train] labels.

        Returns:
            X_balanced, y_balanced.
        """
        if self.smote_cfg.get("enabled", True):
            smote = build_smote(self.smote_cfg)
            logger.info("Applying %s...", type(smote).__name__)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        else:
            # Undersampling baseline (Table I comparison):
            # "selecting the first 15,000 samples ... randomly assigning 300 non-fraudulent"
            logger.info("Using undersampling baseline (Table I comparison config).")
            X_balanced, y_balanced = self._apply_undersampling(X_train, y_train)

        dist = dict(zip(*np.unique(y_balanced, return_counts=True)))
        logger.info("Post-balancing class distribution: %s", dist)
        return X_balanced, y_balanced

    def _apply_undersampling(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Undersampling baseline from paper Section IV-B.

        Paper: "selecting the first 15,000 samples as the basic training set
        containing only 293 fraudulent instances, while randomly assigning
        300 non-fraudulent samples to form a control training set."
        """
        n_majority = self.data_cfg.get("undersample_majority_n", 300)
        rng = np.random.RandomState(self.hw_cfg.get("random_seed", 42))

        X_min = X[y == 1]
        X_maj = X[y == 0]

        # Randomly sample majority
        idx = rng.choice(len(X_maj), size=min(n_majority, len(X_maj)), replace=False)
        X_maj_sub = X_maj[idx]
        y_maj_sub = np.zeros(len(X_maj_sub), dtype=y.dtype)
        y_min = np.ones(len(X_min), dtype=y.dtype)

        X_balanced = np.vstack([X_maj_sub, X_min])
        y_balanced = np.concatenate([y_maj_sub, y_min])
        return X_balanced, y_balanced

    def train_qsvm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> QSVM:
        """
        Instantiate and fit the QSVM model.

        Paper: Section III — QSVM training via quantum kernel + SVC solver.

        Args:
            X_train: [N_balanced, n_features] float64.
            y_train: [N_balanced] int.

        Returns:
            Fitted QSVM model.
        """
        model = QSVM(
            n_qubits=self.model_cfg["n_qubits"],
            reps=self.model_cfg.get("reps", 2),
            entanglement=self.model_cfg.get("entanglement", "full"),
            C=self.model_cfg.get("C", 1.0),
            backend=self.model_cfg.get("backend", "statevector_simulator"),
            cache_kernel=self.model_cfg.get("cache_kernel", True),
            probability=self.model_cfg.get("probability", True),
            random_state=self.hw_cfg.get("random_seed", 42),
        )
        logger.info("Fitting QSVM: %s", model)
        model.fit(X_train, y_train)
        return model

    def train_classical_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> SVC:
        """
        Train classical SVM baseline (Table II comparison).

        ASSUMPTION: RBF kernel (standard default). C=1.0 (ASSUMED).

        Args:
            X_train: [N_train, D] float64.
            y_train: [N_train] int.

        Returns:
            Fitted sklearn SVC.
        """
        svc = SVC(
            kernel="rbf",     # ASSUMED: paper does not specify classical SVM kernel
            C=self.model_cfg.get("C", 1.0),
            probability=True,
            random_state=self.hw_cfg.get("random_seed", 42),
        )
        svc.fit(X_train, y_train)
        return svc

    def _run_classical_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Train and evaluate classical SVM for comparison (Table II)."""
        logger.info("Training classical SVM baseline...")
        svc = self.train_classical_svm(X_train, y_train)
        y_pred_svm = svc.predict(X_test)
        y_score_svm = svc.predict_proba(X_test)
        n_features = self.data_cfg["n_features"]
        result = self._metrics.compute(
            y_test, y_pred_svm, y_score_svm,
            label=f"SVM-{n_features}feat-baseline",
        )
        self._metrics.print_report(result)
        self._all_results.append(result)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self, model: QSVM, n_qubits: int) -> Path:
        """Save QSVM model to checkpoints directory."""
        checkpoints_dir = Path("checkpoints/")
        checkpoints_dir.mkdir(exist_ok=True)
        smote_tag = "qsmote" if self.smote_cfg.get("enabled", True) else "undersample"
        model_path = checkpoints_dir / f"qsvm_{n_qubits}qubit_{smote_tag}.joblib"
        model.save(str(model_path))
        return model_path

    def _save_metrics(self) -> Path:
        """Save all results to JSON and print comparison table."""
        metrics_path = self.results_dir / "metrics.json"

        # Serialise (remove non-JSON-able numpy arrays)
        serialisable = []
        for r in self._all_results:
            r_copy = {k: v for k, v in r.items() if k not in ("confusion_matrix", "classification_report")}
            serialisable.append(r_copy)

        metrics_path.write_text(json.dumps(serialisable, indent=2))
        logger.info("Metrics saved to %s", metrics_path)

        # Print comparison table
        print(self._metrics.compare_table(self._all_results))
        return metrics_path
