#!/usr/bin/env python3
"""
scripts/run_ablation.py — Reproduce Tables I and II from the paper.

Table I: QSVM-10qubit + Quantum-SMOTE vs. Undersampling
Table II: QSVM vs. SVM across 4, 8, 10 features/qubits

Runtime warning: Full ablation (3 QSVM configs + 3 SVM configs) is very
expensive on a statevector simulator. Each QSVM kernel matrix computation
is O(N^2). Recommend running with --max-samples 500 for an initial test.

Usage:
    python scripts/run_ablation.py --csv data/raw/creditcard.csv
    python scripts/run_ablation.py --csv data/raw/creditcard.csv --max-samples 500
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsvm_fraud.utils.config import Config
from qsvm_fraud.data.dataset import FraudDataset
from qsvm_fraud.data.transforms import FraudPreprocessor
from qsvm_fraud.models.quantum_smote import build_smote, ClassicalSMOTE
from qsvm_fraud.models.qsvm import QSVM
from qsvm_fraud.evaluation.metrics import FraudMetrics
from sklearn.svm import SVC

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full paper ablation study")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv", required=True, help="Path to creditcard.csv")
    parser.add_argument("--output-dir", default="results/ablation/")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Subsample training data for speed (e.g. 500). Full run is very slow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)
    Config.set_seed(config["hardware"]["random_seed"])
    Config.setup_logging()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_tool = FraudMetrics()
    all_results = []

    print("\n" + "=" * 70)
    print("  ABLATION STUDY — Reproducing Tables I & II")
    print("  Paper: Quantum SVM for Fraud Detection (Ren & Zhang 2025)")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Shared data loading (with max_samples for debug)
    # ---------------------------------------------------------------
    dataset = FraudDataset(
        n_features=10,  # max features needed
        score_func=config["data"].get("score_func", "f_classif"),
        test_size=0.2,
        random_state=42,
        max_samples=args.max_samples,
    )
    X_raw, y_raw = dataset.load(args.csv)
    X_sel10, feat_names10 = dataset.select_features(X_raw, y_raw)
    X_train10, X_test10, y_train, y_test = dataset.split(X_sel10, y_raw)

    preprocessor = FraudPreprocessor()
    X_train10_sc = preprocessor.fit_transform(X_train10)
    X_test10_sc = preprocessor.transform(X_test10)

    smote_cfg = config["quantum_smote"]

    # ---------------------------------------------------------------
    # TABLE I — SMOTE vs. Undersampling (10-qubit QSVM)
    # ---------------------------------------------------------------
    print("\n--- TABLE I: Quantum-SMOTE vs. Undersampling (10-qubit QSVM) ---\n")

    for smote_enabled, label in [(True, "Quantum-SMOTE"), (False, "Undersampling")]:
        smote_cfg_copy = dict(smote_cfg)
        smote_cfg_copy["enabled"] = smote_enabled
        smote = build_smote(smote_cfg_copy) if smote_enabled else ClassicalSMOTE(random_state=42)

        X_bal, y_bal = smote.fit_resample(X_train10_sc, y_train)
        model = QSVM(n_qubits=10, C=config["model"].get("C", 1.0), cache_kernel=True)
        model.fit(X_bal, y_bal)

        y_pred = model.predict(X_test10_sc)
        y_score = model.predict_proba(X_test10_sc) if model.probability else None
        result = metrics_tool.compute(y_test, y_pred, y_score, label=f"QSVM-10qubit-{label}")
        metrics_tool.print_report(result)
        all_results.append(result)

    # ---------------------------------------------------------------
    # TABLE II — QSVM vs. SVM across 4, 8, 10 features
    # ---------------------------------------------------------------
    print("\n--- TABLE II: QSVM vs. SVM across feature dimensions ---\n")

    for n_q in [4, 8, 10]:
        # Re-select top-n_q features
        ds_nq = FraudDataset(
            n_features=n_q, score_func="f_classif", test_size=0.2,
            random_state=42, max_samples=args.max_samples,
        )
        X_raw2, y_raw2 = ds_nq.load(args.csv)
        X_sel, _ = ds_nq.select_features(X_raw2, y_raw2)
        X_tr, X_te, y_tr, y_te = ds_nq.split(X_sel, y_raw2)
        pre = FraudPreprocessor()
        X_tr_sc = pre.fit_transform(X_tr)
        X_te_sc = pre.transform(X_te)

        # Apply Quantum-SMOTE on training minority
        smote_cfg["enabled"] = True
        smote = build_smote(smote_cfg)
        X_bal, y_bal = smote.fit_resample(X_tr_sc, y_tr)

        # QSVM
        qsvm = QSVM(n_qubits=n_q, C=config["model"].get("C", 1.0), cache_kernel=True)
        qsvm.fit(X_bal, y_bal)
        y_pred_q = qsvm.predict(X_te_sc)
        y_score_q = qsvm.predict_proba(X_te_sc) if qsvm.probability else None
        r_qsvm = metrics_tool.compute(y_te, y_pred_q, y_score_q, label=f"QSVM-{n_q}qubit")
        metrics_tool.print_report(r_qsvm)
        all_results.append(r_qsvm)

        # Classical SVM
        svc = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        svc.fit(X_tr_sc, y_tr)
        y_pred_s = svc.predict(X_te_sc)
        y_score_s = svc.predict_proba(X_te_sc)
        r_svm = metrics_tool.compute(y_te, y_pred_s, y_score_s, label=f"SVM-{n_q}feat")
        metrics_tool.print_report(r_svm)
        all_results.append(r_svm)

    # ---------------------------------------------------------------
    # Summary + save
    # ---------------------------------------------------------------
    print("\n" + metrics_tool.compare_table(all_results))

    serialisable = [
        {k: v for k, v in r.items() if k not in ("confusion_matrix", "classification_report")}
        for r in all_results
    ]
    results_path = out_dir / "ablation_results.json"
    results_path.write_text(json.dumps(serialisable, indent=2))
    print(f"\nAblation results saved to {results_path}")


if __name__ == "__main__":
    main()
