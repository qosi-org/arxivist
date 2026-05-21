#!/usr/bin/env python3
"""
evaluate.py — Load a trained QSVM and evaluate on held-out test set.

Usage:
    python evaluate.py --model-path checkpoints/qsvm_10qubit_qsmote.joblib \\
                       --csv data/raw/creditcard.csv

    # Full ablation study (reproduces Tables I & II):
    python evaluate.py --model-path checkpoints/qsvm_10qubit_qsmote.joblib \\
                       --csv data/raw/creditcard.csv --ablation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qsvm_fraud.utils.config import Config
from qsvm_fraud.data.dataset import FraudDataset
from qsvm_fraud.data.transforms import FraudPreprocessor
from qsvm_fraud.models.qsvm import QSVM
from qsvm_fraud.evaluation.metrics import FraudMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained QSVM model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved .joblib QSVM model")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to creditcard.csv")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file for data preprocessing settings")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation across 4/8/10 qubits + classical SVM")
    parser.add_argument("--save-plots", action="store_true", default=True,
                        help="Save confusion matrix and ROC curve PNGs")
    parser.add_argument("--results-dir", type=str, default="results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)
    Config.set_seed(config["hardware"]["random_seed"])
    Config.setup_logging()

    # Load and preprocess data
    dataset = FraudDataset(
        n_features=config["data"]["n_features"],
        score_func=config["data"].get("score_func", "f_classif"),
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["hardware"]["random_seed"],
    )
    X_raw, y_raw = dataset.load(args.csv)
    X_sel, feat_names = dataset.select_features(X_raw, y_raw)
    _, X_test, _, y_test = dataset.split(X_sel, y_raw)

    preprocessor = FraudPreprocessor()
    preprocessor.fit(X_sel[: int(len(X_sel) * 0.8)])  # refit on train portion
    X_test_scaled = preprocessor.transform(X_test)

    # Load model
    model = QSVM.load(args.model_path)
    print(f"Loaded model: {model}")

    # Evaluate
    metrics = FraudMetrics()
    y_pred = model.predict(X_test_scaled)
    y_score = model.predict_proba(X_test_scaled) if model.probability else None
    result = metrics.compute(y_test, y_pred, y_score, label=Path(args.model_path).stem)
    metrics.print_report(result)

    if args.save_plots:
        out = Path(args.results_dir)
        out.mkdir(exist_ok=True)
        tag = Path(args.model_path).stem
        metrics.save_confusion_matrix_plot(y_test, y_pred, str(out / f"cm_{tag}.png"))
        if y_score is not None:
            metrics.save_roc_curve(y_test, y_score, str(out / f"roc_{tag}.png"))


if __name__ == "__main__":
    main()
