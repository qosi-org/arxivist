#!/usr/bin/env python3
"""
train.py — Main training entrypoint for QSVM fraud detection.

Reproduces the results in Table I & II of:
  "Quantum Support Vector Machine for Fraud Detection"
  Ren & Zhang, IEEE CCPQT 2025

Usage:
    # Primary run (10-qubit QSVM + Quantum-SMOTE) — reproduces Table I primary row
    python train.py --config configs/config.yaml

    # Debug run (4-qubit, 300 samples — completes in ~2 min)
    python train.py --config configs/config_debug.yaml

    # Undersampling baseline (reproduces Table I undersampling row)
    python train.py --config configs/config.yaml --no-smote

    # 8-qubit ablation (reproduces Table II 8-qubit QSVM row)
    python train.py --config configs/config_8qubit.yaml

Expected results (primary config — SIR confidence: 0.77):
    QSVM-10qubit + Quantum-SMOTE: accuracy=98.8%, F1=0.962, recall=0.945, AUC=0.992
    SVM-10feat baseline:           accuracy=95.1%, F1=0.933, recall=0.923, AUC=0.979
"""

import argparse
import sys
from pathlib import Path

# Add src to path for editable install fallback
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qsvm_fraud.utils.config import Config
from qsvm_fraud.training.trainer import QSVMTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QSVM for credit card fraud detection (Ren & Zhang 2025)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Override data.csv_path from config",
    )
    parser.add_argument(
        "--n-qubits", type=int, default=None,
        choices=[4, 8, 10],
        help="Override model.n_qubits (must match data.n_features)",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable Quantum-SMOTE; use undersampling baseline instead",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="Override model.backend (e.g. statevector_simulator, aer_simulator)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override hardware.random_seed",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override evaluation.results_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = Config.load(args.config)

    # Apply CLI overrides
    if args.csv:
        config["data"]["csv_path"] = args.csv
    if args.n_qubits:
        config["model"]["n_qubits"] = args.n_qubits
        config["data"]["n_features"] = args.n_qubits
    if args.no_smote:
        config["quantum_smote"]["enabled"] = False
    if args.backend:
        config["model"]["backend"] = args.backend
    if args.seed is not None:
        config["hardware"]["random_seed"] = args.seed
    if args.output_dir:
        config["evaluation"]["results_dir"] = args.output_dir

    # Set seeds
    Config.set_seed(config["hardware"]["random_seed"])

    # Setup logging
    log_cfg = config.get("logging", {})
    Config.setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file"),
    )

    # Run training pipeline
    trainer = QSVMTrainer(config)
    results = trainer.run()

    print(f"\nTraining complete.")
    print(f"  Model saved to:   {results['model_path']}")
    print(f"  Metrics saved to: {results['metrics_path']}")


if __name__ == "__main__":
    main()
