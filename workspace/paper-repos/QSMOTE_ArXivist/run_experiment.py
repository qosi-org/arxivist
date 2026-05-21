"""Main experiment runner for Quantum-SMOTE.

Pipeline:
    preprocess -> cluster -> optional Quantum-SMOTE -> train -> evaluate -> report
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from quantum_smote.data.dataset import TelcoChurnDataset
from quantum_smote.data.preprocessor import TelcoChurnPreprocessor
from quantum_smote.clustering.kmeans_clusterer import KMeansClusterer
from quantum_smote.smote.quantum_smote import QuantumSMOTE
from quantum_smote.evaluation.classifier import ClassifierFactory
from quantum_smote.utils.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum-SMOTE full experiment runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--target-pct", type=int, default=None, help="Override target minority pct (30/40/50)")
    parser.add_argument("--split-factor", type=int, default=None, help="Override split_factor")
    parser.add_argument("--model", type=str, choices=["rf", "lr", "both"], default="both")
    parser.add_argument("--no-smote", action="store_true", help="Run baseline without SMOTE")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory for outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def _override_config(cfg: Config, args: argparse.Namespace) -> Config:
    data = cfg.to_dict()
    if args.target_pct is not None:
        data["quantum_smote"]["target_pct"] = int(args.target_pct)
    if args.split_factor is not None:
        data["quantum_smote"]["split_factor"] = int(args.split_factor)
    data["data"]["random_state"] = int(args.seed)
    data["clustering"]["random_state"] = int(args.seed)
    data["classifiers"]["random_forest"]["random_state"] = int(args.seed)
    data["classifiers"]["logistic_regression"]["random_state"] = int(args.seed)
    return Config(
        raw_csv_path=data["data"]["raw_csv_path"],
        target_column=data["data"]["target_column"],
        drop_columns=data["data"].get("drop_columns", []),
        test_size=float(data["data"].get("test_size", 0.2)),
        random_state=int(data["data"].get("random_state", args.seed)),
        clustering=data["clustering"],
        quantum_smote=data["quantum_smote"],
        classifiers=data["classifiers"],
        evaluation=data["evaluation"],
        output=data["output"],
        logging=data.get("logging", {}),
    )


def _select_models(model_choice: str):
    if model_choice == "both":
        return ["rf", "lr"]
    return [model_choice]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = _override_config(Config.from_yaml(args.config), args)

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    raw_df = TelcoChurnDataset.load(cfg.raw_csv_path)
    preprocessor = TelcoChurnPreprocessor(
        drop_columns=cfg.to_dict()["data"].get("drop_columns", ["customerID"]),
        target_column=cfg.target_column,
        correlation_threshold=cfg.to_dict()["data"].get("correlation_threshold", 0.9),
    )
    X, y = preprocessor.fit_transform(raw_df)

    # Cluster full feature matrix
    cluster_cfg = cfg.clustering
    kmeans = KMeansClusterer(
        n_clusters=int(cluster_cfg.get("n_clusters", 3)),
        init=cluster_cfg.get("init", "k-means++"),
        n_init=int(cluster_cfg.get("n_init", 10)),
        random_state=cluster_cfg.get("random_state", args.seed),
    ).fit(X)
    labels = kmeans.get_labels()
    centroids = kmeans.get_centroids()

    # Optional synthetic augmentation
    if args.no_smote:
        X_aug, y_aug = X, y
    else:
        qs_cfg = cfg.quantum_smote
        smote = QuantumSMOTE(
            target_pct=int(qs_cfg.get("target_pct", 50)),
            split_factor=int(qs_cfg.get("split_factor", 5)),
            rotation_axis=qs_cfg.get("rotation_axis", "X"),
            angle_increment=float(qs_cfg.get("angle_increment", 0.0174533)),
            use_statevector=bool(qs_cfg.get("use_statevector", True)),
            shots=int(qs_cfg.get("shots", 1024)),
            statevector_extraction_strategy=qs_cfg.get("statevector_extraction_strategy", "first_F"),
        )
        X_aug, y_aug = smote.fit_resample(X, y, labels, centroids)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug,
        y_aug,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        stratify=y_aug,
    )

    # Train and evaluate models
    models = _select_models(args.model)
    results = []
    for model_name in models:
        model = ClassifierFactory.build(model_name, cfg.classifiers)
        metrics = ClassifierFactory.train_evaluate(model, X_train, y_train, X_test, y_test)
        metrics["model"] = model_name
        metrics["condition"] = "no_smote" if args.no_smote else f"smote_{cfg.quantum_smote.get('target_pct', 50)}"
        results.append(metrics)

        # Console report
        print(f"\n[{model_name.upper()}] {metrics['condition']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

    # Persist results
    metrics_df = pd.DataFrame(
        [
            {
                "model": item.get("model"),
                "condition": item.get("condition"),
                "accuracy": item.get("accuracy"),
                "f1": item.get("f1"),
                "pr_auc": item.get("pr_auc"),
                "roc_auc": item.get("roc_auc"),
                "confusion_matrix": item.get("confusion_matrix").tolist(),
            }
            for item in results
        ]
    )
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2), encoding="utf-8")

    if not args.no_smote:
        synthetic = X_aug[len(X) :]
        np.savetxt(output_dir / "synthetic_data.csv", synthetic, delimiter=",")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
