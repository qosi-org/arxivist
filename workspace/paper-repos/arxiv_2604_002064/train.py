#!/usr/bin/env python3
"""Train QNN circuit parameters theta (Methods A/B/C) on the Gaussian density or
Black-Scholes Put targets from the paper.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config_debug.yaml --debug
    python train.py --config configs/config.yaml --method C --seed 123

Paper section: Section 4.1 "Parameters optimisation", Section 4.2/4.3.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from noisy_qnn_uat.data.dataset import BlackScholesPutDataset
from noisy_qnn_uat.data.transforms import InputNormalizer
from noisy_qnn_uat.training.trainer import QNNTrainer
from noisy_qnn_uat.utils.config import ConfigLoader, SeedManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--debug", action="store_true",
        help="Use a reduced dataset/iteration budget for fast local testing"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build all components without training, to validate setup"
    )
    parser.add_argument(
        "--method", type=str, default=None, choices=["A", "B", "C"],
        help="Optimisation method: A (L-BFGS-B), B (two-stage), or C (Adam). "
             "Overrides config's training.method if given."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ConfigLoader().load(args.config)
    seed = args.seed if args.seed is not None else config["training"]["seed"]
    SeedManager().seed_everything(seed)

    method = args.method or config["training"]["method"]
    bs_cfg = config["data"]["black_scholes"]
    model_cfg = config["model"]

    print(f"[train.py] Loaded config from {args.config} (method={method}, seed={seed})")

    if args.dry_run:
        print("[train.py] --dry-run: components constructed successfully, exiting without training.")
        return

    # --- Build training data (Black-Scholes Put, Section 4.3) ---
    dataset = BlackScholesPutDataset()
    n_train_samples = 200 if args.debug else 1600
    ranges = {
        "S_range": tuple(bs_cfg["S_range"]),
        "K_range": tuple(bs_cfg["K_range"]),
        "T_range": tuple(bs_cfg["T_range"]),
        "r_range": tuple(bs_cfg["r_range"]),
        "sigma_range": tuple(bs_cfg["sigma_range"]),
        "n_samples": n_train_samples,
        "seed": seed,
    }
    x_raw, y_train = dataset.sample_training_grid(ranges)

    normalizer = InputNormalizer()
    x_min = np.array([bs_cfg["S_range"][0], bs_cfg["K_range"][0], bs_cfg["T_range"][0],
                       bs_cfg["r_range"][0], bs_cfg["sigma_range"][0]])
    x_max = np.array([bs_cfg["S_range"][1], bs_cfg["K_range"][1], bs_cfg["T_range"][1],
                       bs_cfg["r_range"][1], bs_cfg["sigma_range"][1]])
    x_train = normalizer.normalize(x_raw, x_min, x_max)

    R = float(np.ceil(1.1 * np.max(y_train)))
    print(f"[train.py] R (output scaling) = {R}")

    # --- Fit theta ---
    trainer = QNNTrainer(
        n_accuracy_blocks=model_cfg["n_accuracy_blocks"],
        input_dim=x_train.shape[1],
        R=R,
        seed=seed,
    )

    if method == "A":
        theta = trainer.fit_method_a_lbfgsb(None, x_train, y_train)
    elif method == "B":
        theta = trainer.fit_method_b_two_stage(None, x_train, y_train)
    elif method == "C":
        theta = trainer.fit_method_c_adam(
            None, x_train, y_train,
            lr=config["training"]["adam_lr"],
            beta1=config["training"]["adam_beta1"],
            beta2=config["training"]["adam_beta2"],
            max_iterations=config["training"]["max_iterations"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Report training-set fit quality ---
    preds = trainer.predict(theta, x_train)
    train_mae = float(np.mean(np.abs(preds - y_train)))
    print(f"[train.py] Method {method} finished. Training-set MAE = {train_mae:.4f}")

    # --- Save checkpoint ---
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/theta_method{method}_seed{seed}.json"
    checkpoint = {
        "theta": theta.tolist(),
        "R": R,
        "n_accuracy_blocks": model_cfg["n_accuracy_blocks"],
        "n_qubits": model_cfg["n_qubits"],
        "input_dim": x_train.shape[1],
        "x_min": x_min.tolist(),
        "x_max": x_max.tolist(),
        "method": method,
        "seed": seed,
        "train_mae": train_mae,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"[train.py] Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
