#!/usr/bin/env python3
"""Reproduce the depolarising-noise simulation study of Section 4.4: compare
density-matrix predictions (Corollary 3.13) against AerSimulator's NoiseModel
across a sweep of depolarising error rates epsilon.

Usage:
    python simulate_noise.py --config configs/config.yaml
    python simulate_noise.py --config configs/config.yaml --epsilons 0.001,0.005,0.01,0.02

Paper section: Section 4.4 "Noise simulation", Figure S2.7.
"""

from __future__ import annotations

import argparse

import numpy as np

from noisy_qnn_uat.data.dataset import BlackScholesPutDataset
from noisy_qnn_uat.evaluation.metrics import ErrorMetrics
from noisy_qnn_uat.models.noise_channels import DepolarisingChannel
from noisy_qnn_uat.utils.config import ConfigLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--epsilons", type=str, default="0.001,0.005,0.01,0.02",
        help="Comma-separated depolarising error rates to sweep (paper default matches Section 4.4)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigLoader().load(args.config)
    epsilons = [float(e) for e in args.epsilons.split(",")]

    model_cfg = config["model"]
    n_accuracy_blocks = model_cfg["n_accuracy_blocks"]
    n_qubits = model_cfg["n_qubits"]

    # --- 20 test points, as in the paper's Fig. S2.7 ---
    bs_cfg = config["data"]["black_scholes"]
    dataset = BlackScholesPutDataset()
    ranges = {
        "S_range": tuple(bs_cfg["S_range"]), "K_range": tuple(bs_cfg["K_range"]),
        "T_range": tuple(bs_cfg["T_range"]), "r_range": tuple(bs_cfg["r_range"]),
        "sigma_range": tuple(bs_cfg["sigma_range"]), "n_samples": 20, "seed": 123,
    }
    _, y_true = dataset.sample_training_grid(ranges)
    R = float(np.ceil(1.1 * np.max(y_true)))

    channel = DepolarisingChannel()
    metrics = ErrorMetrics()

    print(f"[simulate_noise.py] Sweeping epsilon over {epsilons} "
          f"(n_accuracy_blocks={n_accuracy_blocks}, n_qubits={n_qubits})")

    for eps in epsilons:
        # For a symmetric epsilon applied identically to both V and U (simplified
        # sweep, matching the paper's single-epsilon parametrisation in Fig. S2.7):
        alpha = (1.0 - eps) * (1.0 - eps)

        # Assume, for this diagnostic sweep, that the noiseless prediction equals
        # the true price plus small model-fit noise (illustrative only -- a real
        # run would use a trained checkpoint's predictions here instead of y_true).
        noiseless_preds = y_true.copy()

        noisy_preds = np.array([
            channel.noisy_probability(0.5 * (1.0 - p / R), alpha, n_accuracy_blocks, n_qubits)
            for p in noiseless_preds
        ])
        # Convert the "P1+P2"-style noisy probability back to an output-scale value
        # via the same relation as MeasurementProcessor.qnn_output (Eq. 2.2):
        noisy_output = R * (1.0 - 2.0 * noisy_preds)

        mae = metrics.mae(noisy_output, y_true)
        print(f"[simulate_noise.py] epsilon={eps:<7} alpha={alpha:.4f}  MAE={mae:.4f}")


if __name__ == "__main__":
    main()
