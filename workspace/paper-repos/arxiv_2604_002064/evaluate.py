#!/usr/bin/env python3
"""Evaluate a trained QNN checkpoint against the Black-Scholes Put analytical target.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/theta_methodA_seed42.json
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/theta_methodA_seed42.json --noise depolarising

Paper sections: Section 4.3 (noiseless evaluation), Section 4.4 (depolarising-noise
simulation), Section 2.3.3 (theoretical error bound example, Black-Scholes Put).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from noisy_qnn_uat.data.dataset import BlackScholesPutDataset
from noisy_qnn_uat.evaluation.hardware_bounds import ErrorBoundCalculator
from noisy_qnn_uat.evaluation.metrics import ErrorMetrics
from noisy_qnn_uat.models.noise_channels import DepolarisingChannel, HardwareNoiseCalibrator
from noisy_qnn_uat.training.trainer import QNNTrainer
from noisy_qnn_uat.utils.config import ConfigLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained theta checkpoint")
    parser.add_argument(
        "--noise", type=str, default="none", choices=["none", "depolarising", "comprehensive"],
        help="Noise model to apply during evaluation (default: none)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigLoader().load(args.config)

    with open(args.checkpoint, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    theta = np.array(ckpt["theta"])
    R = ckpt["R"]
    n_accuracy_blocks = ckpt["n_accuracy_blocks"]
    n_qubits = ckpt["n_qubits"]
    x_min = np.array(ckpt["x_min"])
    x_max = np.array(ckpt["x_max"])

    print(f"[evaluate.py] Loaded checkpoint {args.checkpoint} "
          f"(method={ckpt['method']}, seed={ckpt['seed']}, train_mae={ckpt['train_mae']:.4f})")

    # --- Build the 40x40 evaluation grid (Section 4.3) ---
    bs_cfg = config["data"]["black_scholes"]
    dataset = BlackScholesPutDataset()
    x_raw, y_true = dataset.sample_eval_grid(
        k_range=tuple(bs_cfg["eval_K_range"]),
        sigma_sqrt_t_range=tuple(bs_cfg["eval_sigma_sqrt_T_range"]),
        grid_size=bs_cfg["eval_grid_size"],
        s0=bs_cfg["eval_S0"],
        r=bs_cfg["eval_r"],
    )
    x_norm = (x_raw - x_min) / (x_max - x_min)
    x_norm = np.clip(x_norm, 0.0, 1.0)

    trainer = QNNTrainer(n_accuracy_blocks=n_accuracy_blocks, input_dim=x_norm.shape[1], R=R)
    preds = trainer.predict(theta, x_norm)

    # --- Apply noise model to predictions, if requested ---
    if args.noise != "none":
        hw_cfg = config["hardware"]
        calibrator = HardwareNoiseCalibrator()
        lambda_v = calibrator.compute_lambda_V(hw_cfg["eps_1Q"], n_qubits)
        _, n2q_ucr = calibrator.naive_and_ucr_two_qubit_gate_counts(n_accuracy_blocks, n_qubits)
        lambda_u = calibrator.compute_lambda_U(
            hw_cfg["eps_2Q"], n2q_ucr, hw_cfg["T1_us"], hw_cfg["T2_us"], hw_cfg["t2Q_ns"]
        )
        alpha = calibrator.compute_alpha(lambda_v, lambda_u)
        print(f"[evaluate.py] Noise model='{args.noise}': lambda_V={lambda_v:.6f}, "
              f"lambda_U={lambda_u:.6f}, alpha={alpha:.4f}")

        # Contract predictions toward the noise bias (Corollary 3.13), applied at the
        # level of the scalar QNN output (equivalent to applying it to P1+P2 directly).
        offset_term = 1.0 - (4.0 * n_accuracy_blocks) / (2 ** n_qubits)
        preds = alpha * preds + R * (1.0 - alpha) * offset_term

    # --- Metrics ---
    metrics = ErrorMetrics()
    rmse = metrics.rmse(preds, y_true)
    mae = metrics.mae(preds, y_true)
    max_err = metrics.max_error(preds, y_true)

    # --- Theoretical bound (Example 2.3.3, truncated-Put worked example constant) ---
    K, K_lower, sigma, T = 1.0, 0.4, float(np.mean(bs_cfg["sigma_range"])), float(np.mean(bs_cfg["T_range"]))
    l1_fhat_example = 2.316  # Section 2.3.3 worked example value, B for K=1,K_=0.4,sigma=0.2,T=1
    bound_calc = ErrorBoundCalculator()
    theoretical_bound = bound_calc.noiseless_bound(l1_fhat_example, n_accuracy_blocks)
    ratio = metrics.bound_ratio(mae, theoretical_bound)

    print(f"[evaluate.py] noise={args.noise}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"MaxError={max_err:.4f}  Theoretical bound={theoretical_bound:.4f}  "
          f"MAE/bound ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
