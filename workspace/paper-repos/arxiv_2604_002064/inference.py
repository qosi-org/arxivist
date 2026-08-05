#!/usr/bin/env python3
"""Single-sample QNN inference: price one option given S,K,T,r,sigma.

Usage:
    python inference.py --config configs/config.yaml \
        --checkpoint checkpoints/theta_methodA_seed42.json \
        --input 100,100,1.0,0.03,0.2

Paper section: Eq. (2.2) (QNN scalar output), Section 4.1 (input normalisation).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from noisy_qnn_uat.training.trainer import QNNTrainer
from noisy_qnn_uat.utils.config import ConfigLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained theta checkpoint")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Comma-separated raw input values, in order S,K,T,r,sigma (e.g. '100,100,1.0,0.03,0.2')"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ConfigLoader().load(args.config)  # validated but not otherwise needed here

    with open(args.checkpoint, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    theta = np.array(ckpt["theta"])
    R = ckpt["R"]
    n_accuracy_blocks = ckpt["n_accuracy_blocks"]
    x_min = np.array(ckpt["x_min"])
    x_max = np.array(ckpt["x_max"])

    raw_values = np.array([float(v) for v in args.input.split(",")])
    assert raw_values.shape[0] == x_min.shape[0], (
        f"Expected {x_min.shape[0]} input values (S,K,T,r,sigma), got {raw_values.shape[0]}"
    )

    x_norm = np.clip((raw_values - x_min) / (x_max - x_min), 0.0, 1.0)

    trainer = QNNTrainer(n_accuracy_blocks=n_accuracy_blocks, input_dim=x_norm.shape[0], R=R)
    price = trainer.predict(theta, x_norm.reshape(1, -1))[0]

    print(f"[inference.py] Input (S,K,T,r,sigma) = {tuple(raw_values)}")
    print(f"[inference.py] Predicted Put price = {price:.4f}")


if __name__ == "__main__":
    main()
