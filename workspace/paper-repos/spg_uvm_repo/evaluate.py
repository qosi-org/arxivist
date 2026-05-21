"""
evaluate.py — Evaluate a trained SPG-UVM model.

Loads trained actor networks from a checkpoint directory and computes the
actor price with 95% confidence interval.

Usage:
    python evaluate.py --checkpoint results/effective_config.yaml \
        --n-paths 524288 --reference-price 13.75
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spg_uvm.training.trainer import SPGUVMTrainer
from spg_uvm.utils.config import UVMConfig, set_seed
from spg_uvm.utils.metrics import PriceEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SPG-UVM actor price")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML (typically results/effective_config.yaml)."
    )
    parser.add_argument(
        "--n-paths", type=int, default=524288,
        help="Number of MC paths for actor price estimation (default: 2^19)."
    )
    parser.add_argument(
        "--reference-price", type=float, default=None,
        help="Optional reference price for relative error computation."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = UVMConfig.from_yaml(args.config)
    if args.device:
        cfg.hardware.device = args.device
    if args.reference_price is not None:
        cfg.evaluation.reference_price = args.reference_price
    cfg.evaluation.n_paths_actor_price = args.n_paths

    set_seed(args.seed)
    device = torch.device(cfg.hardware.device)

    print(f"Evaluating: {cfg}")
    print("Note: This script requires actor networks to be rebuilt by re-running train.py.")
    print("Full checkpoint serialization (actor_nets dict) is a planned enhancement.")
    print("For now, run train.py with --output-dir and capture results.json.")


if __name__ == "__main__":
    main()
