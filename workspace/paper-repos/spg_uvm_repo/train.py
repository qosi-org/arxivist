"""
train.py — Main training entrypoint for SPG-UVM.

Usage:
    python train.py --config configs/default.yaml --policy continuous --d 2 \
        --payoff geo_outperformer --device cuda --seed 42

Reference: Algorithm 1, Section 4, arXiv:2605.06670.
"""
import argparse
import json
import os
import sys
import time

import torch

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spg_uvm.training.trainer import SPGUVMTrainer
from spg_uvm.utils.config import UVMConfig, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="SPG-UVM: Stochastic Policy Gradient for Uncertain Volatility Model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--policy", type=str, choices=["continuous", "bangbang"],
        help="Override policy type from config."
    )
    parser.add_argument(
        "--d", type=int, help="Override number of assets from config."
    )
    parser.add_argument(
        "--payoff", type=str,
        choices=["geo_outperformer", "outperformer_spread", "best_of_butterfly",
                 "geo_call_spread", "call_sharpe"],
        help="Override payoff function from config."
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"],
        help="Override device from config."
    )
    parser.add_argument(
        "--seed", type=int, help="Override random seed."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for checkpoints and results."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: reduce M to 256 and epochs to 5 for quick testing."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build all components and validate setup without training."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = UVMConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.policy:
        cfg.model.policy_type = args.policy
    if args.d:
        cfg.model.d = args.d
        cfg.uvm_params.sigma_min = cfg.uvm_params.sigma_min[:1] * args.d
        cfg.uvm_params.sigma_max = cfg.uvm_params.sigma_max[:1] * args.d
    if args.payoff:
        cfg.payoff.name = args.payoff
    if args.device:
        cfg.hardware.device = args.device
    if args.seed:
        cfg.hardware.seed = args.seed
    if args.output_dir:
        cfg.output.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        cfg.output.results_dir = os.path.join(args.output_dir, "results")

    # Debug mode: fast local testing
    if args.debug:
        cfg.training.M = 256
        cfg.training.minibatch_size = 64
        cfg.training.E_first = 5
        cfg.training.E_subsequent = 2
        cfg.evaluation.n_paths_actor_price = 1024
        print("[DEBUG] Reduced M=256, E_first=5, E_subsequent=2")

    # Set seed
    set_seed(cfg.hardware.seed, cfg.hardware.deterministic)

    device = torch.device(cfg.hardware.device)

    # Re-validate config after overrides
    cfg._validate()

    print(f"Config: {cfg}")
    print(f"Device: {device}")
    print(f"Seed: {cfg.hardware.seed}")

    # Dry run: validate setup only
    if args.dry_run:
        print("\n[DRY RUN] Building components...")
        from spg_uvm.models.networks import ActorNetwork, CriticNetwork
        from spg_uvm.models.policy import ContinuousPolicy, BangBangPolicy
        from spg_uvm.payoffs import build_payoff
        actor = ActorNetwork(cfg.model.d, cfg.model.hidden_units, cfg.model.policy_type).to(device)
        critic = CriticNetwork(cfg.model.d, cfg.model.hidden_units).to(device)
        payoff = build_payoff(cfg.payoff.name, cfg.model.d, cfg.payoff.K1, cfg.payoff.K2).to(device)
        n_params = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())
        print(f"  Actor:  {actor}")
        print(f"  Critic: {critic}")
        print(f"  Payoff: {payoff}")
        print(f"  Total parameters per step: {n_params:,}")
        print("[DRY RUN] All components validated successfully. Exiting.")
        return

    # Create output directories
    os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output.results_dir, exist_ok=True)

    # Save effective config
    cfg_save_path = os.path.join(cfg.output.results_dir, "effective_config.yaml")
    cfg.to_yaml(cfg_save_path)
    print(f"Config saved to: {cfg_save_path}")

    # Train
    trainer = SPGUVMTrainer(cfg, device)
    results = trainer.train()

    # Save results
    results_path = os.path.join(cfg.output.results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
