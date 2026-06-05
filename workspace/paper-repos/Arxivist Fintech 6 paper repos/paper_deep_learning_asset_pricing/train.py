"""
train.py — Training entry point for the GAN asset pricing model.

Implements ensemble training with 9 models (Section III.E):
  "A distinguishing feature of neural networks is that the estimation results
   can depend on the starting value used in the optimization. The standard
   practice is to train the models separately with different initial values."

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug          # smoke test
    python train.py --config configs/config.yaml --ensemble-idx 3 # single model

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019).
"""

import argparse
import json
import os
from pathlib import Path

import torch

from dlap.utils.config import load_config, set_seed, get_device
from dlap.models.gan_model import GANAssetPricingModel
from dlap.training.trainer import GANTrainer
from dlap.data.dataset import make_synthetic_dataset


def train_single(cfg: dict, seed: int, ensemble_idx: int, device: torch.device, debug: bool):
    """Train a single ensemble member."""
    set_seed(seed, deterministic=cfg["hardware"].get("deterministic", False))

    print(f"\n{'='*60}")
    print(f"  Ensemble member {ensemble_idx} | seed={seed}")
    print(f"{'='*60}")

    # Build model
    model = GANAssetPricingModel(cfg)

    # Load data
    # NOTE: Replace with real data loader when CRSP data is available.
    print("[WARNING] Using synthetic data. Replace with real CRSP data for paper results.")
    dataset = make_synthetic_dataset(
        T=240 if debug else 600,
        N=100 if debug else 500,
        device=device,
    )
    train_T = 160 if debug else 250
    valid_T = 50 if debug else 100

    macro, chars, returns, panel_weights = dataset.get_all()
    train_data = (macro[:train_T], chars[:train_T], returns[:train_T])
    valid_data = (macro[train_T:train_T + valid_T], chars[train_T:train_T + valid_T], returns[train_T:train_T + valid_T])

    # Override checkpoint dir per ensemble member
    cfg_copy = {**cfg}
    cfg_copy["paths"] = {**cfg["paths"], "best_model_name": f"best_model_ensemble_{ensemble_idx}.pt"}

    trainer = GANTrainer(model, cfg_copy, device)
    history = trainer.fit(
        train_data=train_data,
        valid_data=valid_data,
        max_epochs=2 if debug else cfg["training"].get("max_epochs", 200),
        patience=cfg["training"].get("early_stopping_patience", 20),
        debug=debug,
    )

    # Save training history
    history_path = Path(cfg["paths"]["results_dir"]) / f"history_ensemble_{ensemble_idx}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return trainer.best_valid_sr


def main():
    parser = argparse.ArgumentParser(description="Train GAN Asset Pricing Model")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--debug", action="store_true", help="Smoke test: 2 epochs, small data")
    parser.add_argument("--ensemble-idx", type=int, default=None, help="Train single ensemble member")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["hardware"]["device"])

    print(f"\nDeep Learning in Asset Pricing — Training")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    num_ensemble = cfg["training"]["num_ensemble_models"]
    base_seed = cfg["training"].get("seed", 42)

    if args.ensemble_idx is not None:
        # Train a single ensemble member
        seed = base_seed + args.ensemble_idx
        sr = train_single(cfg, seed, args.ensemble_idx, device, args.debug)
        print(f"\nEnsemble {args.ensemble_idx} best valid SR: {sr:.4f}")
    else:
        # Train all ensemble members
        print(f"\nTraining {num_ensemble} ensemble members...")
        best_srs = []
        for idx in range(num_ensemble):
            seed = base_seed + idx
            sr = train_single(cfg, seed, idx, device, args.debug)
            best_srs.append(sr)
            print(f"Ensemble {idx}: best valid SR = {sr:.4f}")

        print(f"\n{'='*60}")
        print(f"Ensemble training complete.")
        print(f"Valid SR per model: {[f'{sr:.4f}' for sr in best_srs]}")
        print(f"Mean valid SR: {sum(best_srs)/len(best_srs):.4f}")
        print(f"Best valid SR: {max(best_srs):.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
