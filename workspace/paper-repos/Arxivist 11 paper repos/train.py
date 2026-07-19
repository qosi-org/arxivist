#!/usr/bin/env python
"""
Training entrypoint for "Asset Pricing in Pre-trained Transformers" (arXiv:2505.01575).

Trains one model family (config `model.family`) over the rolling-window scheme of
Section 3: 102-month training window, 30-month validation window, re-estimated every
12 months, from Jan 1957 through Dec 2012 (in-sample) — see config.yaml `data` section.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug          # quick local smoke test
    python train.py --config configs/config.yaml --dry-run        # build without training
    python train.py --config configs/config.yaml --resume checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sert_asset_pricing.data.dataset import RollingFactorDataset, RollingWindowSpec  # noqa: E402
from sert_asset_pricing.data.transforms import FactorPreprocessor  # noqa: E402
from sert_asset_pricing.models.transformer_variants import build_model  # noqa: E402
from sert_asset_pricing.training.trainer import RollingWindowTrainer  # noqa: E402
from sert_asset_pricing.utils.config import ConfigLoader, get_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper model variant.")
    parser.add_argument("--config", type=str, required=True, help="path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=None, help="random seed override")
    parser.add_argument("--debug", action="store_true", help="reduced dataset/steps for quick local testing")
    parser.add_argument("--dry-run", action="store_true", help="build all components but don't train")
    return parser.parse_args()


def load_data(cfg: dict, debug: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load factors/returns CSVs, or synthesize a tiny dataset in --debug mode.

    Real data is NOT distributed with this repo — see data/README_data.md for how to
    obtain the 182-factor / 420-stock panel described in Section 3 of the paper.
    """
    if debug or not (os.path.exists(cfg["data"]["factors_path"]) and os.path.exists(cfg["data"]["returns_path"])):
        print("[train.py] Using synthetic debug data (real dataset not found or --debug set).")
        n_dates = 200
        dates = pd.date_range("2000-01-01", periods=n_dates, freq="MS")
        factors = pd.DataFrame(
            torch.randn(n_dates, cfg["data"].get("debug_num_factors", 20)).numpy(),
            index=dates,
        )
        returns = pd.DataFrame(
            torch.randn(n_dates, cfg["data"].get("debug_num_stocks", 10)).numpy() * 0.05,
            index=dates,
        )
        return factors, returns

    factors = pd.read_csv(cfg["data"]["factors_path"], index_col=0, parse_dates=True)
    returns = pd.read_csv(cfg["data"]["returns_path"], index_col=0, parse_dates=True)
    common_idx = factors.index.intersection(returns.index)
    return factors.loc[common_idx], returns.loc[common_idx]


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader(args.config).load()

    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    set_seed(seed, deterministic=cfg["training"].get("deterministic", True))
    device = get_device(cfg["hardware"]["device"])
    print(f"[train.py] Using device: {device}")

    factors, returns = load_data(cfg, args.debug)
    preprocessor = FactorPreprocessor(missing_value_threshold=cfg["data"]["missing_value_threshold"])
    factors_clean = preprocessor.fit_transform(factors)
    print(f"[train.py] Factors after missingness filter: {factors_clean.shape[1]} "
          f"(dropped {len(preprocessor.dropped_columns_)})")

    # Override model.input_factor_dim to match the actual cleaned factor count for debug runs.
    if args.debug:
        cfg["model"]["input_factor_dim"] = factors_clean.shape[1]
        cfg["model"]["num_heads"] = min(cfg["model"]["num_heads"], 4)
        # Keep d_model a multiple of num_heads (required by multi-head attention).
        target_d_model = min(cfg["model"]["d_model"], 32)
        cfg["model"]["d_model"] = max(
            cfg["model"]["num_heads"],
            (target_d_model // cfg["model"]["num_heads"]) * cfg["model"]["num_heads"],
        )
        cfg["training"]["max_epochs"] = min(cfg["training"]["max_epochs"], 3)
        cfg["training"]["train_window"] = min(cfg["training"]["train_window"], 40)
        cfg["training"]["val_window"] = min(cfg["training"]["val_window"], 10)

    spec = RollingWindowSpec(
        train_window=cfg["training"]["train_window"],
        val_window=cfg["training"]["val_window"],
        step_size=cfg["training"]["step_size"],
    )
    train_ds = RollingFactorDataset(factors_clean, returns, spec=spec, split="train")
    val_ds = RollingFactorDataset(factors_clean, returns, spec=spec, split="val")
    print(f"[train.py] Rolling windows: {len(train_ds)} (train), {len(val_ds)} (val)")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train.py] Model: {model}")
    print(f"[train.py] Trainable parameters: {n_params:,}")

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"[train.py] Resumed from checkpoint: {args.resume}")

    if args.dry_run:
        print("[train.py] --dry-run set: components built successfully, exiting without training.")
        return

    trainer = RollingWindowTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, config=cfg,
        checkpoint_dir="checkpoints", device=device,
    )
    history = trainer.fit()
    print(f"[train.py] Training complete. Final train_loss={history['train_loss'][-1]:.6f}, "
          f"val_loss={history['val_loss'][-1]:.6f}")
    print("[train.py] Best checkpoint saved to checkpoints/best.pt")


if __name__ == "__main__":
    main()
