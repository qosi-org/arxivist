#!/usr/bin/env python3
"""
train_kronos.py
Train the Kronos forecasting head on CWS trajectories.
Paper: arXiv:2605.11645, Section 3.3.3

Usage:
    python train_kronos.py --data_dir results/detection/ --epochs 50
    python train_kronos.py --config configs/config.yaml --dry_run
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Train Kronos Forecasting Head")
    parser.add_argument("--data_dir", type=str, default="results/detection/")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Override config device (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints/kronos/")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 2 epochs, small batch")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build model and validate setup without training")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required for Kronos training. pip install torch")
        return

    from geomherd.utils.config import GeomHerdConfig, set_global_seed
    from geomherd.forecasting.kronos_head import KronosHead, PriceTokeniser

    set_global_seed(args.seed)

    if os.path.exists(args.config):
        cfg = GeomHerdConfig.from_yaml(args.config)
    else:
        cfg = GeomHerdConfig()

    kc = cfg.kronos
    device = args.device or cfg.hardware.device
    if args.epochs:
        kc.train_epochs = args.epochs
    if args.debug:
        kc.train_epochs = 2
        kc.batch_size = 4
        print("[DEBUG] 2 epochs, batch_size=4")

    # Build model
    tokeniser = PriceTokeniser(ohlcv_dim=5, embed_dim=kc.d_model,
                                codebook_size=kc.tokeniser_codebook_size)
    model = KronosHead(
        d_model=kc.d_model, n_layers=kc.n_layers, n_heads=kc.n_heads,
        price_tokeniser=tokeniser
    ).to(device)

    # Count trainable params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nKronos Head — trainable: {n_trainable:,} / total: {n_total:,} params")
    print(f"  d_model={kc.d_model}, n_layers={kc.n_layers}, n_heads={kc.n_heads}")
    print(f"  [ASSUMED architecture — see Risk R2 in architecture_plan.json]")

    if args.dry_run:
        # Validate forward pass
        dummy_ohlcv = torch.randn(2, kc.context_len, 5).to(device)
        dummy_triplet = torch.randn(2, 3).to(device)
        pred = model(dummy_ohlcv, dummy_triplet)
        print(f"\n[DRY RUN] Forward pass OK. Output shape: {pred.shape}")
        print("[DRY RUN] Exiting without training.")
        return

    # Training loop (simplified — full data pipeline would load from data_dir)
    print(f"\nTraining for {kc.train_epochs} epochs on device={device}")
    print(f"NOTE: Full training requires CWS trajectory data in {args.data_dir}")
    print("      Run run_detection.py first to generate trajectory data.")
    print("      See data/README_data.md for data preparation instructions.")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=kc.lr
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")

    for epoch in range(kc.train_epochs):
        # Placeholder training loop — replace with real data loader
        # Full implementation requires DataLoader over CWS trajectories
        # with (ohlcv_window, geomherd_triplet, next_log_return) tuples
        t0 = time.time()
        model.train()
        # Synthetic data for structure validation only
        dummy_ohlcv = torch.randn(kc.batch_size, kc.context_len, 5).to(device)
        dummy_triplet = torch.randn(kc.batch_size, 3).to(device)
        dummy_target = torch.randn(kc.batch_size).to(device)
        optimizer.zero_grad()
        pred = model(dummy_ohlcv, dummy_triplet)
        loss = torch.nn.functional.l1_loss(pred, dummy_target)
        loss.backward()
        optimizer.step()
        elapsed = time.time() - t0
        if epoch % max(1, kc.train_epochs // 10) == 0:
            print(f"  Epoch {epoch+1:3d}/{kc.train_epochs} | loss={loss.item():.4f} | {elapsed:.2f}s")

    # Save final checkpoint
    ckpt_path = Path(args.output_dir) / "kronos_final.pt"
    torch.save({
        "epoch": kc.train_epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": kc.d_model,
            "n_layers": kc.n_layers,
            "n_heads": kc.n_heads,
        },
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
