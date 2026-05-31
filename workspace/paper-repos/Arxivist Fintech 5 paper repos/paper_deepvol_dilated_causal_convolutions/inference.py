"""
Single-sample inference with DeepVol.
Usage: python inference.py --input path/to/returns.npy [--checkpoint checkpoints/best.ckpt]
"""
import argparse
import numpy as np
import torch
from pathlib import Path

from deepvol.utils.config import load_config, get_device
from deepvol.training.trainer import DeepVolLightning


def parse_args():
    p = argparse.ArgumentParser(description="DeepVol inference")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--input", required=True, help="Path to numpy array [1, T*J] of intraday returns")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.training.accelerator)

    returns = np.load(args.input).astype(np.float32)
    seq_len = cfg.data.conditioning_range * cfg.data.intervals_per_day
    if returns.ndim == 1:
        returns = returns[np.newaxis, np.newaxis, :]  # [1, 1, L]
    x = torch.from_numpy(returns).to(device)

    ckpt = args.checkpoint or cfg.evaluation.checkpoint_path
    if ckpt and Path(ckpt).exists():
        model = DeepVolLightning.load_from_checkpoint(ckpt, cfg=cfg)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.eval().to(device)
    with torch.no_grad():
        sigma2_hat = model(x).item()

    print(f"Day-ahead realised variance forecast: {sigma2_hat:.6f}")
    print(f"Implied annualised volatility (approx): {(sigma2_hat * 252) ** 0.5 * 100:.2f}%")


if __name__ == "__main__":
    main()
