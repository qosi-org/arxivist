"""
DeepVol evaluation entrypoint.
Usage: python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.ckpt
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from deepvol.utils.config import load_config, set_seed, get_device
from deepvol.data.dataset import VolatilityDataset, SyntheticVolatilityDataset
from deepvol.training.trainer import DeepVolLightning
from deepvol.evaluation.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepVol")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.accelerator)

    data_dir = Path(cfg.data.data_dir)
    seq_len = cfg.data.conditioning_range * cfg.data.intervals_per_day
    if data_dir.exists() and (data_dir / "X_test.npy").exists():
        test_ds = VolatilityDataset.from_numpy_files(
            str(data_dir / "X_test.npy"), str(data_dir / "y_test.npy")
        )
    else:
        print("⚠ Using synthetic test data.")
        test_ds = SyntheticVolatilityDataset(n_samples=500, seq_len=seq_len, seed=99)

    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    ckpt = args.checkpoint or cfg.evaluation.checkpoint_path
    if ckpt and Path(ckpt).exists():
        model = DeepVolLightning.load_from_checkpoint(ckpt, cfg=cfg)
    else:
        print("⚠ No checkpoint found — using untrained model.")
        model = DeepVolLightning(cfg)

    model.eval().to(device)
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    metrics = compute_all_metrics(targets, preds)

    print("\n=== DeepVol Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k.upper():8s}: {v:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(out_dir / "predictions.npy", preds)
    np.save(out_dir / "targets.npy", targets)
    print(f"\n✓ Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
