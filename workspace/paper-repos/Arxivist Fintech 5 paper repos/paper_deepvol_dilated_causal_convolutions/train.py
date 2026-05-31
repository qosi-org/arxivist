"""
DeepVol training entrypoint.
Usage: python train.py --config configs/config.yaml [--seed 42] [--debug] [--dry-run]
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from deepvol.utils.config import load_config, set_seed
from deepvol.data.dataset import VolatilityDataset, SyntheticVolatilityDataset
from deepvol.training.trainer import DeepVolLightning


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepVol")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=None, help="Override seed in config")
    p.add_argument("--debug", action="store_true", help="Tiny dataset + 2 epochs")
    p.add_argument("--dry-run", action="store_true", help="Build all components, no training")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.training.seed
    set_seed(seed, cfg.training.deterministic)

    # Data
    data_dir = Path(cfg.data.data_dir)
    if data_dir.exists() and (data_dir / "X_train.npy").exists():
        train_ds = VolatilityDataset.from_numpy_files(
            str(data_dir / "X_train.npy"), str(data_dir / "y_train.npy")
        )
        val_ds = VolatilityDataset.from_numpy_files(
            str(data_dir / "X_val.npy"), str(data_dir / "y_val.npy")
        )
    else:
        print("⚠ Processed data not found — using synthetic data for demo.")
        print("  See data/README_data.md for real data setup instructions.")
        n = 200 if args.debug else 2000
        seq_len = cfg.data.conditioning_range * cfg.data.intervals_per_day
        train_ds = SyntheticVolatilityDataset(n_samples=n, seq_len=seq_len, seed=seed)
        val_ds = SyntheticVolatilityDataset(n_samples=n // 5, seq_len=seq_len, seed=seed + 1)

    bs = 32 if args.debug else cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=cfg.training.num_workers, pin_memory=True)

    model = DeepVolLightning(cfg)
    print(f"\nModel: {model.model}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    if args.dry_run:
        print("✓ Dry run complete — all components built successfully.")
        return

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.training.early_stopping_patience, mode="min"),
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=2 if args.debug else cfg.training.num_epochs,
        accelerator=cfg.training.accelerator,
        precision=cfg.training.precision,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print(f"\n✓ Training complete. Best checkpoint: {callbacks[1].best_model_path}")


if __name__ == "__main__":
    main()
