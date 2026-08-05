#!/usr/bin/env python
"""
train.py — Training entrypoint for the NeRF reproduction (arXiv:2003.08934).

Usage:
    python train.py --config configs/config.yaml --datadir /data/lego --expname lego_full
    python train.py --config configs/config_debug.yaml --debug        # fast local smoke test
    python train.py --config configs/tiny_consumer.yaml --tiny        # TinyNeRF-style consumer preset
    python train.py --config configs/config.yaml --datadir ... --dry-run   # validate setup only
"""

from __future__ import annotations

import argparse
import os

from nerf.data.dataset import BlenderSyntheticDataset, LLFFRealDataset
from nerf.data.transforms import build_ray_dataset
from nerf.models.radiance_field import NeRFModel
from nerf.training.trainer import Trainer
from nerf.utils.config import NeRFConfig, set_global_seed
from nerf.utils.tiny_variant import TinyNeRFPreset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NeRF model (arXiv:2003.08934).")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--datadir", type=str, required=True, help="Path to scene dataset directory.")
    parser.add_argument("--expname", type=str, required=True, help="Experiment name (log/checkpoint subdir).")
    parser.add_argument(
        "--num_steps", type=int, default=None, help="Override training.num_training_steps."
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Apply the TinyNeRF-inspired consumer-hardware preset (utils/tiny_variant.py).",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Reduce dataset size and steps for a quick local correctness check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build all components (data, model, trainer) but do not train; validates setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = NeRFConfig.from_yaml(args.config)
    overrides: dict = {"data": {"datadir": args.datadir}}
    if args.num_steps is not None:
        overrides.setdefault("training", {})["num_training_steps"] = args.num_steps
    if args.seed is not None:
        overrides.setdefault("training", {})["seed"] = args.seed
    config = config.merge_overrides(overrides)

    if args.tiny:
        config = NeRFConfig(raw=TinyNeRFPreset().apply(config.raw))

    if args.debug:
        config = config.merge_overrides(
            {
                "training": {"num_training_steps": 20, "checkpoint_every": 10, "log_every": 1},
                "data": {"testskip": 1, "half_res": True},
            }
        )

    print(config)
    set_global_seed(config.get("training", "seed", 0), deterministic=config.get("training", "deterministic", False))

    dataset_type = config["data"]["dataset_type"]
    if dataset_type == "blender":
        loaded = BlenderSyntheticDataset(
            args.datadir,
            half_res=config["data"]["half_res"],
            testskip=config["data"]["testskip"],
        ).load()
    elif dataset_type == "llff":
        loaded = LLFFRealDataset(args.datadir).load()
    else:
        raise ValueError(f"Unsupported dataset_type for training: {dataset_type}")

    model = NeRFModel(config.raw)

    log_dir = os.path.join("checkpoints", args.expname)
    trainer = Trainer(model, config.raw, loaded, log_dir=log_dir)

    if args.resume is not None:
        trainer.checkpoint.restore(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    train_images = loaded["images"]["train"] if "images" in loaded and isinstance(loaded["images"], dict) else loaded["images"]
    train_poses = loaded["poses"]["train"] if "poses" in loaded and isinstance(loaded["poses"], dict) else loaded["poses"]
    num_train_rays = train_images.shape[0] * train_images.shape[1] * train_images.shape[2]

    trainer.print_training_summary(num_train_rays)

    if args.dry_run:
        print("Dry run complete: data, model, and trainer built successfully. Exiting without training.")
        return

    train_ds = build_ray_dataset(
        train_images,
        train_poses,
        loaded["hwf"],
        batch_size=config["training"]["ray_batch_size"],
        seed=config.get("training", "seed", 0),
    )
    trainer.fit(train_ds, num_steps=config["training"]["num_training_steps"])


if __name__ == "__main__":
    main()
