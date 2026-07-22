#!/usr/bin/env python
"""
Training entrypoint for LATTICE.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug        # quick smoke test
    python train.py --config configs/config.yaml --dry-run      # build only, no training
    python train.py --config configs/config.yaml --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lattice.data.dataset import SyntheticSpatialMultimodalDataset  # noqa: E402
from lattice.models.lattice_model import LatticeModel  # noqa: E402
from lattice.training.trainer import LatticeTrainer  # noqa: E402
from lattice.utils.config import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LATTICE (arXiv:2607.14410 reproduction)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--debug", action="store_true", help="Reduce dataset size/epochs for a quick smoke test"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build all components without training"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config["training"]["seed"] = args.seed
    set_seed(config["training"]["seed"])

    if args.debug:
        print("[--debug] Using a reduced synthetic dataset and epoch budget for a smoke test.")
        config["data"]["num_synthetic_samples"] = 2
        config["data"]["spots_per_sample"] = 200
        config["training"]["num_epochs"] = 2
        config["training"]["early_stopping_patience"] = 2

    print(f"Loaded config from {args.config}")
    print(
        "NOTE: this repo trains on a SYNTHETIC dataset by default because the paper's "
        "private 11-sample melanoma cohort cannot be released. See data/README_data.md "
        "to plug in real data."
    )

    dataset = SyntheticSpatialMultimodalDataset(
        num_samples=config["data"]["num_synthetic_samples"],
        spots_per_sample=config["data"]["spots_per_sample"],
        gene_count_range=tuple(config["data"]["gene_count_range"]),
        num_modality_blocks=config["model"]["num_modality_blocks"],
        seed=config["training"]["seed"],
    )

    model = LatticeModel(
        modality_dims=dataset.modality_dims,
        hidden_dim=config["model"]["hidden_dim"],
        graph_num_layers=config["model"]["graph_encoder"]["num_layers"],
        graph_num_heads=config["model"]["graph_encoder"]["num_heads"],
        graph_dropout=config["model"]["graph_encoder"]["dropout"],
        decoder_hidden_width=config["model"]["decoder"]["hidden_width"],
        proj_hidden_width=config["model"]["projection_head"]["hidden_width"],
        proj_output_dim=config["model"]["projection_head"]["output_dim"],
        aligned_modality_pair=tuple(config["model"]["projection_head"]["aligned_modality_pair"]),
        knn_k=config["data"]["knn_k"],
        edge_weight_mode=config["data"]["edge_weight_mode"],
        gaussian_sigma=config["data"].get("gaussian_sigma"),
    )
    print(model)

    if args.dry_run:
        sample = dataset[0]
        outputs = model.embed(sample["modality_blocks"], sample["presence_mask"], sample["coords"])
        print(f"[--dry-run] Built model + dataset OK. Sample embedding shape: {tuple(outputs.shape)}")
        return

    trainer = LatticeTrainer(model, config, checkpoint_dir="checkpoints")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=trainer.device)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    history = trainer.fit(dataset)
    print(f"Training complete. Final train_loss={history['train_loss'][-1]:.4f} "
          f"val_loss={history['val_loss'][-1]:.4f}")
    print("Best checkpoint: checkpoints/best.pt")


if __name__ == "__main__":
    main()
