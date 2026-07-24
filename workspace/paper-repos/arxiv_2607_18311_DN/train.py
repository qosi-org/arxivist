#!/usr/bin/env python3
"""
Training entrypoint for the Siamese GIN Regressor (arXiv:2607.18311).

Example:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug   # quick local smoke test
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from spr_gnn.utils.config import Config, resolve_device, set_seed  # noqa: E402
from spr_gnn.data.dataset import TreePairDataModule  # noqa: E402
from spr_gnn.models.siamese_gin import SiameseGINRegressor  # noqa: E402
from spr_gnn.training.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Siamese GIN SPR-distance regressor.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data.data_dir from config")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Where to write checkpoints/logs")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed from config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Reduce dataset/epochs for a quick smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Build everything but skip actual training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)

    if args.debug:
        config = copy.deepcopy(config)
        config.raw["training"]["max_epochs"] = 2
        config.raw["training"]["batch_size"] = 4
        print("[--debug] max_epochs=2, batch_size=4 for a quick smoke test.")

    seed = args.seed if args.seed is not None else config.get("training", "seed", 42)
    set_seed(seed, deterministic=config.get("training", "deterministic", False))
    device = resolve_device(config.get("hardware", "device", "cuda_if_available_else_cpu"))
    print(f"Using device: {device}")

    data_dir = args.data_dir or config.get("data", "data_dir")
    master_csv = str(Path(data_dir) / "master_pairs.csv") if args.data_dir else config.get("data", "master_pairs_csv")

    dm = TreePairDataModule(config)
    dm.setup(master_csv_path=master_csv, seed=config.get("data", "split_seed", 42))

    model = SiameseGINRegressor(
        num_species=dm.num_species,
        species_embedding_dim=config.get("model", "species_embedding_dim", 16),
        node_feature_dim_continuous=config.get("model", "node_feature_dim_continuous", 3),
        num_gin_layers=config.get("model", "num_gin_layers", 2),
        gin_hidden_dim=config.get("model", "gin_hidden_dim", 128),
        mlp_head_dims=config.get("model", "mlp_head_dims", [256, 128, 64, 1]),
        dropout=config.get("model", "dropout", 0.3),
        clamp_min=config.get("training", "clamp_min", 0.0),
    )

    if args.resume:
        import torch

        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed model weights from {args.resume}")

    trainer = Trainer(model, config, device, output_dir=args.output_dir)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if args.dry_run:
        print("[--dry-run] Model, data, and trainer built successfully. Skipping training.")
        print(model)
        return

    history = trainer.fit(train_loader, val_loader)
    print(f"Training complete. Best checkpoint: {args.output_dir}/best_model.pt")
    print(f"Final val R^2: {history['val_metrics'][-1]['r2']:.4f}")


if __name__ == "__main__":
    main()
