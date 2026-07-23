#!/usr/bin/env python
"""
Single-sample inference entrypoint for LATTICE.

Embeds one sample's spot-by-modality tensors into Z using a trained
checkpoint (LatticeModel.embed, no masking applied).

Usage:
    python inference.py --checkpoint checkpoints/best.pt --sample-index 0 \\
        --output results/embedding.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lattice.data.dataset import SyntheticSpatialMultimodalDataset  # noqa: E402
from lattice.models.lattice_model import LatticeModel  # noqa: E402
from lattice.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed one sample with a trained LATTICE model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/embedding.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample = dataset[args.sample_index]
    with torch.no_grad():
        z = model.embed(sample["modality_blocks"], sample["presence_mask"], sample["coords"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, z.numpy())
    print(f"Saved embedding of shape {tuple(z.shape)} to {output_path}")


if __name__ == "__main__":
    main()
