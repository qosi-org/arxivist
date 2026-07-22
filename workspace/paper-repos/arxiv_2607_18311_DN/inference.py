#!/usr/bin/env python3
"""
Single-pair inference entrypoint: predict the SPR distance between two
user-supplied Newick trees using a trained checkpoint.

Example:
    python inference.py --tree-a a.nwk --tree-b b.nwk --checkpoint outputs/best_model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from spr_gnn.utils.config import Config, resolve_device  # noqa: E402
from spr_gnn.data.newick_parser import NewickToGraph  # noqa: E402
from spr_gnn.data.node_features import NodeFeatureExtractor  # noqa: E402
from spr_gnn.models.siamese_gin import SiameseGINRegressor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict SPR distance for a pair of Newick trees.")
    parser.add_argument("--tree-a", type=str, required=True, help="Path to first Newick (.nwk) file")
    parser.add_argument("--tree-b", type=str, required=True, help="Path to second Newick (.nwk) file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--species-id",
        type=int,
        default=0,
        help="Categorical species id for both trees (0-indexed into data.species in config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)
    device = resolve_device(config.get("hardware", "device", "cuda_if_available_else_cpu"))

    num_species = len(config.get("data", "species", ["default"]))
    model = SiameseGINRegressor(
        num_species=num_species,
        species_embedding_dim=config.get("model", "species_embedding_dim", 16),
        node_feature_dim_continuous=config.get("model", "node_feature_dim_continuous", 3),
        num_gin_layers=config.get("model", "num_gin_layers", 2),
        gin_hidden_dim=config.get("model", "gin_hidden_dim", 128),
        mlp_head_dims=config.get("model", "mlp_head_dims", [256, 128, 64, 1]),
        dropout=config.get("model", "dropout", 0.3),
        clamp_min=config.get("training", "clamp_min", 0.0),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    parser_ = NewickToGraph()
    extractor = NodeFeatureExtractor()

    def load_tree(path: str):
        with open(path) as f:
            newick_str = f.read()
        graph = parser_.parse(newick_str)
        edge_index = parser_.to_bidirectional_edge_index(graph)
        continuous, node_ids = extractor.extract(graph, args.species_id)
        from torch_geometric.data import Data

        return Data(x=continuous, edge_index=edge_index, node_id=node_ids)

    tree_a = Batch.from_data_list([load_tree(args.tree_a)]).to(device)
    tree_b = Batch.from_data_list([load_tree(args.tree_b)]).to(device)

    with torch.no_grad():
        pred = model(tree_a, tree_b)

    print(f"Predicted SPR distance ({Path(args.tree_a).name} vs. {Path(args.tree_b).name}): {pred.item():.2f}")
    print("(NOTE: this reproduces the phangorn-heuristic supervision target, not the exact rooted rspr distance -- see Sec 3/5.2 of the paper.)")


if __name__ == "__main__":
    main()
