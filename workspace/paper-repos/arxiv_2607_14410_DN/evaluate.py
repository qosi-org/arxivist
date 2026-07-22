#!/usr/bin/env python
"""
Evaluation entrypoint for LATTICE.

Computes ARI, NMI, spatial contiguity, silhouette, and MUS for a trained
checkpoint (Section 4.1, Appendix A.2). Because the paper's reported numbers
(Table 2/3) come from a private cohort, this script evaluates against the
synthetic dataset's ground-truth "domain" labels as a reference clustering
stand-in -- see README.md "Reproducibility Notes".

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --modality-level M2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lattice.data.dataset import SyntheticSpatialMultimodalDataset  # noqa: E402
from lattice.evaluation.metrics import (  # noqa: E402
    ari_nmi,
    bio_jaccard_overlap,
    bio_knn_consistency,
    embedding_silhouette,
    spatial_contiguity,
)
from lattice.models.lattice_model import LatticeModel  # noqa: E402
from lattice.utils.config import load_config  # noqa: E402

# Modality ladder (Section 1): which of the 5 modality blocks are "on" at each level.
MODALITY_LADDER = {
    "M1": [True, False, False, False, False],
    "M2": [True, True, False, False, False],
    "M3": [True, True, True, False, False],
    "M4": [True, True, True, True, False],
    "M5": [True, True, True, True, True],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LATTICE checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--modality-level", type=str, default="M5", choices=list(MODALITY_LADDER.keys())
    )
    return parser.parse_args()


def leiden_or_kmeans_labels(z: np.ndarray, resolution_sweep: list[float]) -> np.ndarray:
    """Cluster embeddings, preferring Leiden (paper's method) with a KMeans fallback.

    Paper reference: Appendix H, "Downstream analysis" -- Leiden clustering
    with target K imported from (undisclosed) SARSIM metadata. SIR
    ambiguities[4] (confidence 0.5): we instead sweep Leiden resolutions and
    pick the one maximizing silhouette score. If leidenalg/python-igraph are
    unavailable, falls back to KMeans with the same resolution-swept K
    selection logic replaced by a small K grid.
    """
    try:
        import igraph as ig
        import leidenalg
        from sklearn.neighbors import kneighbors_graph

        adj = kneighbors_graph(z, n_neighbors=6, mode="connectivity")
        sources, targets = adj.nonzero()
        g = ig.Graph(directed=False)
        g.add_vertices(z.shape[0])
        g.add_edges(list(zip(sources.tolist(), targets.tolist())))

        best_labels, best_score = None, -1.0
        for res in resolution_sweep:
            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res
            )
            labels = np.array(partition.membership)
            if len(set(labels.tolist())) < 2:
                continue
            score = embedding_silhouette(z, labels)
            if score > best_score:
                best_labels, best_score = labels, score
        if best_labels is not None:
            return best_labels
    except ImportError:
        print("[evaluate.py] leidenalg/python-igraph not available, falling back to KMeans.")

    from sklearn.cluster import KMeans

    best_labels, best_score = None, -1.0
    for k in [4, 6, 8, 10]:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(z)
        score = embedding_silhouette(z, labels)
        if score > best_score:
            best_labels, best_score = labels, score
    return best_labels


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

    active = MODALITY_LADDER[args.modality_level]
    print(f"Evaluating modality level {args.modality_level}: active blocks = {active}")

    all_metrics = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        modality_blocks = [
            blk if is_on else torch.zeros_like(blk)
            for blk, is_on in zip(sample["modality_blocks"], active)
        ]
        presence_mask = sample["presence_mask"].clone()
        for m, is_on in enumerate(active):
            if not is_on:
                presence_mask[:, m] = 0.0

        with torch.no_grad():
            z = model.embed(modality_blocks, presence_mask, sample["coords"]).numpy()
        coords = sample["coords"].numpy()
        reference_labels = sample["domain_labels"].numpy()

        pred_labels = leiden_or_kmeans_labels(z, config["evaluation"]["leiden_resolution_sweep"])

        m = ari_nmi(pred_labels, reference_labels)
        m["spatial_contiguity"] = spatial_contiguity(coords, pred_labels, k=config["data"]["knn_k"])
        m["silhouette"] = embedding_silhouette(z, pred_labels)
        m["bio_knn"] = bio_knn_consistency(z, pred_labels, k=config["data"]["knn_k"])
        m["bio_jaccard"] = bio_jaccard_overlap(z, coords, k=config["data"]["knn_k"])
        all_metrics.append(m)
        print(f"  sample {idx}: {m}")

    print("\n=== Cohort mean +- std ===")
    for key in all_metrics[0]:
        vals = np.array([m[key] for m in all_metrics])
        print(f"  {key}: {vals.mean():.3f} +- {vals.std():.3f}")

    print(
        "\nNOTE: these numbers are computed on SYNTHETIC data against synthetic "
        "domain labels, and are NOT directly comparable to Table 2/3 in the paper "
        "(which used the private melanoma cohort). Use this script's output for "
        "sanity-checking your own reproduction, and feed real results to Stage 6 "
        "(Results Comparator) if you have access to the real cohort."
    )


if __name__ == "__main__":
    main()
