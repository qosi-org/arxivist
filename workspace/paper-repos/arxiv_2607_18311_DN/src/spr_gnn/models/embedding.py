"""
Trainable taxonomic/species identifier embedding.

Implements the SIR's `TaxonomicEmbeddingLayer` module (Sec 4.1, Fig. 4):
maps the categorical per-node identifier to a 16-dim learned vector, with
vocabulary size set dynamically to max_id + 1 to handle large isolate counts.

SIR implementation_assumptions[4] (confidence 0.55): the categorical id is
treated here as a *species* id (per Fig. 4's caption) rather than a
per-isolate id. If the released Zenodo dataset schema indicates otherwise,
swap the `num_species` argument for the isolate-count vocabulary size --
no other code changes are required.
"""
from __future__ import annotations

import torch
from torch import nn


class TaxonomicEmbedding(nn.Module):
    """Embeds a per-node categorical taxonomic id into a dense vector.

    Args:
        num_categories: vocabulary size, i.e. max_id + 1 (Sec 4.1).
        embedding_dim: output embedding dimension (16, per Sec 4.1).
    """

    def __init__(self, num_categories: int, embedding_dim: int = 16) -> None:
        super().__init__()
        if num_categories < 1:
            raise ValueError(f"num_categories must be >= 1, got {num_categories}")
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_dim)

    def forward(self, node_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            node_ids: [N] integer taxonomic/species ids, one per node.

        Returns:
            [N, embedding_dim] float tensor.
        """
        assert node_ids.dim() == 1, f"Expected node_ids of shape [N], got {node_ids.shape}"
        return self.embedding(node_ids)

    def __repr__(self) -> str:  # noqa: D105
        return f"TaxonomicEmbedding(vocab={self.embedding.num_embeddings}, dim={self.embedding_dim})"
