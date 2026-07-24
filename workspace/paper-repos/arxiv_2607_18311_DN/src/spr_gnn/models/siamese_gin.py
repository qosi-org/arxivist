"""
Siamese GIN Regressor — full model (Sec 4.2, Fig. 3 & Fig. 5).

Both trees of a pair pass through the *same* GINEncoder (shared weights,
i.e. a Siamese network). The two 128-d graph embeddings are concatenated
into a 256-d vector and regressed by an MLP head to a single scalar SPR
distance prediction, clamped at zero (negative distances are biologically
meaningless, Sec 4.3).

Implements SIR architecture modules: TaxonomicEmbeddingLayer,
FeatureConcatenation, GINLayer_1/2, GlobalAddPooling, PairConcatenation,
MLPRegressionHead.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch

from spr_gnn.models.embedding import TaxonomicEmbedding
from spr_gnn.models.gin_encoder import GINEncoder


class SiameseGINRegressor(nn.Module):
    """End-to-end Siamese GIN model predicting SPR distance for a tree pair.

    Args:
        num_species: vocabulary size for the taxonomic embedding (max_id + 1).
        species_embedding_dim: dim of the taxonomic embedding (16, Sec 4.1).
        node_feature_dim_continuous: dim of raw continuous node features (3, Sec 4.1).
        num_gin_layers: GIN layer count (SIR confidence 0.6 -- see gin_encoder.py).
        gin_hidden_dim: GIN hidden dimension (128, Sec 4.2).
        mlp_head_dims: MLP head layer sizes, e.g. [256, 128, 64, 1] (Sec 4.2).
        dropout: dropout probability in the MLP head (0.3, Sec 4.2).
        clamp_min: minimum output value (0.0 -- SPR distances are non-negative).
    """

    def __init__(
        self,
        num_species: int,
        species_embedding_dim: int = 16,
        node_feature_dim_continuous: int = 3,
        num_gin_layers: int = 2,
        gin_hidden_dim: int = 128,
        mlp_head_dims: list[int] | None = None,
        dropout: float = 0.3,
        clamp_min: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_head_dims = mlp_head_dims or [256, 128, 64, 1]
        if mlp_head_dims[0] != 2 * gin_hidden_dim:
            raise ValueError(
                f"mlp_head_dims[0] ({mlp_head_dims[0]}) must equal 2*gin_hidden_dim "
                f"({2 * gin_hidden_dim}) since two tree embeddings are concatenated."
            )

        self.clamp_min = clamp_min
        self.species_embedding = TaxonomicEmbedding(num_species, species_embedding_dim)

        gin_in_dim = node_feature_dim_continuous + species_embedding_dim  # = 19
        # Single shared encoder instance -> applied to both trees (Siamese weight sharing).
        self.encoder = GINEncoder(in_dim=gin_in_dim, hidden_dim=gin_hidden_dim, num_layers=num_gin_layers)

        head_layers: list[nn.Module] = []
        for i in range(len(mlp_head_dims) - 1):
            head_layers.append(nn.Linear(mlp_head_dims[i], mlp_head_dims[i + 1]))
            is_last_layer = i == len(mlp_head_dims) - 2
            if not is_last_layer:
                head_layers.append(nn.ReLU())
                head_layers.append(nn.Dropout(dropout))
        self.mlp_head = nn.Sequential(*head_layers)

    def _embed_tree(self, tree: Batch) -> torch.FloatTensor:
        """Encode one batch of trees into [num_graphs, gin_hidden_dim] embeddings.

        Expects `tree.x` = [N,3] continuous features, `tree.node_id` = [N]
        categorical species ids, `tree.edge_index` = [2,E], `tree.batch` = [N].
        """
        species_embed = self.species_embedding(tree.node_id)     # [N, 16]
        node_features = torch.cat([tree.x, species_embed], dim=-1)  # [N, 19] -- FeatureConcatenation
        return self.encoder(node_features, tree.edge_index, tree.batch)  # [num_graphs, 128]

    def forward(self, tree_a: Batch, tree_b: Batch) -> torch.FloatTensor:
        """
        Args:
            tree_a: PyG Batch for the first tree of each pair in the minibatch.
            tree_b: PyG Batch for the second tree of each pair (same batch size).

        Returns:
            [num_graphs] predicted SPR distances, clamped to >= clamp_min.
        """
        h_a = self._embed_tree(tree_a)  # [num_graphs, 128] -- shared encoder
        h_b = self._embed_tree(tree_b)  # [num_graphs, 128] -- same encoder instance
        pair_embedding = torch.cat([h_a, h_b], dim=-1)  # [num_graphs, 256] -- PairConcatenation
        pred = self.mlp_head(pair_embedding).squeeze(-1)  # [num_graphs]
        return torch.clamp(pred, min=self.clamp_min)

    def __repr__(self) -> str:  # noqa: D105
        return f"SiameseGINRegressor(encoder={self.encoder}, head={self.mlp_head})"
