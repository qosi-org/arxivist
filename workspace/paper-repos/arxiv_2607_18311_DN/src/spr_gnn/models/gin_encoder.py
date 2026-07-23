"""
GIN encoder: shared-weight tree encoder used by both branches of the
Siamese regressor.

Implements SIR mathematical_spec[0] (Eq. 1, Sec 4.2), the GIN node update:

    h_i^(k) = MLP^(k)( (1 + eps^(k)) * h_i^(k-1) + sum_{j in N(i)} h_j^(k-1) )

stacked for `num_gin_layers`, each followed by BatchNorm1d + ReLU (Fig. 5),
then global add pooling (SIR mathematical_spec[1], Eq. 2, Sec 4.2):

    v_g = sum_{i in V_g} h_i

WARNING: low-confidence implementation (SIR confidence 0.6, ambiguities[0]).
The paper's body text states "two GIN layers", but Figure 5 diagrams three
stacked GINConv/BatchNorm/ReLU blocks per branch. `num_gin_layers` defaults
to 2 (text) but is fully config-driven -- set model.num_gin_layers: 3 in
configs/config.yaml to reproduce the alternative reading of Fig. 5.
TODO: disambiguate against the authors' released code/results if available.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool


def _build_gin_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """Two-layer MLP (Linear-ReLU-Linear) wrapped by each GINConv, per Sec 4.2."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINEncoder(nn.Module):
    """Stacked GIN layers + global add pooling, producing one embedding per tree.

    Args:
        in_dim: input node feature dimension (19: 3 continuous + 16 species-embed).
        hidden_dim: GIN hidden dimension (128, per Sec 4.2).
        num_layers: number of stacked GIN layers. SIR confidence 0.6 -- see
            module docstring. Default 2 per the paper's body text.
    """

    def __init__(self, in_dim: int = 19, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_dim = in_dim if layer_idx == 0 else hidden_dim
            # Eq. 1: GINConv wraps the (1+eps)*self + sum(neighbors) update internally.
            self.convs.append(GINConv(_build_gin_mlp(layer_in_dim, hidden_dim), train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            x: [N, in_dim] node features for a batch of graphs.
            edge_index: [2, E] bidirectional edge index.
            batch: [N] graph-membership index (PyG convention) mapping each
                node to its graph in the batch.

        Returns:
            [num_graphs, hidden_dim] pooled graph embeddings (Eq. 2).
        """
        assert x.dim() == 2, f"Expected node features [N, in_dim], got {x.shape}"
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, (
            f"Expected edge_index [2, E], got {edge_index.shape}"
        )
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)       # Eq. 1
            h = bn(h)
            h = self.relu(h)
        graph_embedding = global_add_pool(h, batch)  # Eq. 2: v_g = sum_i h_i
        return graph_embedding

    def __repr__(self) -> str:  # noqa: D105
        return f"GINEncoder(num_layers={self.num_layers}, hidden_dim={self.hidden_dim})"
