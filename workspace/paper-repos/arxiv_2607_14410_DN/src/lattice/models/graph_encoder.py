"""
Spatial graph builder and TransformerConv-based graph encoder.

Paper reference: Section 3.2 ("Spatial graph construction"), Eq. 2-3;
Section 3.3 ("Graph encoder..."); Appendix H ("Encoder architecture").
SIR modules: `spatial_knn_graph_builder` (confidence 0.85),
`transformer_conv_encoder_stack` (confidence 0.95).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import LayerNorm as PyGLayerNorm
from torch_geometric.nn import TransformerConv

from lattice.data.graph_utils import build_knn_graph, gaussian_kernel_weights


class SpatialGraphBuilder:
    """Builds a spatial k-nearest-neighbor graph from spot coordinates.

    Paper reference: Eq. 2 (kNN neighborhood), Eq. 3 (optional Gaussian
    kernel edge weight). SIR confidence 0.85 for k=6 + union symmetrization;
    0.5 for whether the Gaussian kernel was actually enabled in the reported
    runs (sigma is never given a numeric value) — see SIR ambiguities[2].

    Args:
        k: number of nearest neighbors (6 in the paper).
        edge_weight_mode: "uniform" (default, ASSUMED) or "gaussian" (Eq. 3).
        gaussian_sigma: sigma for the Gaussian kernel; if None and
            edge_weight_mode="gaussian", defaults to the median pairwise
            spatial distance among constructed edges.
    """

    def __init__(
        self,
        k: int = 6,
        edge_weight_mode: str = "uniform",
        gaussian_sigma: float | None = None,
    ) -> None:
        if edge_weight_mode not in ("uniform", "gaussian"):
            raise ValueError(f"edge_weight_mode must be 'uniform' or 'gaussian', got {edge_weight_mode!r}")
        self.k = k
        self.edge_weight_mode = edge_weight_mode
        self.gaussian_sigma = gaussian_sigma

    def build(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the spatial kNN graph.

        Args:
            coords: [N, 2] spatial coordinates (e.g. Visium array coords).

        Returns:
            edge_index: [2, E] long tensor (COO format, union-symmetrized).
            edge_weight: [E] float tensor, or None if edge_weight_mode="uniform".
        """
        assert coords.dim() == 2 and coords.shape[1] == 2, (
            f"Expected coords of shape [N, 2], got {tuple(coords.shape)}"
        )
        edge_index = build_knn_graph(coords, k=self.k)
        edge_weight = None
        if self.edge_weight_mode == "gaussian":
            edge_weight = gaussian_kernel_weights(coords, edge_index, sigma=self.gaussian_sigma)
        return edge_index, edge_weight


class LatticeGraphEncoder(nn.Module):
    """Stack of TransformerConv layers producing spot-level embeddings Z.

    Paper reference: Appendix H, "Encoder architecture" — 3 TransformerConv
    layers, 4 attention heads, hidden dim 128, dropout 0.1, ReLU activation,
    LayerNorm. SIR confidence 0.95 (fully explicit).

    Args:
        hidden_dim: input AND output embedding dimension (128 in the paper;
            TransformerConv is configured with heads*out_channels == hidden_dim
            via concat=True so each layer preserves this width).
        num_layers: number of TransformerConv layers (3).
        num_heads: number of attention heads per layer (4).
        dropout: dropout rate (0.1).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}) "
            "so TransformerConv(concat=True) preserves width across layers."
        )
        out_channels_per_head = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=out_channels_per_head,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=1,  # allows passing scalar edge_weight through the attention layer
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([PyGLayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode fused node features into spot-level embeddings.

        Args:
            x: [N, hidden_dim] fused modality features.
            edge_index: [2, E] long tensor.
            edge_weight: optional [E] float tensor of edge weights (Eq. 3);
                None uses an unweighted/uniform graph (default assumption).

        Returns:
            z: [N, hidden_dim] node embeddings.
        """
        assert x.dim() == 2 and x.shape[1] == self.hidden_dim, (
            f"Expected [N, {self.hidden_dim}], got {tuple(x.shape)}"
        )
        # TransformerConv is configured with edge_dim=1 (so weighted graphs, Eq. 3,
        # can flow through the attention mechanism when enabled). When no edge
        # weight is supplied (default "uniform" mode, SIR ambiguities[2]), we pass
        # a constant edge_attr of 1.0 rather than None, since edge_dim=1 requires
        # edge_attr to be present.
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)
        edge_attr = edge_weight.unsqueeze(-1)
        z = x
        for conv, norm in zip(self.convs, self.norms):
            z = conv(z, edge_index, edge_attr=edge_attr)
            z = norm(z)
            z = self.activation(z)
            z = self.dropout(z)
        return z

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LatticeGraphEncoder(hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers})"
        )
