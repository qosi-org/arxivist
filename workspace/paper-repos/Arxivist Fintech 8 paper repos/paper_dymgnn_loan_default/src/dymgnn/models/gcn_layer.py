"""
models/gcn_layer.py — Graph Convolutional Network layer.

Implements the GCN formulation from Section 3.2, Equation 1:
    Z = D̃^{-1/2} Ã D̃^{-1/2} X W^T
where Ã = A + I_{nl} (adjacency with self-loops).

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
Reference: Kipf & Welling (2017) "Semi-supervised classification with GCNs"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GCNLayer(nn.Module):
    """Single GCN layer with spectral graph convolution (Section 3.2, Eq. 1).

    Performs isotropic message passing: each neighbour contributes equally
    to update the central node representation.

    Args:
        in_features: Input feature dimension d (number of node features).
        out_features: Output embedding dimension D.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # W^T ∈ R^{d×D} — learnable weight matrix (Section 3.2)
        self.weight = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Apply GCN convolution to one snapshot.

        Implements Eq. 1: Z = D̃^{-1/2} Ã D̃^{-1/2} X W^T

        Args:
            x: Node feature matrix [nl, d] or [nl, in_features].
            adj: Adjacency matrix [nl, nl] (0/1, WITHOUT self-loops; added here).

        Returns:
            Z: Node embedding matrix [nl, D].
        """
        assert x.dim() == 2, f"Expected [nl, d], got {x.shape}"
        assert adj.dim() == 2, f"Expected [nl, nl], got {adj.shape}"
        assert adj.shape[0] == adj.shape[1] == x.shape[0], (
            f"adj shape {adj.shape} must match x.shape[0]={x.shape[0]}"
        )

        nl = x.shape[0]
        # Ã = A + I_{nl}  (add self-loops, Section 3.2)
        adj_hat = adj + torch.eye(nl, device=x.device, dtype=adj.dtype)

        # D̃_{ii} = Σ_j Ã_{ij}  (degree matrix)
        deg = adj_hat.sum(dim=1)                         # [nl]
        # D̃^{-1/2}  (symmetric normalisation)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0  # guard isolated nodes
        D_inv_sqrt = torch.diag(deg_inv_sqrt)            # [nl, nl]

        # Normalised adjacency: D̃^{-1/2} Ã D̃^{-1/2}
        norm_adj = D_inv_sqrt @ adj_hat @ D_inv_sqrt     # [nl, nl]

        # Z = norm_adj X W^T  (Eq. 1)
        out = norm_adj @ self.weight(x)                  # [nl, D]
        return out

    def __repr__(self) -> str:
        return f"GCNLayer(in={self.in_features}, out={self.out_features})"
