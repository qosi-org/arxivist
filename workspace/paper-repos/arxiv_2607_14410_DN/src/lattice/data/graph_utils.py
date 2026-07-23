"""
Graph construction utilities: spatial kNN graph + optional Gaussian edge kernel.

Paper reference: Section 3.2, Eq. 2 (kNN neighborhood) and Eq. 3 (optional
Gaussian kernel edge weight). SIR confidence 0.85 (k=6, union
symmetrization) / 0.55 (Gaussian kernel — sigma never given numerically,
see SIR ambiguities[2]).
"""

from __future__ import annotations

import torch
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(coords: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Build a union-symmetrized k-nearest-neighbor graph from coordinates.

    Args:
        coords: [N, 2] spatial coordinates.
        k: number of nearest neighbors per node (6 in the paper, Eq. 2).

    Returns:
        edge_index: [2, E] long tensor in COO format. Symmetrized via union:
            if i is a kNN of j or j is a kNN of i, the undirected edge (i,j)
            is included once in each direction.
    """
    assert coords.dim() == 2 and coords.shape[1] == 2, (
        f"Expected coords of shape [N, 2], got {tuple(coords.shape)}"
    )
    n = coords.shape[0]
    coords_np = coords.detach().cpu().numpy()

    nn_model = NearestNeighbors(n_neighbors=min(k + 1, n))  # +1: includes self
    nn_model.fit(coords_np)
    _, indices = nn_model.kneighbors(coords_np)

    edges = set()
    for i in range(n):
        for j in indices[i]:
            if i == j:
                continue
            edges.add((i, int(j)))
            edges.add((int(j), i))  # union symmetrization

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return edge_index


def gaussian_kernel_weights(
    coords: torch.Tensor, edge_index: torch.Tensor, sigma: float | None = None
) -> torch.Tensor:
    """Compute Gaussian-kernel edge weights (Eq. 3): w_ij = exp(-||s_i-s_j||^2 / 2*sigma^2).

    Paper reference: Eq. 3. The paper never states a numeric sigma, so if
    `sigma` is None we default to the median pairwise distance among the
    constructed edges, a common heuristic (SIR ambiguities[2], confidence 0.5
    for whether this kernel is even active in the reported runs).

    Args:
        coords: [N, 2] spatial coordinates.
        edge_index: [2, E] long tensor.
        sigma: kernel bandwidth; if None, uses the median edge distance.

    Returns:
        edge_weight: [E] float tensor.
    """
    src, dst = edge_index[0], edge_index[1]
    diffs = coords[src] - coords[dst]  # [E, 2]
    sq_dists = (diffs ** 2).sum(dim=1)  # [E]

    if sigma is None:
        dists = sq_dists.clamp(min=0).sqrt()
        sigma = dists.median().item() if dists.numel() > 0 else 1.0
        sigma = max(sigma, 1e-6)

    return torch.exp(-sq_dists / (2.0 * sigma ** 2))
