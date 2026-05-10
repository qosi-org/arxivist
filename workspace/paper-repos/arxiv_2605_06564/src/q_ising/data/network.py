"""
Network data container for Q-Ising.
Encapsulates adjacency matrix, node features, and bin assignments.

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 2.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class NetworkData:
    """Fixed undirected network with node features and bin assignments.

    Corresponds to the fixed network M ∈ {0,1}^{N×N} and node features
    X ∈ R^{N×d_x} described in Section 2 of arXiv:2605.06564.

    Args:
        M: Binary adjacency matrix [N, N]. Must be symmetric with zero diagonal.
        X: Node feature matrix [N, d_x]. May be None if features are unused.
        bin_labels: Integer array [N] assigning each node to a bin in [0, K-1].
    """

    def __init__(
        self,
        M: np.ndarray,
        X: Optional[np.ndarray] = None,
        bin_labels: Optional[np.ndarray] = None,
    ) -> None:
        assert M.ndim == 2 and M.shape[0] == M.shape[1], \
            f"M must be square 2D, got shape {M.shape}"
        assert np.allclose(M, M.T), "Adjacency matrix M must be symmetric"
        assert np.all(np.diag(M) == 0), "Adjacency matrix must have zero diagonal"

        self.M = M.astype(np.int32)
        self.N = M.shape[0]
        self.X = X
        self.bin_labels = bin_labels

        # Precompute neighbor lists for efficiency
        self._neighbors: List[List[int]] = [
            list(np.where(M[i] > 0)[0]) for i in range(self.N)
        ]

        if bin_labels is not None:
            self.K = int(bin_labels.max()) + 1
            self._bins: List[List[int]] = [
                list(np.where(bin_labels == k)[0]) for k in range(self.K)
            ]
        else:
            self.K = None
            self._bins = None

    def get_neighbors(self, node: int) -> List[int]:
        """Return list of neighbor indices for a given node.

        Args:
            node: Node index in [0, N-1].

        Returns:
            List of neighbor indices.
        """
        return self._neighbors[node]

    def get_bin(self, node: int) -> int:
        """Return bin index for a given node.

        Args:
            node: Node index.

        Returns:
            Bin index in [0, K-1].
        """
        assert self.bin_labels is not None, "Bin labels not assigned. Call assign_bins() first."
        return int(self.bin_labels[node])

    def get_bin_members(self, k: int) -> List[int]:
        """Return list of node indices in bin k.

        Args:
            k: Bin index.

        Returns:
            List of node indices.
        """
        assert self._bins is not None, "Bin labels not assigned."
        return self._bins[k]

    def assign_bins(self, method: str, K: int = None, min_size: int = 10) -> None:
        """Perform community detection and assign bins in-place.

        Args:
            method: "spectral" | "edge_betweenness".
            K: Number of bins (required for spectral).
            min_size: Minimum bin size for edge_betweenness (Appendix E.3).
        """
        from q_ising.utils.community_detection import assign_bins
        self.bin_labels = assign_bins(self.M, method=method, K=K, min_size=min_size)
        self.K = int(self.bin_labels.max()) + 1
        self._bins = [
            list(np.where(self.bin_labels == k)[0]) for k in range(self.K)
        ]

    def __repr__(self) -> str:
        return (
            f"NetworkData(N={self.N}, K={self.K}, "
            f"edges={self.M.sum() // 2}, bin_method={'set' if self.bin_labels is not None else 'not set'})"
        )
