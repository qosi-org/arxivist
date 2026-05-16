"""
src/geomherd/graph/agent_graph.py
Dynamic agent-action graph construction.
Paper: arXiv:2605.11645, Section 2.1 (Eq. 1)

Builds a weighted sparse graph G_t = (V, E_t, w_t) where each edge weight
w_t(i,j) is the windowed action-agreement frequency between agents i and j.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


class AgentGraph:
    """
    Maintains a rolling windowed action-agreement graph over N agents.

    Paper reference: Section 2.1, Eq. 1
        w_t(i,j) = (1/Tw) * sum_{s=t-Tw+1}^{t} 1[a_i(s) = a_j(s)]

    Edges are retained only if w_t(i,j) > w0 (sparsification threshold).
    The graph is reconstructed every delta_t steps with 50% temporal overlap.

    Args:
        N: Number of agents
        Tw: Window length for agreement frequency (default: 100)
        w0: Sparsification threshold (default: 0.5)
        delta_t: Snapshot reconstruction stride (default: 10)
    """

    def __init__(self, N: int, Tw: int = 100, w0: float = 0.5, delta_t: int = 10):
        assert N > 0, f"N must be positive, got {N}"
        assert Tw > 0, f"Tw must be positive, got {Tw}"
        assert 0 < w0 < 1, f"w0 must be in (0,1), got {w0}"
        self.N = N
        self.Tw = Tw
        self.w0 = w0
        self.delta_t = delta_t
        # Rolling buffer of action arrays: deque of shape (N,) int arrays
        self._buffer: deque = deque(maxlen=Tw)
        self._t: int = 0
        self._last_snapshot_t: int = -1
        self._current_W: Optional[np.ndarray] = None  # [N, N] dense weight matrix

    def push(self, actions: np.ndarray) -> bool:
        """
        Push one step's actions into the rolling buffer.

        Args:
            actions: int array of shape [N], each in {0, 1, 2} (buy/hold/sell)
        Returns:
            True if a new snapshot was computed at this step, False otherwise
        """
        assert actions.shape == (self.N,), \
            f"Expected actions shape ({self.N},), got {actions.shape}"
        self._buffer.append(actions.copy())
        self._t += 1
        # Reconstruct every delta_t steps once buffer has at least 1 sample
        if (self._t % self.delta_t == 0) and len(self._buffer) > 0:
            self._compute_snapshot()
            self._last_snapshot_t = self._t
            return True
        return False

    def _compute_snapshot(self) -> None:
        """Compute w_t(i,j) for all pairs from the current buffer. Eq. 1."""
        buf = np.stack(list(self._buffer), axis=1)  # [N, len_buf]
        T_actual = buf.shape[1]
        # w[i,j] = mean over window of 1[a_i(s) == a_j(s)]
        # Efficient: for each pair compute agreement frequency
        # For N=66 this is a 66x66 outer-equal comparison, manageable
        W = np.zeros((self.N, self.N), dtype=np.float32)
        for s in range(T_actual):
            col = buf[:, s]  # [N]
            # Outer equality: W[i,j] += 1 if col[i] == col[j]
            match = (col[:, None] == col[None, :]).astype(np.float32)
            W += match
        W /= T_actual  # normalize to frequency
        np.fill_diagonal(W, 0.0)  # remove self-loops (w(i,i)=0)
        self._current_W = W

    def get_weight_matrix(self) -> np.ndarray:
        """Return the current [N, N] dense weight matrix (before sparsification)."""
        if self._current_W is None:
            return np.zeros((self.N, self.N), dtype=np.float32)
        return self._current_W.copy()

    def get_sparse_graph(self) -> sp.csr_matrix:
        """
        Return the sparsified graph as a CSR matrix.
        Edges with w_t(i,j) <= w0 are zeroed out.

        Paper: Section 2.1 — sparsify by retaining edges with w_t(i,j) > w0 = 0.5
        """
        W = self.get_weight_matrix()
        W_sparse = W.copy()
        W_sparse[W_sparse <= self.w0] = 0.0
        return sp.csr_matrix(W_sparse)

    def get_edge_list(self) -> List[Tuple[int, int, float]]:
        """Return list of (i, j, weight) for all retained edges (upper triangle)."""
        G = self.get_sparse_graph()
        G_coo = G.tocoo()
        edges = []
        for i, j, w in zip(G_coo.row, G_coo.col, G_coo.data):
            if i < j:  # upper triangle only (undirected)
                edges.append((int(i), int(j), float(w)))
        return edges

    def n_edges(self) -> int:
        """Number of retained edges in the current snapshot."""
        return len(self.get_edge_list())

    def reset(self) -> None:
        """Clear buffer and state."""
        self._buffer.clear()
        self._t = 0
        self._last_snapshot_t = -1
        self._current_W = None

    def __repr__(self) -> str:
        return (f"AgentGraph(N={self.N}, Tw={self.Tw}, w0={self.w0}, "
                f"delta_t={self.delta_t}, t={self._t}, "
                f"n_edges={self.n_edges() if self._current_W is not None else 'n/a'})")
