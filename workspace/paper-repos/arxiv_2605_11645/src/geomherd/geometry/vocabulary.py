"""
src/geomherd/geometry/vocabulary.py
Effective vocabulary (V_eff) via fixed FSQ codebook.
Paper: arXiv:2605.11645, Section 2.4

V_eff(t) = exp(H(p_t))
where H(p_t) is the Shannon entropy of codebook utilization distribution
and the codebook is a fixed 3D grid with Ld=4 levels per dimension (K=64).
Non-learned by design to detect the distribution shift it measures.
"""
from __future__ import annotations

import numpy as np


class FSQVocabularyTracker:
    """
    Tracks the effective action vocabulary using a fixed Finite Scalar Quantization codebook.

    Paper reference: Section 2.4
        V_eff(t) = exp(H(p_t))
        H(p_t) = Shannon entropy of FSQ codebook utilization distribution.
        Codebook: 3D grid, Ld=4 levels per dim, K=64 codewords (non-learned).

    The codebook is a uniform grid over [-1, 1]^3 (ASSUMED placement; paper
    only specifies 3D, Ld=4, K=64).

    Args:
        codebook_dims: Number of dimensions (default 3)
        levels_per_dim: Quantization levels per dimension (default 4)
    """

    def __init__(self, codebook_dims: int = 3, levels_per_dim: int = 4):
        self.D = codebook_dims
        self.L = levels_per_dim
        self.K = levels_per_dim ** codebook_dims
        # Build fixed non-learned codebook: uniform grid in [-1, 1]^D
        # ASSUMED: uniform placement; paper does not specify codebook geometry
        levels = np.linspace(-1.0, 1.0, levels_per_dim)
        grids = np.meshgrid(*[levels] * codebook_dims, indexing="ij")
        self._codebook = np.stack([g.ravel() for g in grids], axis=-1)  # [K, D]
        assert self._codebook.shape == (self.K, self.D), \
            f"Codebook shape mismatch: {self._codebook.shape}"

    def encode(self, action_vectors: np.ndarray) -> np.ndarray:
        """
        Assign each agent's action vector to the nearest codebook entry.

        Args:
            action_vectors: [N, D] float array of agent action representations
        Returns:
            indices: [N] int array of codebook indices in [0, K)
        """
        assert action_vectors.ndim == 2 and action_vectors.shape[1] == self.D, \
            f"Expected [N, {self.D}], got {action_vectors.shape}"
        # Nearest-neighbour lookup via L2 distance
        diffs = action_vectors[:, None, :] - self._codebook[None, :, :]  # [N, K, D]
        dists = np.sum(diffs ** 2, axis=-1)  # [N, K]
        return np.argmin(dists, axis=-1)  # [N]

    def actions_to_vectors(self, actions: np.ndarray, n_assets: int = 1) -> np.ndarray:
        """
        Convert discrete action labels to continuous 3D vectors for FSQ encoding.

        For the CWS substrate with |A|=3 actions {buy=0, hold=1, sell=2}:
        We project via one-hot then embed into the D=3 codebook space.
        ASSUMED: simple one-hot embedding padded to D=3 dims.

        Args:
            actions: [N] int array with values in {0, 1, ..., |A|-1}
            n_assets: number of assets (for multi-asset action encoding)
        Returns:
            vectors: [N, D] float array
        """
        N = len(actions)
        # One-hot embed into 3D space (for |A|=3 this is a direct mapping)
        vectors = np.zeros((N, self.D), dtype=np.float32)
        for k in range(min(self.D, 3)):
            vectors[:, k] = (actions == k).astype(np.float32)
        # Normalize to [-1, 1]
        vectors = vectors * 2.0 - 1.0
        return vectors

    def effective_vocab(self, actions: np.ndarray) -> float:
        """
        Compute V_eff = exp(H(p_t)) for a population of agent actions.

        Paper: Section 2.4 — V_eff(t) = exp(H(p_t))
        V_eff contracts as agents homogenize their behavioral repertoire.

        Args:
            actions: [N] int array of current-step discrete actions
        Returns:
            V_eff: float in [1, K]
        """
        vecs = self.actions_to_vectors(actions)
        indices = self.encode(vecs)  # [N] codebook assignments
        # Compute utilization distribution p_t over K codewords
        counts = np.bincount(indices, minlength=self.K).astype(np.float64)
        total = counts.sum()
        if total == 0:
            return 1.0
        p_t = counts / total
        # Shannon entropy H(p_t)
        nonzero = p_t[p_t > 0]
        H = -np.sum(nonzero * np.log(nonzero))
        # V_eff = exp(H) — effective number of active codewords
        return float(np.exp(H))

    def __repr__(self) -> str:
        return (f"FSQVocabularyTracker(D={self.D}, L={self.L}, K={self.K})")
