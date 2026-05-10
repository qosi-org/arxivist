"""
Stochastic Block Model (SBM) graph generator for Q-Ising experiments.
Implements the SBM described in Section 5.1 of arXiv:2605.06564.
"""
from __future__ import annotations

from typing import List

import numpy as np


def generate_sbm(
    n_per_block: List[int],
    p_in: float,
    p_out: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate a symmetric SBM adjacency matrix.

    Section 5.1: N=500 nodes, 4 blocks [187,187,63,63],
    p_in=0.1 (within-block), p_out=0.01 (between-block).

    Args:
        n_per_block: Number of nodes per block.
        p_in: Within-block edge probability.
        p_out: Between-block edge probability.
        seed: Random seed.

    Returns:
        M: Binary symmetric adjacency matrix [N, N] with zero diagonal.
    """
    rng = np.random.default_rng(seed)
    N = sum(n_per_block)
    M = np.zeros((N, N), dtype=np.int32)

    # Build block membership array
    block_ids = np.repeat(np.arange(len(n_per_block)), n_per_block)

    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if block_ids[i] == block_ids[j] else p_out
            if rng.random() < p:
                M[i, j] = 1
                M[j, i] = 1

    return M


def get_sbm_block_labels(n_per_block: List[int]) -> np.ndarray:
    """Return ground-truth block label for each node.

    Args:
        n_per_block: Number of nodes per block.

    Returns:
        labels: Integer array [N] with block index for each node.
    """
    return np.repeat(np.arange(len(n_per_block)), n_per_block).astype(int)
