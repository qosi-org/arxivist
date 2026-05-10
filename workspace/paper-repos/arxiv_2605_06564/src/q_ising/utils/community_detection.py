"""
Community detection utilities for Q-Ising bin construction.
Uses edge-betweenness clustering as specified in Appendix E.3 of arXiv:2605.06564.
"""
from __future__ import annotations

from typing import List

import numpy as np


def adjacency_to_igraph(M: np.ndarray):
    """Convert a NumPy adjacency matrix to an igraph Graph.

    Args:
        M: Binary adjacency matrix of shape [N, N].

    Returns:
        igraph.Graph object.
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError("python-igraph is required. Install with: pip install python-igraph")

    N = M.shape[0]
    edges = list(zip(*np.where(np.triu(M, k=1) > 0)))
    g = ig.Graph(n=N, edges=edges)
    return g


def detect_communities_edge_betweenness(
    M: np.ndarray,
    min_size: int = 10,
) -> np.ndarray:
    """Detect communities using edge-betweenness clustering (Appendix E.3).

    Small communities (< min_size nodes) are merged into the largest community,
    as described in Appendix E.3 of arXiv:2605.06564.

    Args:
        M: Binary adjacency matrix [N, N].
        min_size: Communities with fewer nodes are merged into the largest. Default 10.

    Returns:
        community_labels: Integer array [N] with bin assignment for each node.
    """
    g = adjacency_to_igraph(M)

    # Edge-betweenness community detection (Appendix E.3)
    dendrogram = g.community_edge_betweenness()
    membership = dendrogram.as_clustering().membership
    labels = np.array(membership, dtype=int)

    labels = merge_small_communities(labels, min_size=min_size)
    # Re-index to 0..K-1
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=int)

    return labels


def merge_small_communities(labels: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Merge communities smaller than min_size into the largest community.

    As specified in Appendix E.3: "If an identified cluster has less than 10 nodes,
    these nodes are considered to be part of the largest cluster."

    Args:
        labels: Community label array [N].
        min_size: Minimum community size threshold.

    Returns:
        Updated labels array with small communities merged.
    """
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    largest_community = unique[np.argmax(counts)]

    for comm, count in zip(unique, counts):
        if count < min_size:
            labels[labels == comm] = largest_community

    return labels


def spectral_bin_assignment(M: np.ndarray, K: int) -> np.ndarray:
    """Assign nodes to K bins via spectral clustering.

    Used for SBM experiments where ground-truth block structure is recovered.
    ASSUMED: spectral clustering used for SBM (confidence: 0.78).

    Args:
        M: Adjacency matrix [N, N].
        K: Number of desired bins.

    Returns:
        bin_labels: Integer array [N] with values in [0, K-1].
    """
    from sklearn.cluster import SpectralClustering

    sc = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    )
    labels = sc.fit_predict(M.astype(float))
    return labels.astype(int)


def assign_bins(M: np.ndarray, method: str, K: int = None, min_size: int = 10) -> np.ndarray:
    """Unified bin assignment dispatcher.

    Args:
        M: Adjacency matrix [N, N].
        method: "edge_betweenness" | "spectral".
        K: Number of bins (required for spectral; inferred for edge_betweenness).
        min_size: Min community size for edge_betweenness merge step.

    Returns:
        bin_labels: Integer array [N].
    """
    if method == "edge_betweenness":
        return detect_communities_edge_betweenness(M, min_size=min_size)
    elif method == "spectral":
        if K is None:
            raise ValueError("K must be specified for spectral bin assignment")
        return spectral_bin_assignment(M, K=K)
    else:
        raise ValueError(f"Unknown bin method: {method!r}. Choose 'edge_betweenness' or 'spectral'.")
