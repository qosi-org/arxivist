"""
Evaluation metrics: ARI, NMI, spatial contiguity, silhouette, and the
Multimodal Utility Score (MUS).

Paper reference: Section 4.1 ("Dataset and evaluation setup"), Appendix A.2
("Evaluation Metrics"), Eq. 11 (MUS). SIR confidence 0.9 (evaluation
protocol is very explicit in the paper).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def ari_nmi(pred_labels: np.ndarray, reference_labels: np.ndarray) -> dict[str, float]:
    """Adjusted Rand Index and Normalized Mutual Information vs. a reference clustering.

    Paper reference: Section 4.1 — ARI/NMI compare LATTICE (or baseline)
    clusters with Space Ranger RNA-derived clusters, used as a
    transcriptomic reference (not ground truth).

    Args:
        pred_labels: [N] integer cluster assignment from the embedding under
            evaluation.
        reference_labels: [N] integer reference cluster assignment (e.g.
            Space Ranger RNA clusters, or in this synthetic-data repro, the
            synthetic domain labels).

    Returns:
        {'ari': float, 'nmi': float}
    """
    return {
        "ari": float(adjusted_rand_score(reference_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(reference_labels, pred_labels)),
    }


def spatial_contiguity(coords: np.ndarray, labels: np.ndarray, k: int = 6) -> float:
    """Fraction of a spot's k spatial neighbors sharing its cluster label.

    Paper reference: Section 4.1, "Spatial contiguity measures whether
    neighboring spots receive coherent labels."

    Args:
        coords: [N, 2] spatial coordinates.
        labels: [N] integer cluster labels.
        k: number of spatial neighbors to check (matches the k=6 graph).

    Returns:
        mean same-label fraction across all spots and their k neighbors.
    """
    n = coords.shape[0]
    nn_model = NearestNeighbors(n_neighbors=min(k + 1, n))
    nn_model.fit(coords)
    _, indices = nn_model.kneighbors(coords)

    same_label_fracs = []
    for i in range(n):
        neighbor_idx = [j for j in indices[i] if j != i][:k]
        if not neighbor_idx:
            continue
        same = sum(labels[j] == labels[i] for j in neighbor_idx)
        same_label_fracs.append(same / len(neighbor_idx))
    return float(np.mean(same_label_fracs)) if same_label_fracs else 0.0


def embedding_silhouette(z: np.ndarray, labels: np.ndarray) -> float:
    """Cosine silhouette score of embeddings w.r.t. cluster labels.

    Paper reference: Appendix A.2, "Silhouette_v the mean cosine silhouette
    of embeddings with respect to Leiden clusters."

    Args:
        z: [N, hidden_dim] embeddings.
        labels: [N] integer cluster labels.

    Returns:
        silhouette score in [-1, 1], or 0.0 if fewer than 2 clusters are present.
    """
    if len(set(labels.tolist())) < 2:
        return 0.0
    return float(silhouette_score(z, labels, metric="cosine"))


def bio_knn_consistency(z: np.ndarray, labels: np.ndarray, k: int = 6) -> float:
    """Same-cluster neighbor consistency in embedding space (BioKNN_v, Appendix A.2)."""
    n = z.shape[0]
    nn_model = NearestNeighbors(n_neighbors=min(k + 1, n))
    nn_model.fit(z)
    _, indices = nn_model.kneighbors(z)

    fracs = []
    for i in range(n):
        neighbor_idx = [j for j in indices[i] if j != i][:k]
        if not neighbor_idx:
            continue
        same = sum(labels[j] == labels[i] for j in neighbor_idx)
        fracs.append(same / len(neighbor_idx))
    return float(np.mean(fracs)) if fracs else 0.0


def bio_jaccard_overlap(z: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    """Mean Jaccard overlap between embedding-kNN and spatial-kNN neighbor sets (BioJaccard_v)."""
    n = z.shape[0]
    z_nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(z)
    s_nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, z_idx = z_nn.kneighbors(z)
    _, s_idx = s_nn.kneighbors(coords)

    jaccards = []
    for i in range(n):
        set_z = set(j for j in z_idx[i] if j != i)
        set_s = set(j for j in s_idx[i] if j != i)
        union = set_z | set_s
        if not union:
            continue
        jaccards.append(len(set_z & set_s) / len(union))
    return float(np.mean(jaccards)) if jaccards else 0.0


def _min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a dict of scalar values across "rows" for MUS (Eq. 11)."""
    vals = np.array(list(values.values()))
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def multimodal_utility_score(
    spot_cut_by_row: dict[str, float],
    silhouette_by_row: dict[str, float],
    bio_knn_by_row: dict[str, float],
    bio_jaccard_by_row: dict[str, float],
) -> dict[str, float]:
    """Multimodal Utility Score (Eq. 11), computed jointly across an evaluation pool.

    MUS_v = 1/4 * (SpotCut_v^norm + Silhouette_v^norm + BioKNN_v^norm + BioJaccard_v^norm)

    Each raw metric is min-max normalized ACROSS ALL ROWS in the evaluation
    pool (e.g. all baselines + all LATTICE modality-ladder variants) before
    averaging, per Appendix A.2. Pass in one dict per metric, keyed by row
    name (e.g. "LATTICE_M1", "GraphST_M1", ...) so normalization is joint.

    Args:
        spot_cut_by_row: {row_name: spatial_contiguity_value}
        silhouette_by_row: {row_name: silhouette_value}
        bio_knn_by_row: {row_name: bio_knn_value}
        bio_jaccard_by_row: {row_name: bio_jaccard_value}

    Returns:
        {row_name: MUS_value} for every row present in all four input dicts.
    """
    rows = set(spot_cut_by_row) & set(silhouette_by_row) & set(bio_knn_by_row) & set(bio_jaccard_by_row)
    spot_cut_n = _min_max_normalize(spot_cut_by_row)
    sil_n = _min_max_normalize(silhouette_by_row)
    knn_n = _min_max_normalize(bio_knn_by_row)
    jac_n = _min_max_normalize(bio_jaccard_by_row)

    return {
        row: 0.25 * (spot_cut_n[row] + sil_n[row] + knn_n[row] + jac_n[row]) for row in rows
    }
