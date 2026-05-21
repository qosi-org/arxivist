"""K-Means clustering wrapper for Quantum-SMOTE.

Implements `KMeansClusterer` per the architecture plan. The class is a
thin wrapper around `sklearn.cluster.KMeans` and exposes the required
API: `fit`, `get_labels`, `get_centroids`, and `get_cluster_data`.
"""
from __future__ import annotations

from typing import Tuple, Optional
import logging

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class KMeansClusterer:
    """Wrapper for scikit-learn KMeans used in the Quantum-SMOTE pipeline.

    Parameters
    ----------
    n_clusters: int
        Number of clusters to form.
    init: str
        Method for initialization (default 'k-means++').
    n_init: int
        Number of time the k-means algorithm will be run with different centroid seeds.
    random_state: Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        init: str = "k-means++",
        n_init: int = 10,
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state

        self._model: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        """Fit K-Means to `X` and store labels and centroids.

        Returns
        -------
        self
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        self._model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_.astype(int)
        self.centroids_ = self._model.cluster_centers_.astype(float)

        logger.info("KMeans fitted: n_clusters=%d, n_samples=%d", self.n_clusters, X.shape[0])
        return self

    def get_labels(self) -> np.ndarray:
        """Return cluster labels for the fitted data."""
        if self.labels_ is None:
            raise RuntimeError("KMeansClusterer not fitted. Call fit(X) first.")
        return self.labels_

    def get_centroids(self) -> np.ndarray:
        """Return cluster centroids shape [K, F]."""
        if self.centroids_ is None:
            raise RuntimeError("KMeansClusterer not fitted. Call fit(X) first.")
        return self.centroids_

    def get_cluster_data(self, X: np.ndarray, y: np.ndarray, cluster_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the subset of (X, y) belonging to `cluster_idx`.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix used when fitting (or equivalent ordering).
        y : np.ndarray
            Target array aligned with `X`.
        cluster_idx : int
            Index of the cluster to extract (0 .. n_clusters-1).

        Returns
        -------
        X_cluster, y_cluster : Tuple[np.ndarray, np.ndarray]
            Feature matrix and target vector for samples in the requested cluster.
        """
        if self.labels_ is None:
            raise RuntimeError("KMeansClusterer not fitted. Call fit(X) first.")

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if X_arr.shape[0] != self.labels_.shape[0]:
            raise ValueError("Length of X does not match fitted labels length")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if not (0 <= cluster_idx < self.n_clusters):
            raise IndexError(f"cluster_idx out of range [0,{self.n_clusters-1}]: {cluster_idx}")

        mask = self.labels_ == int(cluster_idx)
        X_cluster = X_arr[mask]
        y_cluster = y_arr[mask]

        return X_cluster, y_cluster
