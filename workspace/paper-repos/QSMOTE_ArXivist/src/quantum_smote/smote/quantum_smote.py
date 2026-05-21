"""Quantum-SMOTE orchestrator (Algorithm 7).

This module coordinates the full synthetic generation pipeline:
cluster-wise minority sample selection, state preparation, compact swap test,
angle computation, and quantum rotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List
import math
import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from quantum_smote.quantum.state_preparation import StatePreparation
from quantum_smote.quantum.swap_test import CompactSwapTest
from quantum_smote.quantum.angle_calculator import AngleCalculator
from quantum_smote.quantum.rotator import QuantumRotator

logger = logging.getLogger(__name__)


@dataclass
class QuantumSMOTE:
    """Top-level orchestrator for Quantum-SMOTE synthetic generation.

    Parameters
    ----------
    target_pct : int
        Desired minority percentage target used to compute synthetic loop iterations.
    split_factor : int
        Controls the rotation magnitude in AngleCalculator.
    rotation_axis : str
        Preserved for configuration compatibility; RX is used by the rotator implementation.
    angle_increment : float
        Per-loop angle increment (retained for config compatibility; the AngleCalculator
        uses the paper's fixed increment per loop).
    use_statevector : bool
        If True, use exact Aer statevector simulation for swap test.
    shots : int
        Number of shots for shot-based swap test fallback.
    statevector_extraction_strategy : str
        Strategy passed to QuantumRotator.extract_synthetic. `first_F` is the default
        mitigation path for R1.
    """

    target_pct: int = 50
    split_factor: int = 5
    rotation_axis: str = "X"
    angle_increment: float = 0.0174533
    use_statevector: bool = True
    shots: int = 1024
    statevector_extraction_strategy: str = "first_F"

    def __post_init__(self) -> None:
        self.state_prep = StatePreparation()
        self.swap_test = CompactSwapTest(use_statevector=self.use_statevector, shots=self.shots)
        self.angle_calc = AngleCalculator()
        self.rotator = QuantumRotator(statevector_extraction_strategy=self.statevector_extraction_strategy)

    def _synthetic_loop_iterations(self, minority_count: int, cluster_count: int) -> int:
        """Compute synthetic loop iterations per cluster.

        Follows the architecture pseudocode:
            minority_pct = N_min_c / N_cluster
            synthetic_loop_itr = ceil((target_pct - minority_pct) / minority_pct)

        Returns 0 when the cluster has no minority samples or when the cluster is
        already at/above the target percentage.
        """
        if cluster_count <= 0 or minority_count <= 0:
            return 0

        minority_pct = minority_count / float(cluster_count)
        target_fraction = self.target_pct / 100.0

        if minority_pct <= 0.0 or minority_pct >= target_fraction:
            return 0

        loop_itr = math.ceil((target_fraction - minority_pct) / minority_pct)
        return max(0, int(loop_itr))

    def generate_for_cluster(self, minority_X: np.ndarray, centroid: np.ndarray, n_synthetic: int) -> np.ndarray:
        """Generate synthetic samples for a single cluster.

        The method iterates over the minority samples and produces one synthetic
        point per sample per synthetic loop iteration.
        """
        minority_X = np.asarray(minority_X, dtype=float)
        centroid = np.asarray(centroid, dtype=float).ravel()

        if minority_X.ndim != 2:
            raise ValueError("minority_X must be a 2D array")
        if centroid.ndim != 1:
            raise ValueError("centroid must be a 1D array")
        if minority_X.shape[1] != centroid.shape[0]:
            raise ValueError("minority_X and centroid must have the same feature dimension")
        if n_synthetic <= 0 or minority_X.shape[0] == 0:
            return np.empty((0, minority_X.shape[1]), dtype=float)

        synthetic_rows: List[np.ndarray] = []

        progress = tqdm(
            range(n_synthetic),
            desc="Quantum-SMOTE loops",
            leave=False,
            disable=False,
        )
        for syn_loop in progress:
            for sample in minority_X:
                phi, psi = self.state_prep.prepare(sample, centroid)
                swap_prob, angular_dist = self.swap_test.run(psi, phi)
                angle = self.angle_calc.compute(angular_dist, self.split_factor, syn_loop)
                synthetic_point = self.rotator.rotate(sample, angle)
                synthetic_rows.append(synthetic_point)

        if not synthetic_rows:
            return np.empty((0, minority_X.shape[1]), dtype=float)

        return np.asarray(synthetic_rows, dtype=float)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Quantum-SMOTE over all clusters and return augmented arrays.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed feature matrix [N, F].
        y : np.ndarray
            Binary target vector [N].
        cluster_labels : np.ndarray
            Cluster assignment for each row in X, shape [N].
        centroids : np.ndarray
            Cluster centroids, shape [K, F].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X_aug, y_aug)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        cluster_labels = np.asarray(cluster_labels)
        centroids = np.asarray(centroids, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if cluster_labels.ndim != 1:
            raise ValueError("cluster_labels must be a 1D array")
        if centroids.ndim != 2:
            raise ValueError("centroids must be a 2D array")
        if X.shape[0] != y.shape[0] or X.shape[0] != cluster_labels.shape[0]:
            raise ValueError("X, y, and cluster_labels must have the same number of samples")
        if X.shape[1] != centroids.shape[1]:
            raise ValueError("X and centroids must have the same feature dimension")

        synthetic_blocks: List[np.ndarray] = []

        cluster_ids = list(range(centroids.shape[0]))
        cluster_iter = tqdm(cluster_ids, desc="Clusters", leave=False, disable=False)

        for cluster_idx in cluster_iter:
            cluster_mask = cluster_labels == cluster_idx
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]

            if X_cluster.size == 0:
                logger.info("Skipping empty cluster %d", cluster_idx)
                continue

            minority_mask = y_cluster.astype(int) == 1
            minority_X = X_cluster[minority_mask]
            minority_count = int(minority_X.shape[0])
            cluster_count = int(X_cluster.shape[0])

            n_synthetic = self._synthetic_loop_iterations(minority_count, cluster_count)
            if n_synthetic <= 0 or minority_count == 0:
                logger.info(
                    "Cluster %d: no synthetic generation needed (minority=%d, total=%d)",
                    cluster_idx,
                    minority_count,
                    cluster_count,
                )
                continue

            centroid = centroids[cluster_idx]
            synthetic_cluster = self.generate_for_cluster(minority_X, centroid, n_synthetic)

            if synthetic_cluster.size > 0:
                synthetic_blocks.append(synthetic_cluster)

        if synthetic_blocks:
            syn_X = np.vstack(synthetic_blocks)
            syn_y = np.ones(syn_X.shape[0], dtype=y.dtype)
            X_aug = np.vstack([X, syn_X])
            y_aug = np.concatenate([y, syn_y])
        else:
            X_aug = X.copy()
            y_aug = y.copy()

        return X_aug, y_aug
