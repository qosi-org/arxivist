"""State preparation utilities (Algorithm 3) for Quantum-SMOTE.

Produces the `phi` (2-dim) and `psi` (2*F dim) vectors used by the
compact swap test. Implements `normalize_array` and `prepare` following
ALGO3_prep in the SIR (rounding as indicated).
"""
from __future__ import annotations

from typing import Tuple
import math

import numpy as np


class StatePreparation:
    """Prepare `phi` and `psi` vectors from two classical vectors.

    Methods
    -------
    prepare(data_point1, data_point2) -> (phi, psi)
        Build phi (shape [2]) and psi (shape [2*F]) according to ALGO3_prep.
    normalize_array(arr) -> arr_normalized
        Scale so sum-of-squares == 1.0.
    """

    @staticmethod
    def normalize_array(arr: np.ndarray) -> np.ndarray:
        """Normalize an array so its sum of squares equals 1.0.

        Args:
            arr: 1D numpy array (float)

        Returns:
            normalized numpy array (dtype float64)
        """
        arr = np.asarray(arr, dtype=float)
        norm2 = float(np.sum(arr ** 2))
        if norm2 <= 0.0:
            raise ValueError("Cannot normalize zero vector")
        factor = math.sqrt(norm2)
        return arr / factor

    @staticmethod
    def prepare(data_point1: np.ndarray, data_point2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct phi [2] and psi [2*F] per ALGO3_prep.

        Pseudocode (ALGO3_prep):
            norm1 = sqrt(sum(dp1^2))
            norm2 = sqrt(sum(dp2^2))
            Z = round(norm1^2 + norm2^2)
            phi = [round(norm1/sqrt(Z), 3), -round(norm2/sqrt(Z), 3)]
            psi = interleaved [dp1[i]/(norm1*sqrt(2)), dp2[i]/(norm2*sqrt(2))] for all i

        After construction, both phi and psi are normalized to unit L2 norm
        (sum-of-squares == 1.0) as required by the compact swap test inputs.

        Args:
            data_point1: numpy array shape [F] (minority sample)
            data_point2: numpy array shape [F] (centroid)

        Returns:
            (phi, psi) where phi.shape == (2,), psi.shape == (2*F,)
        """
        dp1 = np.asarray(data_point1, dtype=float).ravel()
        dp2 = np.asarray(data_point2, dtype=float).ravel()

        if dp1.size != dp2.size:
            raise ValueError("data_point1 and data_point2 must have the same length")

        F = dp1.size

        # norms
        norm1 = float(np.sqrt(np.sum(dp1 ** 2)))
        norm2 = float(np.sqrt(np.sum(dp2 ** 2)))

        if norm1 == 0.0 or norm2 == 0.0:
            raise ValueError("Input vectors must be non-zero")

        # Z (per ALGO3_prep). Keep as float but follow rounding notion.
        Z = float(round(norm1 * norm1 + norm2 * norm2, 6))
        if Z <= 0.0:
            raise ValueError("Invalid normalization constant Z")

        # phi construction with rounding to 3 decimals as in the pseudocode
        phi_0 = round((norm1 / math.sqrt(Z)), 3)
        phi_1 = -round((norm2 / math.sqrt(Z)), 3)
        phi = np.array([phi_0, phi_1], dtype=float)

        # Build interleaved psi: [dp1[0]/(norm1*sqrt(2)), dp2[0]/(norm2*sqrt(2)), dp1[1]/..., dp2[1]/..., ...]
        denom1 = norm1 * math.sqrt(2.0)
        denom2 = norm2 * math.sqrt(2.0)
        psi_list = []
        for i in range(F):
            psi_list.append(dp1[i] / denom1)
            psi_list.append(dp2[i] / denom2)
        psi = np.asarray(psi_list, dtype=float)

        # Normalize both phi and psi to unit norm (sum of squares == 1.0)
        # This follows the SIR requirement that phi and psi be normalized before use.
        phi = phi / math.sqrt(float(np.sum(phi ** 2)))
        psi = psi / math.sqrt(float(np.sum(psi ** 2)))

        return phi, psi
