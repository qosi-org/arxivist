"""
Bandwidth rescaling x~ = lambda*x (Sec 4.2, Sec 8).

Governs spectral decay: large lambda concentrates the kernel toward the
identity, small lambda toward a constant (Shaydulin & Wild 2022; Canatar
et al. 2023) -- the exponential-concentration phenomenon (Thanasilp et al.
2024) the paper's Sec 8 bandwidth diagnostics investigate directly.
"""
from __future__ import annotations

import numpy as np


class BandwidthScaler:
    """Clips inputs to [-3,3] then rescales by the bandwidth lambda."""

    def scale(self, x: np.ndarray, lam: float, clip: float = 3.0) -> np.ndarray:
        """
        Args:
            x: [..., num_qubits] raw (already cross-sectionally standardized) inputs.
            lam: bandwidth.
            clip: symmetric clip range applied before scaling (3.0, per Sec 4.2).

        Returns:
            x~ = lam * clip(x, -clip, clip), same shape as x.
        """
        return lam * np.clip(x, -clip, clip)

    def __repr__(self) -> str:  # noqa: D105
        return "BandwidthScaler()"
