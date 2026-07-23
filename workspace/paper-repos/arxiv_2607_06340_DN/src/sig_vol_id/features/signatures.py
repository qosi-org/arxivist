"""
Truncated path signature computation (Section 2, 3).

Paths are time-augmented (X_hat_t = (t, X_t)) before computing the signature,
per Section 2.2, so that the feature map is injective and speed-sensitive.

Uses the `iisignature` package (CPU) for a correct, well-tested reference
implementation. The paper uses a custom GPU-adapted implementation instead
(see SIR ambiguities[2]); iisignature produces mathematically identical
truncated signatures, just without GPU acceleration.
"""

from __future__ import annotations

import numpy as np

try:
    import iisignature
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "iisignature is required for signature computation. Install with: "
        "pip install iisignature"
    ) from e


class SignatureComputer:
    """Computes time-augmented truncated path signatures.

    Args:
        order: Truncation level N (paper default: 4; also tests 3 and 5).
    """

    def __init__(self, order: int = 4):
        self.order = order

    def compute(self, paths: np.ndarray, T: float = 0.1) -> np.ndarray:
        """Compute the flattened truncated signature for a batch of paths.

        Args:
            paths: [n_paths, n_steps+1] array of raw (1-dimensional) path values.
            T: Time horizon spanned by the path (for the time-augmentation channel).

        Returns:
            [n_paths, d_N] array of flattened truncated signatures, where
            d_N = sum_{k=0}^{order} 2^k (2 channels: time, path value).
        """
        n_paths, n_points = paths.shape
        times = np.linspace(0, T, n_points)

        # Time-augment: each path becomes a 2D path (t, X_t)
        augmented = np.empty((n_paths, n_points, 2), dtype=np.float64)
        augmented[:, :, 0] = times[None, :]
        augmented[:, :, 1] = paths

        sigs = iisignature.sig(augmented, self.order)
        # iisignature.sig returns only the non-constant terms (levels 1..order);
        # prepend the constant "1" term (level 0) to match the paper's vec(S(X)<=N)
        # which explicitly includes the leading 1.
        ones = np.ones((n_paths, 1), dtype=np.float64)
        return np.hstack([ones, sigs])

    @staticmethod
    def dim_for_order(order: int, channels: int = 2) -> int:
        """Returns d_N = sum_{k=0}^{order} channels^k."""
        return sum(channels**k for k in range(order + 1))
