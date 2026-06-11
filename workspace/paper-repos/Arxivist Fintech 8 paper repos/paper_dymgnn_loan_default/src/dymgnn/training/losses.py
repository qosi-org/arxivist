"""
training/losses.py — Loss functions for DYMGNN training.

Implements binary cross-entropy loss from Section 3.6, Equation 19:
    Loss = -(1/n) Σ [Y_i log(Ŷ_i) + (1-Y_i) log(1-Ŷ_i)]

Note: Class imbalance (~5% default rate). Standard BCE without reweighting
      is used in the paper; optional class weighting added here for flexibility.

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
Section 3.6, Equation 19.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def binary_cross_entropy_loss(
    y_hat: Tensor,
    y: Tensor,
    pos_weight: float | None = None,
) -> Tensor:
    """Binary cross-entropy loss (Section 3.6, Eq. 19).

    Loss = -(1/n) Σ [Y_i log(Ŷ_i) + (1-Y_i) log(1-Ŷ_i)]

    Args:
        y_hat: Predicted default probabilities [nl, 1] or [nl].
        y:     True labels [nl] (float, 0 or 1).
        pos_weight: Optional weight for positive class (default loan).
            Set to (num_negatives / num_positives) to handle class imbalance.
            NOTE: paper does not mention reweighting; kept optional here.

    Returns:
        Scalar loss tensor.
    """
    y_hat = y_hat.squeeze(-1)  # [nl]
    assert y_hat.shape == y.shape, (
        f"Shape mismatch: y_hat={y_hat.shape}, y={y.shape}"
    )

    if pos_weight is not None:
        weight = torch.ones_like(y)
        weight[y == 1] = pos_weight
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
    else:
        # Eq. 19 — standard BCE
        loss = F.binary_cross_entropy(y_hat, y)

    return loss
