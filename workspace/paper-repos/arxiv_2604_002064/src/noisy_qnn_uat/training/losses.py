"""Training loss used to fit theta (and, when applicable, beta1/beta2).

Implements: Architecture Plan module `training/losses.py`.
Paper section: Section 4.1, "Parameters optimisation" (loss function L(theta)).

NOTE (flagged per SIR ambiguities[0] / mathematical_spec confidence 0.5): the
exact normalisation constant of L(theta) is ambiguous because the source PDF's
text extraction corrupted the formula's exponents/summation bounds. This module
uses the standard MSE convention L(theta) = 1/(2n) * sum (pred-target)^2, exposed
as a configurable multiplier so it can be swapped without touching the optimiser
(the argmin over theta is invariant to the choice of overall constant).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNNMSELoss(nn.Module):
    """Mean-squared error loss between QNN predictions and target prices/densities.

    L(theta) = normalisation_constant * sum_{i=1}^{n} (f^R_n(x_i;theta) - P_i)^2

    Args:
        normalisation: multiplier applied to the summed squared error. Default
            1/(2n) is applied dynamically based on the batch size n at forward
            time (ASSUMED convention, see module docstring); pass a fixed float
            instead if you want to test the "1/(2n^2)" or unnormalised
            alternatives listed in the SIR ambiguities.
    """

    def __init__(self, normalisation: str | float = "1/(2n)") -> None:
        super().__init__()
        self.normalisation = normalisation

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the (possibly-normalised) sum of squared errors.

        Args:
            predictions: model outputs, shape [n].
            targets: ground-truth values, shape [n].

        Returns:
            Scalar loss tensor.
        """
        assert predictions.shape == targets.shape, (
            f"Shape mismatch: predictions={predictions.shape}, targets={targets.shape}"
        )
        squared_error_sum = torch.sum((predictions - targets) ** 2)
        n = predictions.shape[0]

        if isinstance(self.normalisation, (int, float)):
            const = float(self.normalisation)
        elif self.normalisation == "1/(2n)":
            const = 1.0 / (2 * n)
        elif self.normalisation == "1/(2n^2)":
            const = 1.0 / (2 * n * n)
        elif self.normalisation == "none":
            const = 1.0
        else:
            raise ValueError(f"Unknown normalisation convention: {self.normalisation}")

        return const * squared_error_sum
