"""Two-parameter affine post-processing layer that exactly cancels depolarising bias.

Implements: Architecture Plan module `models/postprocessing.py`.
Paper section: Theorem 3.15 ("Exact depolarising-noise cancellation by affine
post-processing") and Remark 3.16.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AffineNoiseCancellation(nn.Module):
    """Affine correction f~^R_{n,theta_bar}(x) = beta1 * f~^R_{n,theta}(x) + beta2.

    Per Theorem 3.15, choosing
        beta1 = 1 / alpha
        beta2 = -beta1 * R * (1 - alpha) * (1 - 4*n_accuracy_blocks / 2^n_qubits)
    makes this layer exactly recover the noiseless QNN output f^R_{n,theta}(x),
    pointwise, given the noisy output f~^R_{n,theta}(x). beta1/beta2 can either be
    set analytically from a known alpha (via `closed_form_correction`), or learned
    jointly with theta via gradient descent (Remark 3.16) -- both are supported here.
    """

    def __init__(self, beta1_init: float = 1.0, beta2_init: float = 0.0) -> None:
        super().__init__()
        self.beta1 = nn.Parameter(torch.tensor(float(beta1_init)))
        self.beta2 = nn.Parameter(torch.tensor(float(beta2_init)))

    def forward(self, f_noisy: torch.Tensor) -> torch.Tensor:
        """Apply the affine correction: beta1 * f_noisy + beta2.

        Args:
            f_noisy: noisy QNN output(s), any shape.

        Returns:
            Corrected output, same shape as f_noisy.
        """
        return self.beta1 * f_noisy + self.beta2

    @staticmethod
    def closed_form_correction(
        alpha: float, R: float, n_accuracy_blocks: int, n_qubits: int
    ) -> tuple[float, float]:
        """Analytic (beta1, beta2) that exactly cancel the depolarising bias (Theorem 3.15).

        Args:
            alpha: hardware fidelity factor alpha = (1-lambda_V)(1-lambda_U).
            R: output scaling factor.
            n_accuracy_blocks: n (number of accuracy blocks).
            n_qubits: number of qubits.

        Returns:
            (beta1, beta2) tuple.
        """
        assert 0.0 < alpha <= 1.0, f"alpha must be in (0,1], got {alpha}"
        beta1 = 1.0 / alpha
        offset_term = 1.0 - (4.0 * n_accuracy_blocks) / (2 ** n_qubits)
        beta2 = -beta1 * R * (1.0 - alpha) * offset_term
        return beta1, beta2

    def load_closed_form(
        self, alpha: float, R: float, n_accuracy_blocks: int, n_qubits: int
    ) -> None:
        """Set this layer's parameters to the analytic (non-learned) correction."""
        beta1, beta2 = self.closed_form_correction(alpha, R, n_accuracy_blocks, n_qubits)
        with torch.no_grad():
            self.beta1.fill_(beta1)
            self.beta2.fill_(beta2)
