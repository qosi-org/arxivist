"""
Activation functions for internal catnat tree nodes.

Implements the two activation functions evaluated in the paper:

1. NaturalActivation ν(x) — the proposed function (Section 4.2.2, Eq. 12)
   Chosen to render the second component of diagonal FIM entries constant,
   i.e. (∂ν/∂s)² * 1/(ν(1-ν)) = (π/A)² everywhere in the active region.
   This yields the simplest diagonal FIM form (Corollary 4.3).

2. SigmoidActivation σ(x) — standard sigmoid baseline (catnat-σ variant)
   Also yields a diagonal FIM (Theorem 4.2) but with non-constant entries.

Paper: Section 4.2.2, Eqs. 11–13, Corollary 4.3.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class NaturalActivation(nn.Module):
    """Natural activation function ν(x) for catnat internal nodes.

    A smooth, bounded S-shaped function defined piecewise:
        ν(x) = 0                              if x ≤ C - A/2
        ν(x) = (1 + sin(π(x-C)/A)) / 2       if C - A/2 ≤ x ≤ C + A/2
        ν(x) = 1                              if x ≥ C + A/2

    The key property: (∂ν/∂s)² * 1/(ν(1-ν)) = (π/A)² in the active region.
    This renders the diagonal FIM entries G_ν(s)_ii = P(a_i) * (π/A)²,
    depending only on the ancestor-reach probability P(a_i), not on s_i.

    Paper: Eq. 12, Corollary 4.3. Confidence: 0.97.

    Args:
        C: Centre/shift parameter. Default 0.0 (scores near zero → prob ≈ 0.5).
           Not a tunable hyperparameter — fixed at initialisation.
        A: Slope parameter. Default 2π, chosen so that
           ∂ν/∂s|_{s=0} = ∂σ/∂s|_{s=0} = 0.25.
           Not a tunable hyperparameter.
    """

    def __init__(self, C: float = 0.0, A: float = 2 * math.pi) -> None:
        super().__init__()
        self.C = C
        self.A = A
        # Precompute π/A for the FIM diagnostic
        self._pi_over_A = math.pi / A

    def forward(self, x: Tensor) -> Tensor:
        """Apply ν(x) elementwise.

        Args:
            x: Input tensor of any shape (unnormalized scores s_i).

        Returns:
            Tensor of same shape, values in [0, 1].
        """
        # Eq. 12 piecewise definition
        # Active region: C - A/2 ≤ x ≤ C + A/2
        half_A = self.A / 2.0
        lower = self.C - half_A
        upper = self.C + half_A

        # Sinusoidal active region value
        active = (1.0 + torch.sin(math.pi * (x - self.C) / self.A)) / 2.0

        # Clamp to [0, 1] outside active region
        result = torch.where(x <= lower, torch.zeros_like(x),
                 torch.where(x >= upper, torch.ones_like(x), active))
        return result

    def fim_diagonal_factor(self, s: Tensor, p_reach: Tensor) -> Tensor:
        """Compute the diagonal FIM entry G_ν(s)_ii per Corollary 4.3.

        G_ν(s)_ii = P(a_i) * (π/A)²  for |s_i - C| < A/2
                  = 0                  outside active region (saturated)

        Paper: Corollary 4.3. Used for diagnostics/visualisation only.

        Args:
            s:        Internal node scores, shape [B, K-1].
            p_reach:  Probability of reaching each node P(a_i), shape [B, K-1].

        Returns:
            Diagonal FIM entries, shape [B, K-1].
        """
        half_A = self.A / 2.0
        in_active = (s - self.C).abs() < half_A
        factor = (self._pi_over_A ** 2) * p_reach
        return torch.where(in_active, factor, torch.zeros_like(factor))

    def __repr__(self) -> str:
        return f"NaturalActivation(C={self.C}, A={self.A:.4f})"


class SigmoidActivation(nn.Module):
    """Standard sigmoid activation for catnat-σ variant.

    σ(x) = 1 / (1 + exp(-x))

    Also yields a diagonal FIM (Theorem 4.2), but with entries that depend
    on the local score s_i:
        G_σ(s)_ii = P(a_i) * (∂σ/∂s_i)² * 1/(σ(s_i)(1-σ(s_i)))
                  = P(a_i)   (since (∂σ/∂x)² / (σ(1-σ)) = 1 for the sigmoid)

    Wait — this actually simplifies to P(a_i) for sigmoid too, since
    ∂σ/∂x = σ(1-σ), so (∂σ/∂x)² / (σ(1-σ)) = σ(1-σ). Not constant in s.
    The difference from ν is that ν achieves (π/A)² = const, whereas σ gives σ(s)(1-σ(s)).

    Paper: Section 4.2.2, catnat-σ baseline variant.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid elementwise.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of same shape, values in (0, 1).
        """
        return torch.sigmoid(x)

    def fim_diagonal_factor(self, s: Tensor, p_reach: Tensor) -> Tensor:
        """Compute diagonal FIM entry for sigmoid activation per Theorem 4.2.

        G_σ(s)_ii = P(a_i) * σ(s_i) * (1 - σ(s_i))

        Args:
            s:        Internal node scores, shape [B, K-1].
            p_reach:  P(a_i), shape [B, K-1].

        Returns:
            Diagonal FIM entries, shape [B, K-1].
        """
        a = torch.sigmoid(s)
        return p_reach * a * (1.0 - a)

    def __repr__(self) -> str:
        return "SigmoidActivation()"


def build_activation(name: str, C: float = 0.0, A: float = 2 * math.pi) -> nn.Module:
    """Factory: return the correct activation module by name.

    Args:
        name: "natural" or "sigmoid".
        C:    Centre parameter for NaturalActivation (ignored for sigmoid).
        A:    Slope parameter for NaturalActivation (ignored for sigmoid).

    Returns:
        An nn.Module implementing the activation.

    Raises:
        ValueError: If name is not recognised.
    """
    if name == "natural":
        return NaturalActivation(C=C, A=A)
    elif name == "sigmoid":
        return SigmoidActivation()
    else:
        raise ValueError(f"Unknown activation '{name}'. Must be 'natural' or 'sigmoid'.")
