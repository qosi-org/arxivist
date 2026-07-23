"""
FIM diagnostic tools for verifying theoretical properties of catnat.

Implements:
  - Empirical verification that the catnat FIM is diagonal (Theorem 4.2)
  - Comparison of loss landscapes between softmax and catnat (Figure 1)

These are analysis utilities, not required for training.
Paper: Section 4, Figure 1, Theorem 4.2, Corollary 4.3.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor


class FIMAnalyzer:
    """Tools for computing and comparing Fisher Information Matrices.

    Verifies Theorem 4.2: that the catnat FIM is diagonal.
    Compares the FIM structure of catnat vs softmax.
    """

    @staticmethod
    def compute_diagonal(model, s: Tensor) -> Tensor:
        """Compute diagonal FIM entries analytically via Theorem 4.2.

        For catnat: G_a(s)_ii = P(a_i) * (∂a_i/∂s_i)² * 1/(a_i*(1-a_i))
        Delegates to CatNat.fim_diagonal().

        Args:
            model: CatNat model.
            s:     Scores, shape [B, K-1].

        Returns:
            Diagonal entries, shape [B, K-1].
        """
        if not hasattr(model, "fim_diagonal"):
            raise ValueError("model must be a CatNat instance with fim_diagonal() method.")
        return model.fim_diagonal(s)

    @staticmethod
    def compute_empirical_fim(model, s: Tensor, n_samples: int = 1000) -> Tensor:
        """Estimate the FIM empirically via Monte Carlo sampling.

        FIM = E_{C ~ Cat(p)}[(∇_s log p(C|s)) (∇_s log p(C|s))^T]

        Used to numerically verify that the analytical FIM (Theorem 4.2) is correct.

        Args:
            model:     CatNat or SoftmaxParam model.
            s:         Scores, shape [1, K-1] or [1, K] (single example).
            n_samples: Number of Monte Carlo samples.

        Returns:
            Estimated FIM, shape [S, S] where S = K-1 or K.
        """
        s = s.detach().requires_grad_(True)
        K_minus_1 = s.shape[-1]

        outer_sum = torch.zeros(K_minus_1, K_minus_1)

        for _ in range(n_samples):
            # Forward pass
            if hasattr(model, "log_prob"):
                log_p = model.log_prob(s)       # [1, K]
            else:
                log_p = torch.log(model(s).clamp(min=1e-8))

            # Sample category
            with torch.no_grad():
                cat = torch.distributions.Categorical(logits=log_p.squeeze(0))
                k = cat.sample()

            # Log-prob of sampled category
            log_pk = log_p[0, k]

            # Gradient
            if s.grad is not None:
                s.grad.zero_()
            log_pk.backward(retain_graph=True)
            g = s.grad.squeeze(0).detach()     # [K-1]

            outer_sum += torch.outer(g, g)

        return outer_sum / n_samples

    @staticmethod
    def verify_diagonal(catnat_model, K: int, n_test: int = 10) -> Dict:
        """Verify that catnat's FIM is diagonal for random score inputs.

        Computes empirical FIM and checks that off-diagonal entries are ~0.

        Args:
            catnat_model: CatNat model.
            K:            Number of categories.
            n_test:       Number of random score inputs to test.

        Returns:
            Dict with 'max_offdiag_abs' and 'diag_mean'.
        """
        max_offdiag = 0.0
        diag_vals = []

        for _ in range(n_test):
            s = torch.randn(1, K - 1)
            fim = FIMAnalyzer.compute_empirical_fim(catnat_model, s, n_samples=500)
            # Off-diagonal entries
            mask = 1.0 - torch.eye(K - 1)
            offdiag = (fim * mask).abs().max().item()
            max_offdiag = max(max_offdiag, offdiag)
            diag_vals.append(fim.diag().mean().item())

        return {
            "max_offdiag_abs": max_offdiag,
            "diag_mean": sum(diag_vals) / len(diag_vals),
            "is_diagonal": max_offdiag < 1e-2,  # threshold for "approximately diagonal"
        }

    @staticmethod
    def compare_landscape(
        softmax_model,
        catnat_model,
        K: int,
        grid_size: int = 30,
    ) -> Dict:
        """Generate loss landscape data for softmax vs catnat (reproduces Figure 1).

        Computes cross-entropy loss over a 2D grid of score values for a
        3-class (K=3 or K=4) problem with uniform target distribution.

        Paper: Figure 1 — "Cross-entropy loss landscapes for softmax and catnat."

        Args:
            softmax_model: SoftmaxParam model.
            catnat_model:  CatNat model.
            K:             Number of categories.
            grid_size:     Grid resolution.

        Returns:
            Dict with 'softmax_loss', 'catnat_loss', 's1_grid', 's2_grid'.
        """
        import numpy as np

        s_range = torch.linspace(-6, 6, grid_size)
        s1_grid, s2_grid = torch.meshgrid(s_range, s_range, indexing="ij")

        # Uniform target
        target = torch.full((1, K), 1.0 / K)

        softmax_loss = torch.zeros(grid_size, grid_size)
        catnat_loss  = torch.zeros(grid_size, grid_size)

        for i in range(grid_size):
            for j in range(grid_size):
                s1, s2 = s1_grid[i, j].item(), s2_grid[i, j].item()

                # Softmax: K scores (fix remaining to 0)
                s_smx = torch.zeros(1, K)
                s_smx[0, 0] = s1
                s_smx[0, 1] = s2
                p_smx = softmax_model(s_smx)
                loss_smx = -(target * torch.log(p_smx.clamp(min=1e-8))).sum()
                softmax_loss[i, j] = loss_smx.item()

                # CatNat: K-1 scores
                s_cat = torch.zeros(1, K - 1)
                s_cat[0, 0] = s1
                if K - 1 > 1:
                    s_cat[0, 1] = s2
                p_cat = catnat_model(s_cat)
                loss_cat = -(target * torch.log(p_cat.clamp(min=1e-8))).sum()
                catnat_loss[i, j] = loss_cat.item()

        return {
            "softmax_loss": softmax_loss.numpy(),
            "catnat_loss":  catnat_loss.numpy(),
            "s1_grid":      s1_grid.numpy(),
            "s2_grid":      s2_grid.numpy(),
        }
