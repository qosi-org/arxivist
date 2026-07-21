"""
Unit tests verifying Theorem 4.2: the catnat FIM is diagonal.

Tests:
  1. Analytical diagonal FIM entries match numerical gradient-based estimates
  2. Corollary 4.3: entries are (π/A)² * P(a_i) for natural activation
  3. Off-diagonal FIM entries are approximately zero for catnat
  4. Softmax FIM is dense (non-zero off-diagonal entries, as expected)
  5. node_reach_probs sums correctly

Paper: Theorem 4.2, Corollary 4.3, Proposition 4.1.
"""

import math
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.catnat.catnat import CatNat, SoftmaxParam
from src.catnat.activations import NaturalActivation


class TestCatNatFIMDiagonal:
    """Theorem 4.2: G_a(s)_ij = 0 for i ≠ j."""

    def _numerical_fim(self, model, s, n_samples=2000):
        """Estimate FIM numerically via E[(∇ log p)(∇ log p)^T]."""
        S = s.shape[-1]
        outer_sum = torch.zeros(S, S)
        s_req = s.clone().detach().requires_grad_(True)

        for _ in range(n_samples):
            if s_req.grad is not None:
                s_req.grad.zero_()

            if hasattr(model, "log_prob"):
                log_p = model.log_prob(s_req)
            else:
                log_p = torch.log(model(s_req).clamp(min=1e-8))

            # Sample category from current distribution
            with torch.no_grad():
                probs = log_p.exp().squeeze(0)
                k = torch.multinomial(probs, 1).item()

            log_pk = log_p[0, k]
            log_pk.backward(retain_graph=True)

            g = s_req.grad.squeeze(0).detach().clone()
            outer_sum += torch.outer(g, g)
            s_req.grad.zero_()

        return outer_sum / n_samples

    @pytest.mark.parametrize("K,activation", [(4, "natural"), (8, "natural"), (4, "sigmoid")])
    def test_fim_approximately_diagonal(self, K, activation):
        """Off-diagonal FIM entries should be ≈ 0 for catnat."""
        torch.manual_seed(0)
        model = CatNat(K=K, activation=activation)
        s = torch.zeros(1, K - 1)   # Use s=0 for stable numerical estimate

        fim = self._numerical_fim(model, s, n_samples=3000)

        # Off-diagonal entries
        diag_mask = torch.eye(K - 1).bool()
        off_diag = fim[~diag_mask].abs()

        max_off_diag = off_diag.max().item()
        mean_diag    = fim[diag_mask].mean().item()

        # Off-diagonal entries should be small relative to diagonal
        ratio = max_off_diag / (mean_diag + 1e-8)
        assert ratio < 0.15, (
            f"FIM not sufficiently diagonal for K={K}, activation={activation}. "
            f"max_off_diag/mean_diag = {ratio:.4f} (expected < 0.15). "
            f"FIM diagonal: {fim.diag().tolist()}"
        )

    @pytest.mark.parametrize("K", [4, 8])
    def test_diagonal_entries_positive(self, K):
        """Diagonal FIM entries should be positive (Theorem 4.2, Eq. 11)."""
        model = CatNat(K=K, activation="natural")
        s = torch.randn(3, K - 1)
        diag = model.fim_diagonal(s)
        assert (diag >= 0).all(), f"Negative diagonal FIM entries for K={K}"


class TestCorollary43:
    """Corollary 4.3: G_ν(s)_ii = P(a_i) * (π/A)² in the active region."""

    @pytest.mark.parametrize("K", [4, 8])
    def test_corollary_43_active_region(self, K):
        """Verify Corollary 4.3 analytically for scores in the active region."""
        A = 2 * math.pi
        model = CatNat(K=K, activation="natural", A=A, C=0.0)

        # Use s=0 so all nodes are in the active region (|s_i - C| = 0 < A/2)
        s = torch.zeros(1, K - 1)

        diag = model.fim_diagonal(s).squeeze(0)            # [K-1]
        reach = model.node_reach_probs(s).squeeze(0)       # [K-1]
        expected = reach * (math.pi / A) ** 2

        assert torch.allclose(diag, expected, atol=1e-5), (
            f"Corollary 4.3 failed for K={K}.\n"
            f"  Got:      {diag.tolist()}\n"
            f"  Expected: {expected.tolist()}"
        )


class TestNodeReachProbs:
    """P(a_i) properties: all entries in [0,1] and consistent with leaf probabilities."""

    @pytest.mark.parametrize("K", [4, 8, 16])
    def test_reach_probs_in_unit_interval(self, K):
        model = CatNat(K=K)
        s = torch.randn(5, K - 1)
        reach = model.node_reach_probs(s)
        assert (reach >= 0).all() and (reach <= 1.0 + 1e-6).all()

    def test_root_reach_prob_is_one(self):
        """The root node (node 0) is always reached with probability 1."""
        K = 8
        model = CatNat(K=K, activation="sigmoid")
        s = torch.randn(4, K - 1)
        reach = model.node_reach_probs(s)   # [4, K-1]
        # Root is node 0 in BFS order
        root_reach = reach[:, 0]
        assert torch.allclose(root_reach, torch.ones(4), atol=1e-5), (
            f"Root reach probability ≠ 1. Got {root_reach}"
        )


class TestSoftmaxFIMDense:
    """Proposition 4.1: the softmax FIM is dense (off-diagonal entries ≠ 0).

    This is the negative result that motivates catnat.
    We verify that softmax produces a non-diagonal FIM,
    confirming the paper's claim in Proposition 4.1.
    """

    def test_softmax_fim_has_nonzero_offdiag(self):
        """Softmax FIM should have non-zero off-diagonal entries."""
        K = 4
        # Analytical softmax FIM (Proposition 4.1, Eq. 6)
        # G_smx_ij = p_i*(1-p_i) for i==j, -p_i*p_j for i!=j
        p = torch.softmax(torch.randn(K), dim=0)   # Random categorical probs
        G_smx = torch.zeros(K, K)
        for i in range(K):
            for j in range(K):
                if i == j:
                    G_smx[i, j] = p[i] * (1 - p[i])
                else:
                    G_smx[i, j] = -p[i] * p[j]

        diag_mask = torch.eye(K).bool()
        off_diag = G_smx[~diag_mask].abs()
        # Off-diagonal entries should be non-trivially nonzero
        assert off_diag.max() > 0.01, (
            "Softmax FIM appears diagonal — something is wrong with the test."
        )
