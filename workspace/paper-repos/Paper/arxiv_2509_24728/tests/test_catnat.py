"""
Unit tests for the core CatNat parameterization.

Tests:
  1. Output probabilities sum to 1 (Eq. 8)
  2. Output shape matches [B, K]
  3. Gradient flows back through the forward pass
  4. log_prob and forward are consistent
  5. K validation raises on non-powers-of-2
  6. All four parameterizations (natural, sigmoid, softmax, sparsemax) produce
     valid probability vectors
  7. CatNat with K=2 is equivalent to a Bernoulli
  8. NaturalActivation saturates correctly outside [C-A/2, C+A/2]

Paper: Section 4.2, Eqs. 8, 12, Theorem 4.2.
"""

import math
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.catnat.catnat import CatNat, SoftmaxParam, SparseMaxParam, build_parameterization
from src.catnat.activations import NaturalActivation, SigmoidActivation
from src.catnat.utils.tree_utils import BinaryTreeIndex


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(params=[2, 4, 8, 16, 32])
def K(request):
    return request.param


@pytest.fixture(params=["natural", "sigmoid"])
def activation(request):
    return request.param


@pytest.fixture
def batch_scores(K):
    """Random scores of shape [B, K-1]."""
    B = 6
    return torch.randn(B, K - 1)


# ------------------------------------------------------------------
# 1. Probabilities sum to 1
# ------------------------------------------------------------------

class TestProbabilitiesSumToOne:
    """Core correctness test: Eq. 8 must produce a valid probability vector."""

    @pytest.mark.parametrize("K", [2, 4, 8, 16, 32])
    @pytest.mark.parametrize("activation", ["natural", "sigmoid"])
    def test_sum_to_one(self, K, activation):
        model = CatNat(K=K, activation=activation)
        s = torch.randn(8, K - 1)
        p = model(s)
        sums = p.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5), (
            f"Probabilities do not sum to 1 for K={K}, activation={activation}. "
            f"Got sums: {sums}"
        )

    @pytest.mark.parametrize("K", [2, 4, 8, 16, 32])
    @pytest.mark.parametrize("activation", ["natural", "sigmoid"])
    def test_all_nonnegative(self, K, activation):
        model = CatNat(K=K, activation=activation)
        s = torch.randn(8, K - 1)
        p = model(s)
        assert (p >= 0).all(), f"Negative probabilities for K={K}, activation={activation}"

    @pytest.mark.parametrize("K", [2, 4, 8, 16, 32])
    @pytest.mark.parametrize("activation", ["natural", "sigmoid"])
    def test_all_leq_one(self, K, activation):
        model = CatNat(K=K, activation=activation)
        s = torch.randn(8, K - 1)
        p = model(s)
        assert (p <= 1.0 + 1e-6).all(), f"Probabilities > 1 for K={K}"


# ------------------------------------------------------------------
# 2. Output shape
# ------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("K", [2, 4, 8, 16, 32])
    @pytest.mark.parametrize("B", [1, 4, 16])
    def test_shape_natural(self, K, B):
        model = CatNat(K=K, activation="natural")
        s = torch.randn(B, K - 1)
        p = model(s)
        assert p.shape == (B, K), f"Expected [{B},{K}], got {p.shape}"

    def test_log_prob_shape(self):
        K, B = 8, 5
        model = CatNat(K=K)
        s = torch.randn(B, K - 1)
        lp = model.log_prob(s)
        assert lp.shape == (B, K)

    def test_node_reach_probs_shape(self):
        K, B = 8, 5
        model = CatNat(K=K)
        s = torch.randn(B, K - 1)
        reach = model.node_reach_probs(s)
        assert reach.shape == (B, K - 1)

    def test_fim_diagonal_shape(self):
        K, B = 8, 5
        model = CatNat(K=K)
        s = torch.randn(B, K - 1)
        diag = model.fim_diagonal(s)
        assert diag.shape == (B, K - 1)

    def test_multidim_batch(self):
        """CatNat should handle [..., K-1] input shapes."""
        K = 8
        model = CatNat(K=K)
        s = torch.randn(3, 5, K - 1)   # 2D batch
        p = model(s)
        assert p.shape == (3, 5, K)
        assert torch.allclose(p.sum(dim=-1), torch.ones(3, 5), atol=1e-5)


# ------------------------------------------------------------------
# 3. Gradient flow
# ------------------------------------------------------------------

class TestGradientFlow:
    @pytest.mark.parametrize("K", [4, 8, 16])
    @pytest.mark.parametrize("activation", ["natural", "sigmoid"])
    def test_gradient_flows(self, K, activation):
        model = CatNat(K=K, activation=activation)
        s = torch.randn(4, K - 1, requires_grad=True)
        p = model(s)
        loss = p.sum()
        loss.backward()
        assert s.grad is not None, "No gradient reached scores"
        assert not torch.isnan(s.grad).any(), "NaN in gradient"
        assert not torch.isinf(s.grad).any(), "Inf in gradient"

    @pytest.mark.parametrize("K", [4, 8, 16])
    def test_gradient_nonzero(self, K):
        """Gradient should be non-zero (activation is not saturated at random init)."""
        torch.manual_seed(42)
        model = CatNat(K=K, activation="natural")
        s = torch.randn(4, K - 1, requires_grad=True)
        loss = model(s).sum()
        loss.backward()
        assert s.grad.abs().sum() > 0, "All gradients are zero"


# ------------------------------------------------------------------
# 4. log_prob consistency
# ------------------------------------------------------------------

class TestLogProbConsistency:
    @pytest.mark.parametrize("K", [2, 4, 8, 16])
    def test_log_prob_exp_matches_forward(self, K):
        model = CatNat(K=K)
        s = torch.randn(6, K - 1)
        p_forward = model(s)
        p_logprob = model.log_prob(s).exp()
        assert torch.allclose(p_forward, p_logprob, atol=1e-5), (
            f"log_prob().exp() != forward() for K={K}"
        )

    @pytest.mark.parametrize("K", [2, 4, 8])
    def test_log_probs_are_negative(self, K):
        """log(p_k) ≤ 0 for all valid probabilities."""
        model = CatNat(K=K)
        s = torch.randn(6, K - 1)
        lp = model.log_prob(s)
        assert (lp <= 1e-6).all(), "log_prob has values > 0"


# ------------------------------------------------------------------
# 5. K validation
# ------------------------------------------------------------------

class TestKValidation:
    def test_non_power_of_2_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            CatNat(K=3)

    def test_non_power_of_2_raises_7(self):
        with pytest.raises(ValueError, match="power of 2"):
            CatNat(K=7)

    def test_non_power_of_2_raises_10(self):
        with pytest.raises(ValueError, match="power of 2"):
            CatNat(K=10)

    def test_k_1_raises(self):
        with pytest.raises(ValueError):
            CatNat(K=1)

    def test_power_of_2_accepted(self):
        for K in [2, 4, 8, 16, 32, 64]:
            model = CatNat(K=K)
            assert model.K == K


# ------------------------------------------------------------------
# 6. All four parameterizations produce valid vectors
# ------------------------------------------------------------------

class TestAllParameterizations:
    @pytest.mark.parametrize("name,K", [
        ("natural",  8),
        ("sigmoid",  8),
        ("softmax",  8),
        ("sparsemax", 8),
    ])
    def test_valid_probs(self, name, K):
        model = build_parameterization(name, K=K)
        # softmax/sparsemax take K scores; catnat takes K-1
        S = K - 1 if name in ("natural", "sigmoid") else K
        s = torch.randn(5, S)
        p = model(s)
        assert p.shape[-1] == K
        assert (p >= 0).all()
        assert torch.allclose(p.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_unknown_parameterization_raises(self):
        with pytest.raises(ValueError):
            build_parameterization("unknown", K=8)


# ------------------------------------------------------------------
# 7. K=2 Bernoulli equivalence
# ------------------------------------------------------------------

class TestBernoulliEquivalence:
    """For K=2, catnat should be equivalent to a simple Bernoulli."""

    def test_k2_probabilities_sum_to_one(self):
        model = CatNat(K=2, activation="sigmoid")
        s = torch.randn(10, 1)   # K-1 = 1 score
        p = model(s)
        assert p.shape == (10, 2)
        assert torch.allclose(p.sum(dim=-1), torch.ones(10), atol=1e-5)

    def test_k2_natural_sigmoid_consistent(self):
        """For K=2, p[0] = 1 - p[1] by construction."""
        model = CatNat(K=2, activation="sigmoid")
        s = torch.randn(8, 1)
        p = model(s)
        assert torch.allclose(p[:, 0] + p[:, 1], torch.ones(8), atol=1e-5)

    def test_k2_natural_activation(self):
        model = CatNat(K=2, activation="natural")
        s = torch.randn(8, 1)
        p = model(s)
        assert torch.allclose(p.sum(dim=-1), torch.ones(8), atol=1e-5)


# ------------------------------------------------------------------
# 8. NaturalActivation saturation
# ------------------------------------------------------------------

class TestNaturalActivation:
    def test_saturates_to_zero_below(self):
        nu = NaturalActivation(C=0.0, A=2 * math.pi)
        x = torch.tensor([-10.0, -5.0, -math.pi - 0.01])
        out = nu(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_saturates_to_one_above(self):
        nu = NaturalActivation(C=0.0, A=2 * math.pi)
        x = torch.tensor([10.0, 5.0, math.pi + 0.01])
        out = nu(x)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-6)

    def test_half_at_centre(self):
        """ν(C) = 0.5 (sin(0) = 0 → (1+0)/2 = 0.5)."""
        nu = NaturalActivation(C=0.0, A=2 * math.pi)
        out = nu(torch.tensor([0.0]))
        assert abs(out.item() - 0.5) < 1e-6

    def test_slope_matches_sigmoid_at_zero(self):
        """∂ν/∂s|_{s=0} = ∂σ/∂s|_{s=0} = 0.25 when A=2π."""
        A = 2 * math.pi
        nu = NaturalActivation(C=0.0, A=A)
        sig = SigmoidActivation()
        s = torch.tensor([0.0], requires_grad=True)

        out_nu = nu(s)
        out_nu.backward()
        grad_nu = s.grad.item()
        s.grad.zero_()

        out_sig = sig(s)
        out_sig.backward()
        grad_sig = s.grad.item()

        assert abs(grad_nu - grad_sig) < 1e-4, (
            f"Slopes differ: ∂ν/∂s={grad_nu:.6f}, ∂σ/∂s={grad_sig:.6f}"
        )

    def test_output_in_unit_interval(self):
        nu = NaturalActivation()
        x = torch.linspace(-10, 10, 200)
        out = nu(x)
        assert (out >= 0).all() and (out <= 1).all()


# ------------------------------------------------------------------
# 9. BinaryTreeIndex correctness
# ------------------------------------------------------------------

class TestBinaryTreeIndex:
    @pytest.mark.parametrize("K", [2, 4, 8, 16])
    def test_leaf_paths_shape(self, K):
        idx = BinaryTreeIndex(K)
        assert idx.leaf_paths.shape == (K, int(math.log2(K)))

    @pytest.mark.parametrize("K", [2, 4, 8, 16])
    def test_ancestor_mask_shape(self, K):
        idx = BinaryTreeIndex(K)
        assert idx.ancestor_mask.shape == (K, K - 1)

    @pytest.mark.parametrize("K", [4, 8])
    def test_each_leaf_has_H_ancestors(self, K):
        """Each leaf should have exactly H = log2(K) ancestors (root to parent)."""
        H = int(math.log2(K))
        idx = BinaryTreeIndex(K)
        n_ancestors = idx.ancestor_mask.sum(dim=-1)  # [K]
        assert (n_ancestors == H).all(), (
            f"Not all leaves have {H} ancestors: {n_ancestors}"
        )

    def test_non_power_of_2_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            BinaryTreeIndex(K=3)

    @pytest.mark.parametrize("K", [2, 4, 8])
    def test_branch_taken_binary(self, K):
        """branch_taken entries must be 0 or 1."""
        idx = BinaryTreeIndex(K)
        vals = idx.branch_taken.unique()
        assert set(vals.tolist()).issubset({0, 1})


# ------------------------------------------------------------------
# 10. Numerical stability at extreme scores
# ------------------------------------------------------------------

class TestNumericalStability:
    @pytest.mark.parametrize("K", [4, 8, 16])
    def test_large_positive_scores(self, K):
        model = CatNat(K=K)
        s = torch.full((4, K - 1), 100.0)
        p = model(s)
        assert not torch.isnan(p).any(), "NaN with large positive scores"
        assert not torch.isinf(p).any(), "Inf with large positive scores"
        assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-4)

    @pytest.mark.parametrize("K", [4, 8, 16])
    def test_large_negative_scores(self, K):
        model = CatNat(K=K)
        s = torch.full((4, K - 1), -100.0)
        p = model(s)
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()
        assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-4)

    @pytest.mark.parametrize("K", [4, 8])
    def test_zero_scores_uniform(self, K):
        """s=0 → all node activations = 0.5 → uniform leaf probabilities."""
        model = CatNat(K=K, activation="sigmoid")
        s = torch.zeros(1, K - 1)
        p = model(s)
        expected = torch.full((1, K), 1.0 / K)
        assert torch.allclose(p, expected, atol=1e-5), (
            f"Zero scores → expected uniform, got {p}"
        )
