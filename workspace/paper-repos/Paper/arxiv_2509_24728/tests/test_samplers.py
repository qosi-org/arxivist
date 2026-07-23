"""
Unit tests for gradient estimator samplers.

Tests:
  1. GumbelSoftmaxSampler: output shape, hard one-hot, gradient flow, temperature effect
  2. REINFORCESampler: output shape, log_prob, valid samples
  3. LOOBaseline: shape, unbiasedness property

Paper: Appendix E.3 (REINFORCE), Appendix F.3 (Gumbel-Softmax).
"""

import math
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.catnat.samplers import GumbelSoftmaxSampler, REINFORCESampler
from src.catnat.training.baseline import LOOBaseline, MovingAverageBaseline


class TestGumbelSoftmaxSampler:
    """Tests for Gumbel-Softmax with Straight-Through estimator (Appendix F.3)."""

    @pytest.fixture
    def sampler(self):
        return GumbelSoftmaxSampler()

    @pytest.fixture
    def log_probs(self):
        K, B = 8, 6
        return torch.log_softmax(torch.randn(B, K), dim=-1)

    def test_output_shape(self, sampler, log_probs):
        out = sampler(log_probs, tau=1.0, hard=True)
        assert out.shape == log_probs.shape

    def test_hard_output_is_one_hot(self, sampler, log_probs):
        """With hard=True, output should be a one-hot vector."""
        out = sampler(log_probs, tau=0.1, hard=True)
        # Each row sums to 1 and has exactly one non-zero entry
        assert torch.allclose(out.sum(dim=-1), torch.ones(log_probs.shape[0]), atol=1e-5)
        # One-hot: each entry is 0 or 1
        assert torch.allclose(out, out.round(), atol=1e-5)

    def test_soft_output_sums_to_one(self, sampler, log_probs):
        out = sampler(log_probs, tau=1.0, hard=False)
        assert torch.allclose(out.sum(dim=-1), torch.ones(log_probs.shape[0]), atol=1e-5)
        assert (out >= 0).all() and (out <= 1.0 + 1e-6).all()

    def test_gradient_flows_through_hard_sample(self, sampler):
        """Straight-through: gradient should flow back through the hard sample."""
        K, B = 8, 4
        log_probs = torch.log_softmax(torch.randn(B, K, requires_grad=True), dim=-1)
        out = sampler(log_probs, tau=1.0, hard=True)
        loss = out.sum()
        loss.backward()
        assert log_probs.grad is not None
        assert not torch.isnan(log_probs.grad).any()

    def test_low_temperature_approaches_one_hot(self, sampler, log_probs):
        """At τ→0, soft samples should approach one-hot."""
        soft_high_tau = sampler(log_probs, tau=10.0, hard=False)
        soft_low_tau  = sampler(log_probs, tau=0.01, hard=False)
        # Low-τ samples should be more "peaky"
        entropy_high = -(soft_high_tau * (soft_high_tau + 1e-8).log()).sum(dim=-1).mean()
        entropy_low  = -(soft_low_tau  * (soft_low_tau  + 1e-8).log()).sum(dim=-1).mean()
        assert entropy_low < entropy_high, "Low temperature should give lower entropy"

    def test_temperature_annealing(self, sampler):
        """Temperature should decay and clamp to tau_min."""
        tau = sampler.anneal_temperature(step=0, tau_init=1.0, tau_min=0.5, anneal_rate=3e-5)
        assert abs(tau - 1.0) < 1e-5, f"At step=0, expected tau=1.0, got {tau}"

        tau_large = sampler.anneal_temperature(step=10**7, tau_init=1.0, tau_min=0.5,
                                               anneal_rate=3e-5)
        assert abs(tau_large - 0.5) < 1e-5, f"At large step, expected tau=0.5, got {tau_large}"

    def test_unknown_injection_point_raises(self, sampler, log_probs):
        with pytest.raises(NotImplementedError):
            sampler(log_probs, injection_point="node_logits")


class TestREINFORCESampler:
    """Tests for REINFORCE / Score Function gradient estimator (Appendix E.3)."""

    @pytest.fixture
    def sampler(self):
        return REINFORCESampler()

    @pytest.fixture
    def probs(self):
        K = 2   # Bernoulli (K=2 catnat)
        n_edges = 20
        return torch.softmax(torch.randn(n_edges, K), dim=-1)

    def test_sample_shape(self, sampler, probs):
        M = 8
        samples = sampler.sample(probs, n_samples=M)
        assert samples.shape == (M, *probs.shape)

    def test_samples_are_one_hot(self, sampler, probs):
        samples = sampler.sample(probs, n_samples=16)
        # Each sample in the last dim should be one-hot
        assert torch.allclose(samples.sum(dim=-1), torch.ones_like(samples.sum(dim=-1)), atol=1e-5)
        assert ((samples == 0) | (samples == 1)).all()

    def test_log_prob_shape(self, sampler, probs):
        M = 5
        samples = sampler.sample(probs, n_samples=M)
        lp = sampler.log_prob(samples, probs)
        assert lp.shape == (M, probs.shape[0])

    def test_log_prob_nonpositive(self, sampler, probs):
        """Log-probabilities of valid events should be ≤ 0."""
        samples = sampler.sample(probs, n_samples=10)
        lp = sampler.log_prob(samples, probs)
        assert (lp <= 1e-6).all(), "log_prob has positive values"

    def test_log_prob_correct_for_known_dist(self, sampler):
        """For K=2 with p=[0.8, 0.2], log P(C=0) = log(0.8)."""
        p = torch.tensor([[0.8, 0.2]])   # [1, 2]
        # Force sample C=0 (one-hot [1, 0])
        sample = torch.tensor([[[1.0, 0.0]]])   # [1, 1, 2]
        lp = sampler.log_prob(sample, p)
        expected = math.log(0.8)
        assert abs(lp.item() - expected) < 1e-5, (
            f"log P(C=0) expected {expected:.5f}, got {lp.item():.5f}"
        )


class TestLOOBaseline:
    """Tests for Leave-One-Out baseline (Appendix E.3)."""

    def test_shape(self):
        baseline = LOOBaseline()
        losses = torch.randn(8, 4)   # [M=8, B=4]
        b = baseline.compute(losses)
        assert b.shape == losses.shape

    def test_loo_excludes_own_loss(self):
        """Each baseline b_m should be the mean of all OTHER M-1 losses."""
        baseline = LOOBaseline()
        M, B = 4, 1
        losses = torch.tensor([[1.0], [2.0], [3.0], [4.0]])   # [M, B]
        b = baseline.compute(losses)
        # b[0] = (2+3+4)/3 = 3.0
        assert abs(b[0].item() - 3.0) < 1e-5, f"b[0] expected 3.0, got {b[0].item()}"
        # b[1] = (1+3+4)/3 = 8/3
        assert abs(b[1].item() - 8.0/3) < 1e-5

    def test_detached(self):
        """Baseline should not carry gradients (it's a control variate)."""
        baseline = LOOBaseline()
        losses = torch.randn(4, 3, requires_grad=True)
        b = baseline.compute(losses)
        assert not b.requires_grad, "Baseline should be detached from computation graph"

    def test_requires_m_geq_2(self):
        baseline = LOOBaseline()
        with pytest.raises(ValueError, match="M >= 2"):
            baseline.compute(torch.randn(1, 4))


class TestMovingAverageBaseline:
    def test_initializes_to_first_value(self):
        b = MovingAverageBaseline(alpha=0.99)
        val = b.update(5.0)
        assert abs(val - 5.0) < 1e-8

    def test_decays_toward_new_values(self):
        b = MovingAverageBaseline(alpha=0.5)
        b.update(10.0)
        val = b.update(0.0)   # EMA = 0.5*10 + 0.5*0 = 5.0
        assert abs(val - 5.0) < 1e-8

    def test_value_property(self):
        b = MovingAverageBaseline()
        b.update(3.0)
        assert abs(b.value - 3.0) < 1e-8
