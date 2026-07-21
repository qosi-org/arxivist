"""
Unit tests for loss functions.

Tests:
  1. EnergyScore: shape, non-negativity, zero when preds == target
  2. KLDivCategorical: non-negative, zero for uniform posterior, correct analytical form
  3. VAELoss: total = recon + kl, all components non-negative

Paper: Eq. 10 (Energy Score), Appendix F.4 (KL, ELBO).
"""

import math
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.catnat.training.losses import EnergyScore, KLDivCategorical, VAELoss


class TestEnergyScore:
    """Tests for the Energy Score loss (Eq. 10, Appendix E.4)."""

    def test_output_is_scalar(self):
        es = EnergyScore()
        preds  = torch.randn(8, 4, 3)   # [M, B, D]
        target = torch.randn(4, 3)       # [B, D]
        loss = es(preds, target)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_nonnegative(self):
        """ES is a proper scoring rule — always ≥ 0."""
        es = EnergyScore()
        preds  = torch.randn(16, 8, 5)
        target = torch.randn(8, 5)
        assert es(preds, target).item() >= 0

    def test_zero_when_perfect_and_zero_variance(self):
        """If all M predictions are exactly equal to target, term1 = term2 = 0 → ES = 0."""
        es = EnergyScore()
        target = torch.ones(4, 3)
        # M identical predictions equal to target
        preds = target.unsqueeze(0).expand(8, -1, -1)  # [8, 4, 3]
        loss = es(preds, target)
        assert abs(loss.item()) < 1e-5, f"Expected ES ≈ 0, got {loss.item()}"

    def test_gradient_flows(self):
        es = EnergyScore()
        preds  = torch.randn(4, 3, 2, requires_grad=True)
        target = torch.randn(3, 2)
        loss = es(preds, target)
        loss.backward()
        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_shape_check_raises(self):
        es = EnergyScore()
        with pytest.raises(AssertionError):
            es(torch.randn(4, 3), torch.randn(3, 2))   # preds must be 3D

    def test_larger_M_reduces_variance_proxy(self):
        """With more samples, the ES should converge (not a strict test, just sanity)."""
        es = EnergyScore()
        torch.manual_seed(0)
        target = torch.zeros(2, 4)
        # M=2 vs M=32: the ES value changes but should remain non-negative
        for M in [2, 8, 32]:
            preds = torch.randn(M, 2, 4)
            assert es(preds, target).item() >= 0


class TestKLDivCategorical:
    """Tests for the analytic KL divergence KL(Cat(q) || Uniform)."""

    def test_zero_for_uniform_posterior(self):
        """KL(Uniform || Uniform) = 0."""
        kl = KLDivCategorical()
        K = 8
        N, B = 3, 4
        # Uniform posterior
        q = torch.full((B, N, K), 1.0 / K)
        loss = kl(q)
        assert abs(loss.item()) < 1e-5, f"KL(Uniform||Uniform) ≠ 0, got {loss.item()}"

    def test_nonnegative(self):
        """KL divergence is always ≥ 0."""
        kl = KLDivCategorical()
        q = torch.softmax(torch.randn(4, 5, 8), dim=-1)
        assert kl(q).item() >= 0

    def test_peaked_posterior_high_kl(self):
        """A one-hot posterior should have high KL vs uniform."""
        kl = KLDivCategorical()
        K = 8
        q = torch.zeros(2, 3, K)
        q[:, :, 0] = 1.0 - 1e-6
        q[:, :, 1:] = 1e-6 / (K - 1)
        loss = kl(q)
        assert loss.item() > 0.5, f"One-hot posterior has low KL: {loss.item()}"

    def test_shape_check_raises(self):
        kl = KLDivCategorical()
        with pytest.raises(AssertionError):
            kl(torch.randn(4, 8))   # Must be 3D [B, N, K]

    def test_analytical_value_k4(self):
        """Check KL against manual calculation for K=4."""
        kl_fn = KLDivCategorical()
        K = 4
        # q = [0.7, 0.1, 0.1, 0.1]
        q_vals = torch.tensor([0.7, 0.1, 0.1, 0.1])
        q = q_vals.view(1, 1, K)
        # Analytical: Σ q_k * log(q_k * K)
        expected = (q_vals * (q_vals.log() + math.log(K))).sum().item()
        got = kl_fn(q).item()
        assert abs(got - expected) < 1e-5, f"Expected {expected:.6f}, got {got:.6f}"


class TestVAELoss:
    """Tests for the combined VAE ELBO loss."""

    def _make_inputs(self, B=4, N=5, K=8, H=28, W=28):
        x_recon = torch.sigmoid(torch.randn(B, 1, H, W))
        x_target = torch.randint(0, 2, (B, 1, H, W)).float()
        q_probs = torch.softmax(torch.randn(B, N, K), dim=-1)
        return x_recon, x_target, q_probs

    def test_total_equals_recon_plus_kl(self):
        vae_loss = VAELoss(beta=1.0)
        x_recon, x_target, q_probs = self._make_inputs()
        result = vae_loss(x_recon, x_target, q_probs)
        expected_total = result["recon"] + result["kl"]
        assert torch.allclose(result["total"], expected_total, atol=1e-5)

    def test_all_components_nonnegative(self):
        vae_loss = VAELoss(beta=1.0)
        x_recon, x_target, q_probs = self._make_inputs()
        result = vae_loss(x_recon, x_target, q_probs)
        assert result["recon"].item() >= 0
        assert result["kl"].item() >= 0
        assert result["total"].item() >= 0

    def test_beta_scales_kl(self):
        """beta=2 should give total = recon + 2*kl."""
        x_recon, x_target, q_probs = self._make_inputs()
        r1 = VAELoss(beta=1.0)(x_recon, x_target, q_probs)
        r2 = VAELoss(beta=2.0)(x_recon, x_target, q_probs)
        expected = r1["recon"] + 2.0 * r1["kl"]
        assert torch.allclose(r2["total"], expected, atol=1e-5)

    def test_gradient_flows(self):
        vae_loss = VAELoss()
        x_recon = torch.sigmoid(torch.randn(2, 1, 28, 28, requires_grad=True))
        x_target = torch.randint(0, 2, (2, 1, 28, 28)).float()
        q_probs = torch.softmax(torch.randn(2, 3, 8), dim=-1)
        result = vae_loss(x_recon, x_target, q_probs)
        result["total"].backward()
        assert x_recon.grad is not None
        assert not torch.isnan(x_recon.grad).any()

    def test_shape_mismatch_raises(self):
        vae_loss = VAELoss()
        with pytest.raises(AssertionError):
            vae_loss(
                torch.randn(4, 1, 28, 28),
                torch.randn(4, 1, 32, 32),  # wrong size
                torch.softmax(torch.randn(4, 3, 8), dim=-1),
            )
