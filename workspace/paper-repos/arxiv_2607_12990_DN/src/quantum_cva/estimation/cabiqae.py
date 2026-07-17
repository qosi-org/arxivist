"""
Contrast-Aware Bayesian Iterative Quantum Amplitude Estimation (CABIQAE) --
the central algorithmic contribution of arXiv:2607.12990.

Implements Section 2.2.2 and Appendix A.3 (Algorithm 1) in full:
  - noise-aware observation model (Eq. 40): p_obs(theta,k) = b + c(k)(sin^2(K theta)-b)
  - Beta-conjugate Bayesian update (Eq. 59)
  - posterior pullback to latent-angle credible interval (Eq. 60-62)
  - posterior-averaged Fisher-information Grover-depth scheduler (Eq. 64-66, 70)
  - prior transport across Grover-depth changes (Eq. 67-69)

Also implements the three comparison baselines used in the paper:
  - BIQAE: ideal-likelihood Beta-BIQAE (Li et al. 2026) -- noise-naive,
    i.e. CABIQAE with c(k) == 1 for all k (no contrast decay).
  - BAE: a simplified Bayesian Amplitude Estimation baseline (Ramôa & Santos
    2025) using sequential-Monte-Carlo-style particle posterior propagation.
  - DirectCircuitSampling (DCS): unamplified k=0 Bernoulli sampling baseline.

SIR reference: architecture.modules "CABIQAE classical estimator",
mathematical_spec entries for Eq. 38-40, 59-66, 72.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import stats


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Version-safe trapezoidal integration (NumPy >=2.0 removed np.trapz in
    favour of np.trapezoid; NumPy <2.0 only has np.trapz)."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


CircuitExecutor = Callable[[int, int], Tuple[int, int]]
"""A callable (k, n_shots) -> (n_success, n_shots) that runs Q^k A for
n_shots shots and returns the number of successes (ancillas measured 111)
together with the shots actually used."""


@dataclass
class EstimationResult:
    """Container for the outcome of an amplitude-estimation run."""

    a_hat: float
    a_lower: float
    a_upper: float
    total_queries: int
    max_k: int
    n_stages: int

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"EstimationResult(a_hat={self.a_hat:.6g}, "
            f"CI=[{self.a_lower:.6g}, {self.a_upper:.6g}], "
            f"N_q={self.total_queries}, K_max={2*self.max_k+1})"
        )


class CABIQAE:
    """Contrast-aware Bayesian iterative quantum amplitude estimator.

    Args:
        c0: initial contrast prefactor (paper: ~1.0 ideal / ~1.16 or 0.67
            hardware-calibrated, Tables 17-18).
        tau_c: contrast-decay scale in units of amplification depth K=2k+1
            (paper: 33.96 for the validation circuit, 2.20 for the full CVA
            oracle -- Tables 17-18).
        b: asymptotic baseline probability (paper: 0.5 for the balanced
            validation circuit, 0.15 for the full CVA oracle).
        rho_min: minimum stage-growth ratio K_next/K_current (paper uses 2).
        n_prior_samples: number of posterior samples used for prior transport
            (paper uses 2000).
        delta: numerical floor to avoid division by zero in Fisher-information
            computations (Appendix A.3 "Numerical implementation").
    """

    def __init__(
        self,
        c0: float = 1.0,
        tau_c: float = 1e12,
        b: float = 0.5,
        rho_min: float = 2.0,
        n_prior_samples: int = 2000,
        delta: float = 1e-6,
        grid_size: int = 4001,
    ) -> None:
        self.c0 = c0
        self.tau_c = tau_c
        self.b = b
        self.rho_min = rho_min
        self.n_prior_samples = n_prior_samples
        self.delta = delta
        self.grid_size = grid_size

    def __repr__(self) -> str:  # noqa: D105
        return f"CABIQAE(c0={self.c0}, tau_c={self.tau_c}, b={self.b})"

    # ------------------------------------------------------------------
    # Noise-aware observation model (Eq. 40)
    # ------------------------------------------------------------------
    def contrast_model(self, k: int) -> float:
        """c(k) = c0 * exp(-(2k+1)/tau_c)  (Eq. 40)."""
        K = 2 * k + 1
        return self.c0 * np.exp(-K / self.tau_c)

    def p_obs(self, theta: np.ndarray, k: int) -> np.ndarray:
        """p_obs(theta, k) = b + c(k) * (sin^2(K*theta) - b)  (Eq. 40)."""
        K = 2 * k + 1
        q = np.sin(K * theta) ** 2
        return self.b + self.contrast_model(k) * (q - self.b)

    # ------------------------------------------------------------------
    # Bayesian update (Eq. 59)
    # ------------------------------------------------------------------
    def bayesian_update(
        self, alpha0: float, beta0: float, n_shots: int, n_success: int
    ) -> Tuple[float, float]:
        """Conjugate Beta posterior update over the observed probability p_t.

        alpha_post = alpha0 + Y_t,  beta_post = beta0 + N_t - Y_t   (Eq. 59)
        """
        return alpha0 + n_success, beta0 + (n_shots - n_success)

    # ------------------------------------------------------------------
    # Posterior pullback + credible interval (Eq. 60-62, 70)
    # ------------------------------------------------------------------
    def compute_cri(
        self,
        interval: Tuple[float, float],
        k: int,
        alpha_post: float,
        beta_post: float,
        alpha: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], np.ndarray, np.ndarray]:
        """Pull the Beta(alpha_post, beta_post) posterior over p_t back to
        latent-angle space and compute the equal-tailed (1-alpha) credible
        interval (Eq. 60-62).

        Args:
            interval: current identifiable branch I_t = [theta_l, theta_u].
            k: current Grover power.
            alpha_post, beta_post: posterior Beta parameters over p_t.
            alpha: failure probability (paper uses 0.10).

        Returns:
            (new_theta_interval, new_amplitude_interval, theta_grid, posterior_density)
        """
        theta_l, theta_u = interval
        theta_grid = np.linspace(theta_l, theta_u, self.grid_size)
        K = 2 * k + 1

        p_theta = self.p_obs(theta_grid, k)
        p_theta_clipped = np.clip(p_theta, self.delta, 1 - self.delta)

        # |p'_k(theta)| = |c_k * K * sin(2K theta)|   (Eq. 60)
        dp_dtheta = np.abs(self.contrast_model(k) * K * np.sin(2 * K * theta_grid))
        dp_dtheta = np.maximum(dp_dtheta, self.delta)

        beta_pdf = stats.beta.pdf(p_theta_clipped, alpha_post, beta_post)
        unnorm_density = beta_pdf * dp_dtheta

        Z = _trapz(unnorm_density, theta_grid)
        if not np.isfinite(Z) or Z <= 0:
            # Fallback: uniform density on the current interval (Appendix A.3)
            density = np.full_like(theta_grid, 1.0 / (theta_u - theta_l))
        else:
            density = unnorm_density / Z

        # Trapezoidal CDF
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(theta_grid))]
        )
        cdf = cdf / cdf[-1]

        theta_lo = float(np.interp(alpha / 2, cdf, theta_grid))
        theta_hi = float(np.interp(1 - alpha / 2, cdf, theta_grid))

        a_lo = float(np.sin(theta_lo) ** 2)
        a_hi = float(np.sin(theta_hi) ** 2)
        return (theta_lo, theta_hi), (min(a_lo, a_hi), max(a_lo, a_hi)), theta_grid, density

    # ------------------------------------------------------------------
    # Grover-depth scheduler (Eq. 63-66, 70)
    # ------------------------------------------------------------------
    def find_next_k(
        self,
        interval: Tuple[float, float],
        current_k: int,
        theta_grid: np.ndarray,
        posterior_density: np.ndarray,
        k_max: int = 200,
    ) -> int:
        """Select the next Grover power maximising posterior-averaged Fisher
        information per query, subject to single-branch identifiability and
        the minimum stage-growth ratio (Eq. 63, 66, 70).

        Args:
            interval: new identifiable interval I_t^new = (theta_lo, theta_hi).
            current_k: current Grover power k_t.
            theta_grid, posterior_density: discretised posterior pi_t(theta).
            k_max: search ceiling for the candidate Grover power.

        Returns:
            k_next (== current_k if no admissible candidate improves the score).
        """
        theta_lo, theta_hi = interval
        best_k = current_k
        best_score = -np.inf

        for k in range(current_k, k_max + 1):
            K = 2 * k + 1
            if K < self.rho_min * (2 * current_k + 1) and k != current_k:
                continue
            # Single-branch identifiability: K * interval must not cross a
            # pi/2 branch boundary (Eq. 63).
            lo_branch = np.floor(K * theta_lo / (np.pi / 2))
            hi_branch = np.floor(K * theta_hi / (np.pi / 2))
            if lo_branch != hi_branch:
                continue

            p_k = self.p_obs(theta_grid, k)
            p_k_clipped = np.clip(p_k, self.delta, 1 - self.delta)
            dp_dtheta = self.contrast_model(k) * K * np.sin(2 * K * theta_grid)

            fisher = (dp_dtheta**2) / (p_k_clipped * (1 - p_k_clipped))
            gamma_k = _trapz(posterior_density * np.abs(np.sin(2 * K * theta_grid)), theta_grid)
            gamma_k = max(gamma_k, self.delta)

            score = (gamma_k / K) * _trapz(posterior_density * fisher, theta_grid)  # Eq. 70
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    # ------------------------------------------------------------------
    # Prior transport (Eq. 67-69)
    # ------------------------------------------------------------------
    def prepare_prior(
        self,
        theta_grid: np.ndarray,
        posterior_density: np.ndarray,
        next_k: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, float]:
        """Transport the latent posterior through the new likelihood at
        `next_k` and moment-match it to a Beta prior (Eq. 67-69).
        """
        rng = rng or np.random.default_rng()
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (posterior_density[1:] + posterior_density[:-1]) * np.diff(theta_grid))]
        )
        cdf = cdf / cdf[-1]
        u = rng.uniform(0, 1, self.n_prior_samples)
        theta_samples = np.interp(u, cdf, theta_grid)
        p_samples = self.p_obs(theta_samples, next_k)

        mu = float(np.mean(p_samples))
        var = float(np.var(p_samples))
        var = max(var, 1e-12)

        phi_hat = mu * (1 - mu) / var - 1
        cap = self.contrast_model(next_k) ** 2 * self.n_prior_samples  # Eq. 68
        phi = min(phi_hat, cap) if phi_hat > 0 else 1.0

        alpha0 = max(mu * phi, 0.5)
        beta0 = max((1 - mu) * phi, 0.5)
        return alpha0, beta0

    # ------------------------------------------------------------------
    # Full adaptive loop (Algorithm 1)
    # ------------------------------------------------------------------
    def estimate(
        self,
        circuit_executor: CircuitExecutor,
        epsilon: float,
        alpha: float = 0.10,
        n_batch: int = 256,
        max_stages: int = 200,
        k_max: int = 200,
    ) -> EstimationResult:
        """Run the full CABIQAE adaptive loop (Algorithm 1, Appendix A.3).

        Args:
            circuit_executor: callable (k, n_shots) -> (n_success, n_shots)
                that runs Q^k A on the target backend (statevector sampler,
                Aer noisy simulator, or hardware-replay model).
            epsilon: target credible-interval half-width.
            alpha: failure probability (paper uses 0.10).
            n_batch: shots per batch (paper uses 256 for validation, 128 for
                the full CVA experiment).
            max_stages: safety cap on the number of adaptive stages.
            k_max: safety cap on the Grover power search.

        Returns:
            EstimationResult with point estimate, credible interval, and
            query-cost accounting.
        """
        k_t = 0
        interval = (0.0, np.pi / 2)
        alpha0, beta0 = 0.5, 0.5
        N_t, Y_t = 0, 0
        total_queries = 0
        max_k_reached = 0

        for stage in range(max_stages):
            n_success, n_shots = circuit_executor(k_t, n_batch)
            N_t += n_shots
            Y_t += n_success
            total_queries += n_shots * (2 * k_t + 1)
            max_k_reached = max(max_k_reached, k_t)

            alpha_post, beta_post = self.bayesian_update(alpha0, beta0, N_t, Y_t)
            new_theta_interval, (a_lo, a_hi), theta_grid, density = self.compute_cri(
                interval, k_t, alpha_post, beta_post, alpha
            )

            if (a_hi - a_lo) <= 2 * epsilon:
                a_hat = 0.5 * (a_lo + a_hi)
                return EstimationResult(
                    a_hat=a_hat,
                    a_lower=a_lo,
                    a_upper=a_hi,
                    total_queries=total_queries,
                    max_k=max_k_reached,
                    n_stages=stage + 1,
                )

            k_next = self.find_next_k(new_theta_interval, k_t, theta_grid, density, k_max=k_max)

            if k_next > k_t:
                alpha0, beta0 = self.prepare_prior(theta_grid, density, k_next)
                k_t = k_next
                interval = new_theta_interval
                N_t, Y_t = 0, 0
            else:
                interval = new_theta_interval

        a_hat = 0.5 * (a_lo + a_hi)
        return EstimationResult(
            a_hat=a_hat,
            a_lower=a_lo,
            a_upper=a_hi,
            total_queries=total_queries,
            max_k=max_k_reached,
            n_stages=max_stages,
        )


class BIQAE(CABIQAE):
    """Noise-naive Beta-BIQAE baseline (Li et al. 2026).

    Implemented as CABIQAE with an ideal (non-decaying) contrast model:
    c(k) == 1 for all k, i.e. tau_c -> infinity. This exactly recovers the
    ideal amplified-observation likelihood q(theta,k) = sin^2(K*theta) used
    by Beta-BIQAE, matching the paper's statement that "in the high-contrast
    limit, the likelihood used by CABIQAE collapses to the ideal amplified
    model" (Section 2.2.2).
    """

    def __init__(self, rho_min: float = 2.0, n_prior_samples: int = 2000, delta: float = 1e-6):
        super().__init__(
            c0=1.0,
            tau_c=1e12,
            b=0.5,
            rho_min=rho_min,
            n_prior_samples=n_prior_samples,
            delta=delta,
        )

    def __repr__(self) -> str:  # noqa: D105
        return "BIQAE(noise_naive=True)"


class DirectCircuitSampling:
    """Unamplified k=0 Bernoulli-sampling baseline (DCS).

    Corresponds to repeated unamplified sampling of the circuit success bit,
    the circuit-level analogue of classical Monte Carlo sampling (Section
    3.2.2). Error scales as O(1/sqrt(N_q)) rather than the amplified
    O(1/N_q) rate.
    """

    def estimate(self, circuit_executor: CircuitExecutor, n_shots: int) -> EstimationResult:
        """Run n_shots unamplified measurements and return the empirical mean.

        Args:
            circuit_executor: callable (k, n_shots) -> (n_success, n_shots);
                always called with k=0.
            n_shots: total number of shots.

        Returns:
            EstimationResult with a normal-approximation 90% interval.
        """
        n_success, n = circuit_executor(0, n_shots)
        p_hat = n_success / n
        se = np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / n)
        z = 1.645  # 90% two-sided normal quantile, matching alpha=0.10 convention
        return EstimationResult(
            a_hat=p_hat,
            a_lower=max(0.0, p_hat - z * se),
            a_upper=min(1.0, p_hat + z * se),
            total_queries=n,
            max_k=0,
            n_stages=1,
        )


class BAE:
    """Simplified Bayesian Amplitude Estimation baseline (Ramôa & Santos 2025).

    Uses sequential-Monte-Carlo-style particle propagation over the latent
    angle theta, with an ideal (or user-supplied contrast-aware) likelihood.
    This is a simplified reimplementation matching the paper's description
    that BAE is "Bayesian, adaptive, competitive in the ideal regime" but
    incurs greater classical post-processing cost than CABIQAE due to
    unrestricted particle-based posterior propagation (vs. CABIQAE's
    lightweight conjugate Beta updates).

    SIR note: BAE's exact resampling/refinement hyperparameters
    (Appendix A.1, Table 10: warm-up shots, particle count, resampling
    threshold, refinement parameters) are cited by name in the paper but not
    fully re-derived here; see architecture_plan.json risk_assessment.

    Args:
        n_particles: number of posterior particles (paper uses 300).
        contrast_model: optional CABIQAE instance supplying a (possibly
            noise-aware) p_obs(theta, k) likelihood; if None, uses the ideal
            sin^2(K*theta) model.
    """

    def __init__(self, n_particles: int = 300, contrast_model: Optional[CABIQAE] = None):
        self.n_particles = n_particles
        self.contrast_model = contrast_model or CABIQAE(c0=1.0, tau_c=1e12, b=0.5)

    def __repr__(self) -> str:  # noqa: D105
        return f"BAE(n_particles={self.n_particles})"

    def estimate(
        self,
        circuit_executor: CircuitExecutor,
        epsilon: float,
        k_schedule: Optional[list] = None,
        n_batch: int = 256,
        rng: Optional[np.random.Generator] = None,
    ) -> EstimationResult:
        """Run a simplified particle-filter BAE loop.

        Args:
            circuit_executor: callable (k, n_shots) -> (n_success, n_shots).
            epsilon: target half-width for stopping.
            k_schedule: sequence of Grover powers to sweep (default:
                exponentially growing schedule 0,1,2,4,8,...).
            n_batch: shots per stage.
            rng: optional NumPy random generator for particle resampling.

        Returns:
            EstimationResult from the particle-weighted posterior mean.
        """
        rng = rng or np.random.default_rng()
        k_schedule = k_schedule or [0] + [2**i for i in range(0, 12)]

        particles = rng.uniform(0, np.pi / 2, self.n_particles)
        log_weights = np.zeros(self.n_particles)
        total_queries = 0
        max_k_reached = 0

        for k in k_schedule:
            n_success, n_shots = circuit_executor(k, n_batch)
            total_queries += n_shots * (2 * k + 1)
            max_k_reached = max(max_k_reached, k)

            p_pred = self.contrast_model.p_obs(particles, k)
            p_pred = np.clip(p_pred, 1e-6, 1 - 1e-6)
            log_lik = n_success * np.log(p_pred) + (n_shots - n_success) * np.log(1 - p_pred)
            log_weights += log_lik
            log_weights -= log_weights.max()
            weights = np.exp(log_weights)
            weights /= weights.sum()

            ess = 1.0 / np.sum(weights**2)
            if ess < 0.4 * self.n_particles:
                idx = rng.choice(self.n_particles, size=self.n_particles, p=weights)
                particles = particles[idx] + rng.normal(0, 0.01, self.n_particles)
                particles = np.clip(particles, 0, np.pi / 2)
                log_weights = np.zeros(self.n_particles)
                weights = np.full(self.n_particles, 1.0 / self.n_particles)

            a_particles = np.sin(particles) ** 2
            a_mean = float(np.sum(weights * a_particles))
            a_var = float(np.sum(weights * (a_particles - a_mean) ** 2))
            half_width = 1.645 * np.sqrt(max(a_var, 0.0))
            if half_width <= epsilon:
                break

        return EstimationResult(
            a_hat=a_mean,
            a_lower=max(0.0, a_mean - half_width),
            a_upper=min(1.0, a_mean + half_width),
            total_queries=total_queries,
            max_k=max_k_reached,
            n_stages=len(k_schedule),
        )
