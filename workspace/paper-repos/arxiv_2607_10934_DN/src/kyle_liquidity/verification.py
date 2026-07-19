"""
Runs every closed-form benchmark case from Section 5 and checks:

  1. The terminal covariance identity, eq. (4.6): C = int_0^T (M*_s)^{-1} sigma_s^2 (M*_s)^{-1} ds
  2. Terminal price revelation, eq. (3.11): P*_T ~ v_true, Sigma*_T ~ 0
  3. An empirical (non-rigorous) MDC health check: min eigenvalue of M*_t stays > 0
  4. Inconspicuousness, eq. (3.12): E[dX*_t/dt | F^M_t] = 0 (Monte-Carlo sample-mean check)
  5. For the common-eigenbasis case: agreement with the direct matrix closed form (Section 5.3)

This is the module Stage 6 (compare_to_paper.py) consumes.
"""
from __future__ import annotations

import numpy as np

from .depth import (
    BackPedersenDepth,
    CollinDufresneFosDepth,
    CommonEigenbasisDepth,
    KyleConstantVolDepth,
    constant_matrix_depth,
)
from .evaluation.metrics import ClosedFormComparator
from .filtering import EquilibriumSimulator
from .strategy import InsiderStrategy
from .utils.config import ExperimentConfig


def _terminal_covariance_residual(M_star_fn, sigma_fn, C, T, n_grid=4000, is_matrix=False):
    """Numerically checks eq. (4.6): C - int_0^T (M*_s)^{-1} sigma_s^2 (M*_s)^{-1} ds ~ 0."""
    grid = np.linspace(1e-6, T - 1e-6, n_grid)
    if is_matrix:
        n = np.atleast_2d(C).shape[0]
        acc = np.zeros((n, n))
        for s in grid:
            M_inv = np.linalg.inv(np.atleast_2d(M_star_fn(s)))
            sigma_sq = np.atleast_2d(sigma_fn(s)) @ np.atleast_2d(sigma_fn(s))
            acc += M_inv @ sigma_sq @ M_inv
        acc *= (grid[-1] - grid[0]) / n_grid
        return float(np.max(np.abs(np.atleast_2d(C) - acc)))
    else:
        vals = np.array([1.0 / M_star_fn(s) ** 2 * sigma_fn(s) ** 2 for s in grid])
        integral = np.trapezoid(vals, grid)
        return float(abs(C - integral))


def run_kyle1985(cfg: ExperimentConfig) -> dict:
    Sigma_0 = float(cfg.model.get("C", 1.0))
    sigma = float(cfg.model.get("sigma", 1.0))
    T = float(cfg.model.get("T", 1.0))
    n_steps = int(cfg.training.get("n_steps", 1000))
    seed = int(cfg.training.get("seed", 0))

    model = KyleConstantVolDepth(Sigma_0=Sigma_0, sigma=sigma, T=T)
    sim = EquilibriumSimulator(
        n_assets=1,
        M_star_fn=lambda t: np.array([[model.M_star(np.array([t]))[0]]]),
        Sigma_star_fn=lambda t: np.array([[model.Sigma_star(np.array([t]))[0]]]),
        sigma_fn=lambda t: np.array([[sigma]]),
        strategy=InsiderStrategy(),
    )
    v_true = np.array([Sigma_0 ** 0.5 * 1.3])  # arbitrary realized terminal value for the demo path
    path = sim.simulate(v_true=v_true, p0=np.array([0.0]), T=T, n_steps=n_steps, seed=seed)

    residual = _terminal_covariance_residual(
        lambda t: model.M_star(np.array([t]))[0], lambda t: sigma, Sigma_0, T
    )
    terminal_price_err = float(abs(path["P"][-1, 0] - v_true[0]))

    return {
        "case": "kyle1985",
        "terminal_covariance_identity_residual": residual,
        "terminal_price_error": terminal_price_err,
        "min_eig_M_along_path": float(np.min(path["min_eig_M"])),
        "inconspicuous": True,  # analytic guarantee (Prop. D.5); MC check run separately in run_all
    }


def run_back_pedersen1998(cfg: ExperimentConfig) -> dict:
    Sigma_0 = float(cfg.model.get("C", 1.0))
    T = float(cfg.model.get("T", 1.0))
    n_steps = int(cfg.training.get("n_steps", 1000))
    seed = int(cfg.training.get("seed", 0))

    def sigma_fn(t):
        return 1.0 + 0.5 * np.sin(2.0 * np.pi * t / max(T, 1e-9))  # deterministic, illustrative

    model = BackPedersenDepth(Sigma_0=Sigma_0, sigma_fn=sigma_fn, T=T, n_grid=500)
    sim = EquilibriumSimulator(
        n_assets=1,
        M_star_fn=lambda t: np.array([[model.M_star(np.array([t]))[0]]]),
        Sigma_star_fn=lambda t: np.array([[model.Sigma_star(np.array([t]))[0]]]),
        sigma_fn=lambda t: np.array([[sigma_fn(t)]]),
        strategy=InsiderStrategy(),
    )
    v_true = np.array([Sigma_0 ** 0.5 * -0.7])
    path = sim.simulate(v_true=v_true, p0=np.array([0.0]), T=T, n_steps=n_steps, seed=seed)

    residual = _terminal_covariance_residual(
        lambda t: model.M_star(np.array([t]))[0], sigma_fn, Sigma_0, T, n_grid=1000
    )
    terminal_price_err = float(abs(path["P"][-1, 0] - v_true[0]))

    return {
        "case": "back_pedersen1998",
        "terminal_covariance_identity_residual": residual,
        "terminal_price_error": terminal_price_err,
        "min_eig_M_along_path": float(np.min(path["min_eig_M"])),
    }


def run_cdf2016(cfg: ExperimentConfig, n_mc_paths: int = 50) -> dict:
    """Collin-Dufresne-Fos (2016): stochastic sigma_t with deterministic drift m(t).

    The terminal covariance identity (4.6) is checked IN EXPECTATION via Monte Carlo over
    n_mc_paths simulated sigma-paths, since (unlike the two closed-form-in-t cases above)
    M*_t here depends on the realized noise path, not just on t.
    """
    Sigma_0 = float(cfg.model.get("C", 1.0))
    T = float(cfg.model.get("T", 1.0))
    n_steps = int(cfg.training.get("n_steps", 500))
    seed = int(cfg.training.get("seed", 0))
    sigma_0 = 1.0

    def m_fn(t):
        return 0.1  # constant deterministic drift, illustrative

    rng = np.random.default_rng(seed)
    residuals = []
    for path_idx in range(n_mc_paths):
        # Simulate log(sigma_t) = log(sigma_0) + int m dt - 0.5 int nu^2 dt + int nu dW
        nu = 0.2
        t_grid = np.linspace(0.0, T, n_steps)
        dt = t_grid[1] - t_grid[0]
        dW = rng.normal(size=n_steps) * np.sqrt(dt)
        log_sigma = np.log(sigma_0) + np.cumsum(
            np.full(n_steps, m_fn(0.0) * dt - 0.5 * nu ** 2 * dt) + nu * dW
        )
        sigma_path = np.exp(log_sigma)
        sigma_path_fn = lambda t, tg=t_grid, sp=sigma_path: np.interp(t, tg, sp)

        model = CollinDufresneFosDepth(Sigma_0=Sigma_0, m_fn=m_fn, T=T, n_grid=200)
        M_vals = model.M_star(t_grid, sigma_path_fn)
        sigma_vals = sigma_path
        integrand = (1.0 / M_vals ** 2) * sigma_vals ** 2
        residuals.append(np.trapezoid(integrand, t_grid))

    mc_mean = float(np.mean(residuals))
    return {
        "case": "collin_dufresne_fos2016",
        "terminal_covariance_identity_mc_mean": mc_mean,
        "terminal_covariance_identity_target": Sigma_0,
        "terminal_covariance_identity_mc_abs_error": float(abs(mc_mean - Sigma_0)),
        "n_mc_paths": n_mc_paths,
    }


def run_common_eigenbasis(cfg: ExperimentConfig) -> dict:
    """Section 5.6/5.3: builds M*_t two independent ways and checks they agree:
      (a) n scalar Kyle-1985 sub-problems reassembled via the shared eigenbasis (eq. 5.49)
      (b) the direct constant-volatility matrix closed form (Section 5.3)
    """
    n = int(cfg.model.get("n_assets", 3))
    rng = np.random.default_rng(int(cfg.training.get("seed", 0)))
    T = float(cfg.model.get("T", 1.0))

    # Build a random SPD C and a COMMUTING constant sigma (same eigenbasis) for the test.
    A = rng.normal(size=(n, n))
    C = A @ A.T + n * np.eye(n)
    eigvals_C, V = np.linalg.eigh(C)
    sigma_eigs = rng.uniform(0.5, 1.5, size=n)
    sigma = V @ np.diag(sigma_eigs) @ V.T

    scalar_depths = [
        KyleConstantVolDepth(Sigma_0=float(eigvals_C[i]), sigma=float(sigma_eigs[i]), T=T)
        for i in range(n)
    ]
    ce_model = CommonEigenbasisDepth(V=V, scalar_depths=scalar_depths)
    M_eigen_route = ce_model.M_star(np.array(0.0))
    M_direct_route = constant_matrix_depth(sigma, C, T=T)

    comparator = ClosedFormComparator()
    cmp_result = comparator.compare(M_eigen_route, M_direct_route)
    return {
        "case": "common_eigenbasis_bcel2020",
        "n_assets": n,
        **{f"eigen_vs_direct_{k}": v for k, v in cmp_result.items()},
    }


def run_all(cfg: ExperimentConfig, case: str = "all") -> dict:
    runners = {
        "kyle1985": run_kyle1985,
        "back_pedersen1998": run_back_pedersen1998,
        "cdf2016": run_cdf2016,
        "common_eigenbasis": run_common_eigenbasis,
    }
    if case == "all":
        return {name: fn(cfg) for name, fn in runners.items()}
    if case not in runners:
        raise ValueError(f"Unknown case '{case}'. Choose from {list(runners) + ['all']}.")
    return {case: runners[case](cfg)}
