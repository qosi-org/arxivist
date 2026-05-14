"""
models/signature_vol.py
───────────────────────
Core signature-based stochastic volatility model.
Orchestrates the full offline precomputation and returns objects ready for calibration.

Implements Section 4 of the paper:
  - Eq. (4.1)–(4.3): signature-driven SDE model
  - Proposition 4.2: closed-form MC evaluation formula
  - Section 4.3: the algorithm (simulate → signatures → Q → calibrate)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from volsig.models.primary_process import PrimaryProcessSimulator
from volsig.signatures.compute import SignatureComputer, QMatrixAssembler, sig_dimension
from volsig.pricing.mc_pricer import (
    SignatureMCPricer,
    MultiMaturityPricer,
    compute_stochastic_integrals,
)
from volsig.utils.config import Config


class SignatureVolModel:
    """
    Full signature-based stochastic volatility model pipeline.

    Given a config, this class:
    1. Simulates Brownian motions W, B and the primary process X.
    2. Time-augments X → X̂ = (t, X).
    3. Computes truncated signatures S(X̂)^{≤N} and S(X̂)^{≤2N+1}.
    4. Assembles the Q-matrix and its Cholesky factor U.
    5. Computes stochastic integrals ∫vec(S)dZ.
    6. Returns a MultiMaturityPricer ready for calibration.

    All steps 1–5 are offline (computed once). Step 6 feeds into
    SignatureCalibrator which optimises ℓ online.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.N = cfg.model.signature_truncation_N
        self.d = 2  # always 2: (time, primary_process)
        self.n_coords = sig_dimension(self.d, self.N)   # 15 for N=3
        self.n_ext = sig_dimension(self.d, 2 * self.N + 1)  # 255 for N=3

        self.sig_computer = SignatureComputer(d=self.d, N=self.N)
        self.ext_sig_computer = SignatureComputer(d=self.d, N=2 * self.N + 1)
        self.q_assembler = QMatrixAssembler(N=self.N, d=self.d)

    def __repr__(self) -> str:
        return (f"SignatureVolModel(N={self.N}, n_coords={self.n_coords}, "
                f"primary={self.cfg.model.primary_process})")

    def precompute(
        self,
        maturity: float,
        seed: Optional[int] = None,
    ) -> Tuple[SignatureMCPricer, np.ndarray]:
        """
        Run the full offline precomputation for a single maturity T.

        Steps (Section 4.3 algorithm):
          1. Simulate W, B → Z = ρW + √(1-ρ²)B
          2. Simulate primary process X via Euler
          3. Time-augment: X̂ = (t, X)
          4. Compute S(X̂)^{≤N} paths [nMC, T_steps+1, 15]  and
                     S(X̂)^{≤2N+1} terminal [nMC, 255]
          5. Compute stochastic integrals ∫vec(S^{≤N}_s)dZ_s [nMC, 15]
          6. Assemble Q(T) [nMC, 15, 15] and U(T) = chol(-Q) [nMC, 15, 15]

        Args:
            maturity: Option maturity T (years).
            seed:     Random seed.

        Returns:
            pricer:    SignatureMCPricer for this maturity.
            W:         [nMC, T_steps] BM increments W (needed for primary process in correlated models).
        """
        cfg = self.cfg
        nMC = cfg.simulation.nMC
        dt = 1.0 / cfg.simulation.T_steps_per_unit
        T_steps = max(int(maturity / dt), 1)
        dt_actual = maturity / T_steps

        if seed is None:
            seed = cfg.simulation.seed

        print(f"\n[SignatureVolModel] Precomputing for T={maturity:.2f}  "
              f"(nMC={nMC}, T_steps={T_steps}, dt={dt_actual:.5f})")

        rng = np.random.default_rng(seed)

        # ── Step 1: Simulate BMs ──────────────────────────────────────────
        t0 = time.time()
        sqrt_dt = np.sqrt(dt_actual)
        W = rng.standard_normal((nMC, T_steps)) * sqrt_dt   # [nMC, T_steps]
        B = rng.standard_normal((nMC, T_steps)) * sqrt_dt   # [nMC, T_steps]

        # Get correlation from config
        rho_asset = self._get_rho()
        dZ = rho_asset * W + np.sqrt(1.0 - rho_asset ** 2) * B  # [nMC, T_steps]
        print(f"  BM simulation: {time.time()-t0:.1f}s")

        # ── Step 2: Simulate primary process ─────────────────────────────
        t0 = time.time()
        X = PrimaryProcessSimulator.simulate(
            variant=cfg.model.primary_process,
            W=W,
            dt=dt_actual,
            nMC=nMC,
            T_steps=T_steps,
            **self._primary_kwargs(),
        )  # [nMC, T_steps+1]
        print(f"  Primary process: {time.time()-t0:.1f}s")

        # ── Step 3: Time augmentation ─────────────────────────────────────
        X_aug = self.sig_computer.time_augment(X, dt_actual)   # [nMC, T_steps+1, 2]

        # ── Step 4: Signatures ────────────────────────────────────────────
        t0 = time.time()
        # Paths needed for stochastic integral (shape [nMC, T_steps+1, 15])
        sig_paths = self.sig_computer.compute_signature_paths(X_aug)  # [nMC, T+1, 15]
        # Terminal extended signature for Q matrix (shape [nMC, 255])
        sig_ext = self.ext_sig_computer.compute_terminal_signature(X_aug)  # [nMC, 255]
        print(f"  Signatures (N={self.N}, N_ext={2*self.N+1}): {time.time()-t0:.1f}s")

        # ── Step 5: Stochastic integrals ──────────────────────────────────
        t0 = time.time()
        stoch_int = compute_stochastic_integrals(sig_paths, dZ)  # [nMC, 15]
        print(f"  Stochastic integrals: {time.time()-t0:.1f}s")

        # ── Step 6: Q matrix and Cholesky ─────────────────────────────────
        t0 = time.time()
        Q = self.q_assembler.assemble(sig_ext)       # [nMC, 15, 15]
        U = self.q_assembler.cholesky(
            Q, eps=cfg.simulation.cholesky_reg_eps
        )                                             # [nMC, 15, 15]
        print(f"  Q matrix + Cholesky: {time.time()-t0:.1f}s")

        pricer = SignatureMCPricer(
            U=U,
            stoch_int=stoch_int,
            S0=cfg.model.S0,
            r=cfg.model.r,
            maturity=maturity,
        )
        return pricer, W

    def build_multi_maturity_pricer(
        self,
        maturities: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> MultiMaturityPricer:
        """
        Precompute pricers for all calibration maturities.

        Args:
            maturities: List of maturities. Defaults to cfg.calibration.maturities.
            seed:       Base random seed. Each maturity gets seed + i for independence.

        Returns:
            MultiMaturityPricer with one SignatureMCPricer per maturity.
        """
        if maturities is None:
            maturities = self.cfg.calibration.maturities
        if seed is None:
            seed = self.cfg.simulation.seed

        pricers = {}
        for i, T in enumerate(maturities):
            pricer, _ = self.precompute(maturity=T, seed=seed + i)
            pricers[T] = pricer

        return MultiMaturityPricer(
            pricers=pricers,
            strikes=np.array(self.cfg.calibration.strikes),
            S0=self.cfg.model.S0,
            r=self.cfg.model.r,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_rho(self) -> float:
        """Get asset-vol correlation from config based on primary process."""
        if self.cfg.model.primary_process == "heston_variance":
            return self.cfg.heston_primary.rho_asset_vol
        else:
            return self.cfg.fbm_primary.rho_asset_vol

    def _primary_kwargs(self) -> dict:
        """Build keyword arguments for PrimaryProcessSimulator.simulate()."""
        p = self.cfg.model.primary_process
        if p == "heston_variance":
            h = self.cfg.heston_primary
            return {
                "X0": h.X0,
                "nu": h.nu,
                "kappa": h.kappa,
                "theta": h.theta,
            }
        elif p in ("fbm_raw", "fbm_exp"):
            return {"H": self.cfg.fbm_primary.H}
        elif p == "fbm_shifted_exp":
            return {
                "X0": self.cfg.fbm_primary.X0,
                "H": self.cfg.fbm_primary.H,
            }
        else:
            raise ValueError(f"Unknown primary_process: {p}")
