"""Orchestrates the three circuit-parameter optimisation methods compared in the paper.

Implements: Architecture Plan module `training/trainer.py`.
Paper section: Section 4.1, "Parameters optimisation" (Methods A: L-BFGS-B,
B: two-stage, C: Adam).

Design note: fitting theta requires many thousands of forward evaluations of the
QNN during optimisation. The paper validates (Section 4.1.1) that its circuit's
expected output exactly matches the closed-form reference formula

    f^R_{n,theta}(x) = (1/n) * sum_i R * cos(gamma_i) * cos(b_i + a_i . x)

so, following standard practice for training highly-shot-noise-sensitive circuits,
this trainer optimises theta against that closed-form expectation (which the
real shot-sampled circuit converges to as N_shots -> infinity) rather than
re-running a full Qiskit simulation at every optimiser step. `models/qnn_circuit.py`
and `models/measurement.py` remain the ground truth for actually *executing* the
circuit (simulator or hardware) at evaluation/inference time -- see
`evaluate.py` / `run_hardware.py`. This is a documented engineering choice, not a
paper-stated detail (SIR training_pipeline confidence 0.45); swap
`_closed_form_forward_batch` for a real shot-based forward pass if exact
shot-noise-aware training is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize

from noisy_qnn_uat.models.postprocessing import AffineNoiseCancellation


def _pack_theta(theta: list[tuple[np.ndarray, float, float]]) -> np.ndarray:
    """Flatten a list of (a_k, b_k, gamma_k) tuples into a 1D array for scipy."""
    parts = []
    for a_k, b_k, gamma_k in theta:
        parts.append(np.concatenate([np.asarray(a_k, dtype=np.float64).ravel(),
                                      [b_k], [gamma_k]]))
    return np.concatenate(parts)


def _unpack_theta(flat: np.ndarray, n_accuracy_blocks: int, d: int) -> list[tuple[np.ndarray, float, float]]:
    """Inverse of `_pack_theta`: reshape a flat array back into (a_k, b_k, gamma_k) tuples."""
    block_size = d + 2
    theta = []
    for k in range(n_accuracy_blocks):
        block = flat[k * block_size:(k + 1) * block_size]
        a_k, b_k, gamma_k = block[:d], float(block[d]), float(block[d + 1])
        theta.append((a_k, b_k, gamma_k))
    return theta


def _closed_form_forward_batch(
    flat_theta: np.ndarray, x_batch: np.ndarray, R: float, n_accuracy_blocks: int, d: int
) -> np.ndarray:
    """Vectorised closed-form QNN output over a batch of inputs (see module docstring)."""
    theta = _unpack_theta(flat_theta, n_accuracy_blocks, d)
    preds = np.zeros(x_batch.shape[0], dtype=np.float64)
    for a_k, b_k, gamma_k in theta:
        preds += math.cos(gamma_k) * np.cos(b_k + x_batch @ a_k)
    return (R / n_accuracy_blocks) * preds


@dataclass
class QNNTrainer:
    """Fits circuit parameters theta using one of the paper's three compared methods."""

    n_accuracy_blocks: int
    input_dim: int
    R: float
    seed: int = 42

    def _init_theta_flat(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        block_size = self.input_dim + 2
        return rng.uniform(-1.0, 1.0, size=self.n_accuracy_blocks * block_size)

    def fit_method_a_lbfgsb(
        self, theta_init: np.ndarray | None, x_train: np.ndarray, y_train: np.ndarray
    ) -> np.ndarray:
        """Method A: simultaneous L-BFGS-B optimisation of all parameters (a,b,gamma).

        Args:
            theta_init: flat initial parameter vector, or None to random-initialise.
            x_train: training inputs, shape [n_train, input_dim] (normalised).
            y_train: training targets, shape [n_train].

        Returns:
            Optimised flat theta parameter vector.
        """
        x0 = theta_init if theta_init is not None else self._init_theta_flat()

        def objective(flat_theta: np.ndarray) -> float:
            preds = _closed_form_forward_batch(
                flat_theta, x_train, self.R, self.n_accuracy_blocks, self.input_dim
            )
            return float(np.mean((preds - y_train) ** 2))

        result = minimize(objective, x0, method="L-BFGS-B")
        return result.x

    def fit_method_b_two_stage(
        self, theta_init: np.ndarray | None, x_train: np.ndarray, y_train: np.ndarray
    ) -> np.ndarray:
        """Method B: first optimise (a,b) with gamma fixed at 0 (cos(gamma)=1),
        then refine gamma with (a,b) fixed (Section 4.1, "two-stage").

        Args:
            theta_init: flat initial parameter vector, or None to random-initialise.
            x_train: training inputs, shape [n_train, input_dim].
            y_train: training targets, shape [n_train].

        Returns:
            Optimised flat theta parameter vector.
        """
        x0 = theta_init if theta_init is not None else self._init_theta_flat()
        block_size = self.input_dim + 2
        theta0 = _unpack_theta(x0, self.n_accuracy_blocks, self.input_dim)

        # --- Stage 1: fix gamma=0, optimise (a,b) only ---
        ab_flat0 = np.concatenate(
            [np.concatenate([a_k, [b_k]]) for a_k, b_k, _ in theta0]
        )

        def stage1_objective(ab_flat: np.ndarray) -> float:
            theta = []
            ab_block = self.input_dim + 1
            for k in range(self.n_accuracy_blocks):
                block = ab_flat[k * ab_block:(k + 1) * ab_block]
                theta.append((block[: self.input_dim], float(block[self.input_dim]), 0.0))
            preds = np.zeros(x_train.shape[0])
            for a_k, b_k, gamma_k in theta:
                preds += math.cos(gamma_k) * np.cos(b_k + x_train @ a_k)
            preds = (self.R / self.n_accuracy_blocks) * preds
            return float(np.mean((preds - y_train) ** 2))

        stage1_result = minimize(stage1_objective, ab_flat0, method="L-BFGS-B")
        ab_block = self.input_dim + 1
        theta_stage1 = []
        for k in range(self.n_accuracy_blocks):
            block = stage1_result.x[k * ab_block:(k + 1) * ab_block]
            theta_stage1.append((block[: self.input_dim], float(block[self.input_dim]), 0.0))

        # --- Stage 2: fix (a,b) from stage 1, refine gamma only ---
        gamma0 = np.zeros(self.n_accuracy_blocks)

        def stage2_objective(gamma_flat: np.ndarray) -> float:
            preds = np.zeros(x_train.shape[0])
            for k, (a_k, b_k, _) in enumerate(theta_stage1):
                preds += math.cos(gamma_flat[k]) * np.cos(b_k + x_train @ a_k)
            preds = (self.R / self.n_accuracy_blocks) * preds
            return float(np.mean((preds - y_train) ** 2))

        stage2_result = minimize(stage2_objective, gamma0, method="L-BFGS-B")

        final_theta = [
            (a_k, b_k, float(stage2_result.x[k]))
            for k, (a_k, b_k, _) in enumerate(theta_stage1)
        ]
        return _pack_theta(final_theta)

    def fit_method_c_adam(
        self,
        theta_init: np.ndarray | None,
        x_train: np.ndarray,
        y_train: np.ndarray,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        max_iterations: int = 1000,
    ) -> np.ndarray:
        """Method C: Adam gradient descent with adaptive learning rates (Section 4.1).

        NOTE (SIR implementation_assumptions[0], confidence 0.3): lr/beta1/beta2 and
        max_iterations are ASSUMED defaults -- the paper does not state them.

        Args:
            theta_init: flat initial parameter vector, or None to random-initialise.
            x_train: training inputs, shape [n_train, input_dim].
            y_train: training targets, shape [n_train].
            lr: Adam learning rate. ASSUMED default 1e-3.
            beta1: Adam beta1. ASSUMED default 0.9.
            beta2: Adam beta2. ASSUMED default 0.999.
            max_iterations: number of gradient steps. ASSUMED default 1000.

        Returns:
            Optimised flat theta parameter vector.
        """
        x0 = theta_init if theta_init is not None else self._init_theta_flat()
        theta0 = _unpack_theta(x0, self.n_accuracy_blocks, self.input_dim)

        a = torch.tensor(np.stack([t[0] for t in theta0]), requires_grad=True, dtype=torch.float64)
        b = torch.tensor([t[1] for t in theta0], requires_grad=True, dtype=torch.float64)
        gamma = torch.tensor([t[2] for t in theta0], requires_grad=True, dtype=torch.float64)

        x_train_t = torch.tensor(x_train, dtype=torch.float64)
        y_train_t = torch.tensor(y_train, dtype=torch.float64)

        optimizer = torch.optim.Adam([a, b, gamma], lr=lr, betas=(beta1, beta2))

        for _ in range(max_iterations):
            optimizer.zero_grad()
            phase = b.unsqueeze(1) + x_train_t @ a.T  # [n_train, n_accuracy_blocks]
            terms = torch.cos(gamma).unsqueeze(0) * torch.cos(phase.T).T  # broadcast
            preds = (self.R / self.n_accuracy_blocks) * terms.sum(dim=1)
            loss = torch.mean((preds - y_train_t) ** 2)
            loss.backward()
            optimizer.step()

        final_theta = [
            (a[k].detach().numpy(), float(b[k].detach()), float(gamma[k].detach()))
            for k in range(self.n_accuracy_blocks)
        ]
        return _pack_theta(final_theta)

    def fit_affine_correction(
        self, alpha: float, R: float, n_accuracy_blocks: int, n_qubits: int
    ) -> tuple[float, float]:
        """Fit (beta1, beta2) for the Theorem 3.15 affine noise-cancellation layer.

        Uses the closed-form solution directly (exact, no gradient descent needed)
        given a known hardware fidelity factor alpha; see
        `AffineNoiseCancellation.closed_form_correction`.

        Args:
            alpha: hardware fidelity factor.
            R: output scaling factor.
            n_accuracy_blocks: n (number of accuracy blocks).
            n_qubits: number of qubits.

        Returns:
            (beta1, beta2) tuple.
        """
        return AffineNoiseCancellation.closed_form_correction(
            alpha, R, n_accuracy_blocks, n_qubits
        )

    def predict(self, flat_theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the closed-form QNN output for a batch of inputs given fitted theta."""
        return _closed_form_forward_batch(
            flat_theta, x, self.R, self.n_accuracy_blocks, self.input_dim
        )
