"""
error_mitigation.py — Zero-Noise Extrapolation (ZNE)

Implements the error mitigation technique used in Havlicek et al. (2018),
Supplementary Section "Error mitigation for kernel estimation":

    "We again apply the error-mitigation protocol to first order.
     The kernel entries are obtained by running a time-stretched copy
     of the circuit and reporting the mitigated entry."

Richardson first-order extrapolation with two scale factors λ_1, λ_2:
    E_mitigated = (λ_2 · E(λ_1) − λ_1 · E(λ_2)) / (λ_2 − λ_1)

For λ_1=1.0, λ_2=1.5 (paper values):
    E_mitigated = (1.5 · E_normal − 1.0 · E_stretched) / (1.5 − 1.0)
                = 3 · E_normal − 2 · E_stretched

In simulation: gate stretching is approximated by scaling depolarising error
rates (ASSUMED — conf=0.65). On real hardware, pulse durations are scaled.

This module is disabled by default (config: error_mitigation.enabled=false).
Set enabled=true and provide a depolarising error rate for noisy simulation.

WARNING: Low-confidence implementation (conf=0.72). The simulation
approximation (scaled depolarising rates) differs from the paper's hardware
implementation (scaled pulse durations). Use for qualitative comparison only.

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


class ZeroNoiseExtrapolation:
    """
    Richardson first-order Zero-Noise Extrapolation (ZNE).

    WARNING: Low-confidence implementation — conf=0.72.
    TODO: Verify Richardson formula and noise-scaling approximation against
    paper's hardware ZNE implementation (gate-time stretching).

    Parameters
    ----------
    scale_factors : List[float]
        Noise scale factors. Paper uses [1.0, 1.5].
    order : int
        Extrapolation order. Paper uses first order (order=1, requires 2 points).
    """

    def __init__(
        self,
        scale_factors: Optional[List[float]] = None,
        order: int = 1,
    ) -> None:
        if scale_factors is None:
            scale_factors = [1.0, 1.5]   # Paper values — conf=0.95
        if len(scale_factors) < 2:
            raise ValueError("Need at least 2 scale factors for Richardson extrapolation")
        if order != 1:
            raise NotImplementedError("Only first-order Richardson extrapolation implemented")

        self.scale_factors = scale_factors
        self.order = order

    def mitigate_expectation(
        self,
        noisy_values: List[float],
        scale_factors: Optional[List[float]] = None,
    ) -> float:
        """
        Apply Richardson first-order extrapolation to two noisy expectation values.

        Formula:
            E_mitigated = (λ_2 · E(λ_1) − λ_1 · E(λ_2)) / (λ_2 − λ_1)

        For scale_factors=[1.0, 1.5] (paper values):
            E_mitigated = 3 · E(1.0) − 2 · E(1.5)

        Parameters
        ----------
        noisy_values : List[float]
            [E(λ_1), E(λ_2)] — expectation values at each noise scale.
        scale_factors : List[float], optional
            Override self.scale_factors.

        Returns
        -------
        float : zero-noise extrapolated expectation value
        """
        sf = scale_factors if scale_factors is not None else self.scale_factors
        if len(noisy_values) != len(sf):
            raise ValueError(
                f"len(noisy_values)={len(noisy_values)} != len(scale_factors)={len(sf)}"
            )

        lam1, lam2 = sf[0], sf[1]
        e1, e2 = noisy_values[0], noisy_values[1]

        # Richardson first-order extrapolation  [Supp. "Error mitigation for kernel estimation"]
        # WARNING: low-confidence — approximation of hardware gate stretching — conf=0.65
        e_mitigated = (lam2 * e1 - lam1 * e2) / (lam2 - lam1)
        return float(e_mitigated)

    def apply_to_expectation(
        self,
        circuit: QuantumCircuit,
        observable_fn: Callable[[QuantumCircuit, AerSimulator], float],
        base_error_rate: float = 0.01,
    ) -> float:
        """
        Run circuit at each noise scale, then extrapolate.

        Noise scaling is approximated by adjusting depolarising error rates.
        ASSUMED: This approximates hardware gate-time stretching — conf=0.60.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit without noise model.
        observable_fn : Callable[[QuantumCircuit, AerSimulator], float]
            Function that executes circuit on backend and returns expectation value.
        base_error_rate : float
            Depolarising error rate at scale factor 1.0.
            ASSUMED: 0.01 as rough hardware approximation — conf=0.60.

        Returns
        -------
        float : zero-noise extrapolated expectation value
        """
        # WARNING: low-confidence implementation — conf=0.65
        # TODO: Replace scaled depolarising rate with proper gate-time stretching
        #       if executing on real hardware or Qiskit Pulse.
        values = []
        for scale in self.scale_factors:
            noise_model = self._build_noise_model(circuit, base_error_rate * scale)
            backend = AerSimulator(noise_model=noise_model)
            val = observable_fn(circuit, backend)
            values.append(val)

        return self.mitigate_expectation(values)

    def _build_noise_model(
        self,
        circuit: QuantumCircuit,
        error_rate: float,
    ) -> NoiseModel:
        """
        Build a depolarising noise model for the given circuit.

        ASSUMED: uniform depolarising noise on all gates — conf=0.60.
        Paper uses gate-specific noise characterised by randomised benchmarking.
        """
        noise_model = NoiseModel()
        n_qubits = circuit.num_qubits

        # 1-qubit depolarising
        error_1q = depolarizing_error(error_rate, 1)
        # 2-qubit depolarising (typically ~10x higher)
        error_2q = depolarizing_error(min(error_rate * 10.0, 1.0), 2)

        # Apply to all single-qubit gate types
        for gate in ["h", "ry", "rz", "x", "y", "z"]:
            noise_model.add_all_qubit_quantum_error(error_1q, gate)

        # Apply to 2-qubit gates
        for gate in ["cx", "cz"]:
            noise_model.add_all_qubit_quantum_error(error_2q, gate)

        return noise_model

    def __repr__(self) -> str:
        return (
            f"ZeroNoiseExtrapolation("
            f"scale_factors={self.scale_factors}, "
            f"order={self.order})"
        )


class NoOpMitigation:
    """
    Passthrough (no error mitigation).

    Used when config.error_mitigation.enabled=False (default for ideal simulation).
    Provides the same API as ZeroNoiseExtrapolation for transparent swapping.
    """

    def mitigate_expectation(
        self,
        noisy_values: List[float],
        scale_factors: Optional[List[float]] = None,
    ) -> float:
        """Return the first (unscaled) value unchanged."""
        return float(noisy_values[0])

    def apply_to_expectation(
        self,
        circuit: QuantumCircuit,
        observable_fn: Callable[[QuantumCircuit, AerSimulator], float],
        base_error_rate: float = 0.0,
    ) -> float:
        backend = AerSimulator(method="statevector")
        return observable_fn(circuit, backend)

    def __repr__(self) -> str:
        return "NoOpMitigation()"


def build_mitigation(
    enabled: bool,
    scale_factors: Optional[List[float]] = None,
    order: int = 1,
):
    """
    Factory function: return ZNE or passthrough based on config.

    Parameters
    ----------
    enabled : bool  (config.error_mitigation.enabled)
    scale_factors : List[float], optional
    order : int

    Returns
    -------
    ZeroNoiseExtrapolation or NoOpMitigation
    """
    if enabled:
        return ZeroNoiseExtrapolation(scale_factors=scale_factors, order=order)
    return NoOpMitigation()
