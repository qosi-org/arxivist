"""Depolarising noise channel, hardware-calibrated noise parameters, and readout error.

Implements: Architecture Plan module `models/noise_channels.py`.
Paper sections: Definition 3.8 (depolarising channel), Proposition 3.9 (noisy density
operator), Proposition 3.12 (noisy measurement probability), Section 3.6 (hardware
calibration of lambda_V, lambda_U), Definition 3.19 (readout error / confusion matrix).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DepolarisingChannel:
    """The depolarising quantum channel and its effect on measurement probabilities.

    Paper reference: Definition 3.8, Proposition 3.9, Proposition 3.12 (Eq. 3.7).
    """

    def apply_to_density_matrix(self, rho: np.ndarray, lam: float) -> np.ndarray:
        """Apply the d-dimensional depolarising channel Delta_lambda(rho).

        Delta_lambda(rho) = (1 - lambda) * rho + (lambda / d) * I   [Definition 3.8]

        Args:
            rho: density matrix, shape [d, d] complex, Hermitian, trace 1.
            lam: depolarising parameter, lambda in [0, 1 + 1/(d^2-1)].

        Returns:
            The depolarised density matrix, shape [d, d].
        """
        assert rho.ndim == 2 and rho.shape[0] == rho.shape[1], (
            f"Expected a square density matrix [d,d], got shape {rho.shape}"
        )
        d = rho.shape[0]
        identity = np.eye(d, dtype=rho.dtype)
        return (1.0 - lam) * rho + (lam / d) * identity

    def noisy_probability(
        self,
        p_noiseless: float,
        alpha: float,
        n_accuracy_blocks: int,
        n_qubits: int,
    ) -> float:
        """Noisy outcome probability under depolarising noise (Proposition 3.12, Eq. 3.7).

        P~_m = alpha * P_m + (1 - alpha) * (n_accuracy_blocks / 2^n_qubits)
        with alpha = (1 - lambda_V) * (1 - lambda_U).

        Args:
            p_noiseless: the noiseless probability P_m.
            alpha: hardware fidelity factor alpha = (1-lambda_V)(1-lambda_U).
            n_accuracy_blocks: n (number of accuracy blocks).
            n_qubits: number of qubits (the "mathfrak n" in the paper).

        Returns:
            The noisy probability P~_m.
        """
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0,1], got {alpha}"
        offset = n_accuracy_blocks / (2 ** n_qubits)
        return alpha * p_noiseless + (1.0 - alpha) * offset


@dataclass
class HardwareNoiseCalibrator:
    """Computes effective depolarising parameters lambda_V, lambda_U from hardware
    calibration data (Section 3.6), and the combined fidelity factor alpha.
    """

    def compute_lambda_V(self, eps_1q: float, n_qubits: int) -> float:
        """Effective depolarising parameter for the V gate (state preparation).

        V is built from N1Q = n_qubits - 2 single-qubit Hadamard gates (Section 3.6):
            lambda_V = 1 - (1 - eps_1Q)^(n_qubits - 2)

        Args:
            eps_1q: single-qubit gate error rate (e.g. IBM's sqrt(X) gate error).
            n_qubits: total qubit count "n".

        Returns:
            lambda_V in [0, 1].
        """
        n1q = max(n_qubits - 2, 0)
        return 1.0 - (1.0 - eps_1q) ** n1q

    def compute_lambda_U_gate_only(
        self, eps_2q: float, n2q_gate_count: int
    ) -> float:
        """Effective depolarising parameter for U from two-qubit gate errors alone
        (before folding in decoherence).

        lambda_U(gate) = 1 - (1 - eps_2Q)^N2Q   [Section 3.6, "Noisy U gate"]
        """
        return 1.0 - (1.0 - eps_2q) ** n2q_gate_count

    def naive_and_ucr_two_qubit_gate_counts(
        self, n_accuracy_blocks: int, n_qubits: int
    ) -> tuple[int, float]:
        """Two-qubit gate counts for the naive vs. UCR-decomposed implementation of U.

        Naive:  N2Q_naive = n_accuracy_blocks * n_qubits
        UCR:    N2Q_ucr   ~= (n_accuracy_blocks * n_qubits) / 15   [Section 3.6, 4.1]

        Returns:
            (N2Q_naive, N2Q_ucr) -- N2Q_ucr is float since the /15 factor is only
            approximate in the paper ("reduces this by a factor of roughly 15").
        """
        n2q_naive = n_accuracy_blocks * n_qubits
        n2q_ucr = n2q_naive / 15.0
        return n2q_naive, n2q_ucr

    def compute_lambda_U(
        self,
        eps_2q: float,
        n2q_gate_count: float,
        t1_us: float,
        t2_us: float,
        t2q_duration_ns: float,
    ) -> float:
        """Full effective lambda_U including gate error AND decoherence (Section 3.6).

        tcirc = N2Q * t2Q
        p_T1 = 1 - exp(-tcirc / T1),  p_T2 = 1 - exp(-tcirc / T2)
        lambda_U ~= 1 - (1 - lambda_U_gate)(1 - p_T1/2)(1 - p_T2/2)

        Args:
            eps_2q: two-qubit gate error rate.
            n2q_gate_count: number of two-qubit gates (use the UCR count from
                `naive_and_ucr_two_qubit_gate_counts`, not the naive count, to match
                the paper's hardware-calibrated experiments).
            t1_us: relaxation time T1, in microseconds.
            t2_us: dephasing time T2, in microseconds.
            t2q_duration_ns: duration of a single two-qubit gate, in nanoseconds.

        Returns:
            lambda_U in [0, 1] including decoherence contributions.
        """
        lambda_u_gate = self.compute_lambda_U_gate_only(eps_2q, n2q_gate_count)

        tcirc_ns = n2q_gate_count * t2q_duration_ns
        t1_ns = t1_us * 1000.0
        t2_ns = t2_us * 1000.0
        p_t1 = 1.0 - math.exp(-tcirc_ns / t1_ns)
        p_t2 = 1.0 - math.exp(-tcirc_ns / t2_ns)

        lambda_u = 1.0 - (1.0 - lambda_u_gate) * (1.0 - 0.5 * p_t1) * (1.0 - 0.5 * p_t2)
        return lambda_u

    def compute_alpha(self, lambda_v: float, lambda_u: float) -> float:
        """Combined hardware fidelity factor alpha = (1-lambda_V)(1-lambda_U)."""
        assert 0.0 <= lambda_v <= 1.0, f"lambda_v out of range: {lambda_v}"
        assert 0.0 <= lambda_u <= 1.0, f"lambda_u out of range: {lambda_u}"
        return (1.0 - lambda_v) * (1.0 - lambda_u)


@dataclass
class ReadoutErrorModel:
    """Classical readout (measurement) bit-flip error model (Definition 3.19).

    NOTE: the readout error probability `p` used by this class is, for ibm_fez,
    an ASSUMED value (SIR implementation_assumptions[5], confidence 0.35) --
    it is not listed in the paper's own Appendix A hardware table despite being
    used in the reported total error bound (Eq. 4.4). Replace `p` with a live
    calibration value before trusting hardware-bound outputs.
    """

    @staticmethod
    def _hamming_distance(m: int, m_prime: int, n_bits: int = 2) -> int:
        """Hamming distance between the n_bits-bit binary representations of m, m'."""
        return bin(m ^ m_prime).count("1")

    def confusion_matrix(self, p: float) -> np.ndarray:
        """Build the 4x4 readout confusion matrix Q (Definition 3.19).

        q[m, m'] = p^H(m,m') * (1-p)^(2 - H(m,m'))
        where H is the Hamming distance between the 2-bit encodings of m, m'.

        Args:
            p: single-qubit readout (bit-flip) error probability.

        Returns:
            Q, shape [4, 4], rows sum to 1.
        """
        assert 0.0 <= p <= 1.0, f"readout error p must be in [0,1], got {p}"
        q = np.zeros((4, 4), dtype=np.float64)
        for m in range(4):
            for m_prime in range(4):
                h = self._hamming_distance(m, m_prime)
                q[m, m_prime] = (p ** h) * ((1 - p) ** (2 - h))
        return q

    def apply_readout_error(self, probs_true: np.ndarray, p: float) -> np.ndarray:
        """Map true outcome probabilities through the readout confusion matrix.

        P_m = sum_{m'} q[m, m'] * P~_{m'}   (Definition 3.19)

        Args:
            probs_true: true (pre-readout-error) probabilities, shape [4].
            p: readout bit-flip error probability.

        Returns:
            Observed probabilities after classical readout error, shape [4].
        """
        assert probs_true.shape == (4,), f"Expected shape [4], got {probs_true.shape}"
        q = self.confusion_matrix(p)
        return q @ probs_true
