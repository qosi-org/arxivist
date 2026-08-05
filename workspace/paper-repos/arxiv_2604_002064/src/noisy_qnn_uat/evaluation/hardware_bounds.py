"""Analytical error bounds: Statement 2.1, Theorem 3.6, Theorem 3.17, Proposition 3.20.

Implements: Architecture Plan module `evaluation/hardware_bounds.py`.
Paper sections: Statement 2.1 (noiseless bound), Theorem 3.6 (general CPTP noise
bound), Theorem 3.17 / Eq. (3.8) (depolarising-noise bound, three terms), and
Proposition 3.20 / Eq. (4.4) (total bound including classical readout error --
the exact bound validated against ibm_fez hardware in Section 4.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorBoundCalculator:
    """Computes the paper's four nested analytical error bounds."""

    def noiseless_bound(self, l1_fhat: float, n_accuracy_blocks: int) -> float:
        """Statement 2.1 / Eq. (2.2)-adjacent noiseless bound: L^1[f-hat] / sqrt(n).

        Args:
            l1_fhat: L^1 norm of the Fourier transform of the target function f,
                L^1[f-hat] (paper notation: L^1[fb]). See Section 2.3 for worked
                examples (Gaussian, Bachelier Put, Black-Scholes Put).
            n_accuracy_blocks: n (number of accuracy blocks).

        Returns:
            The noiseless L^2(mu) error bound.
        """
        assert n_accuracy_blocks > 0, "n_accuracy_blocks must be positive"
        return l1_fhat / np.sqrt(n_accuracy_blocks)

    def cptp_noise_bound(
        self, l1_fhat: float, n_accuracy_blocks: int, R: float, fidelity_min: float
    ) -> float:
        """General CPTP-noise bound (Theorem 3.6).

        bound = L^1[f-hat]/sqrt(n) + 4R * sqrt(1 - fidelity_min^2)

        Args:
            l1_fhat: L^1[f-hat], as in `noiseless_bound`.
            n_accuracy_blocks: n (number of accuracy blocks).
            R: output scaling factor.
            fidelity_min: worst-case fidelity F_min (Theorem 3.6 definition),
                infimum over Kraus-index pairs, theta, and x of the trajectory
                fidelity F_{k,l}(theta,x).

        Returns:
            The Theorem 3.6 error bound.
        """
        assert 0.0 <= fidelity_min <= 1.0, f"fidelity_min must be in [0,1], got {fidelity_min}"
        noiseless_term = self.noiseless_bound(l1_fhat, n_accuracy_blocks)
        noise_term = 4.0 * R * np.sqrt(max(1.0 - fidelity_min ** 2, 0.0))
        return noiseless_term + noise_term

    def depolarising_bound(
        self,
        alpha: float,
        l1_fhat: float,
        n_accuracy_blocks: int,
        f_l2_norm: float,
        R: float,
        n_qubits: int,
    ) -> float:
        """Depolarising-noise bound, three terms (Theorem 3.17, Eq. 3.8).

        bound = alpha*L^1[f-hat]/sqrt(n)          [statistical]
              + (1-alpha)*||f||_{L^2(mu)}          [systematic]
              + R*(1-alpha)*(1 - 4n/2^{n_qubits})  [offset]

        Args:
            alpha: hardware fidelity factor alpha = (1-lambda_V)(1-lambda_U).
            l1_fhat: L^1[f-hat].
            n_accuracy_blocks: n (number of accuracy blocks).
            f_l2_norm: ||f||_{L^2(mu)}, the L^2 norm of the target function under
                the evaluation measure mu.
            R: output scaling factor.
            n_qubits: number of qubits.

        Returns:
            The Theorem 3.17 three-term error bound.
        """
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0,1], got {alpha}"
        statistical = alpha * l1_fhat / np.sqrt(n_accuracy_blocks)
        systematic = (1.0 - alpha) * f_l2_norm
        offset = R * (1.0 - alpha) * (1.0 - (4.0 * n_accuracy_blocks) / (2 ** n_qubits))
        return statistical + systematic + offset

    def total_bound_with_readout(
        self,
        alpha: float,
        l1_fhat: float,
        n_accuracy_blocks: int,
        f_l2_norm: float,
        R: float,
        n_qubits: int,
        readout_p: float,
    ) -> float:
        """Total error bound including classical readout error (Proposition 3.20, Eq. 4.4).

        epsilon_total = [Theorem 3.17 three-term bound] + 4*R*readout_p

        This is the exact quantity validated against IBM ibm_fez hardware in
        Section 4.5 (reported total bound $18.578 there, with empirical MAE
        $2.345, within bound on 10/10 test points).

        NOTE: `readout_p` is an ASSUMED value for ibm_fez (SIR
        implementation_assumptions[5], confidence 0.35) -- not listed in the
        paper's own Appendix A hardware table. See configs/config.yaml.

        Args:
            alpha: hardware fidelity factor.
            l1_fhat: L^1[f-hat].
            n_accuracy_blocks: n (number of accuracy blocks).
            f_l2_norm: ||f||_{L^2(mu)}.
            R: output scaling factor.
            n_qubits: number of qubits.
            readout_p: single-qubit readout (bit-flip) error probability.

        Returns:
            epsilon_total, the full hardware-validated error bound.
        """
        base_bound = self.depolarising_bound(
            alpha, l1_fhat, n_accuracy_blocks, f_l2_norm, R, n_qubits
        )
        readout_term = 4.0 * R * readout_p
        return base_bound + readout_term

    def decompose_total_bound(
        self,
        alpha: float,
        l1_fhat: float,
        n_accuracy_blocks: int,
        f_l2_norm: float,
        R: float,
        n_qubits: int,
        readout_p: float,
    ) -> dict[str, float]:
        """Return the total bound broken down into its four named components.

        Mirrors Figure 4.2 panel (e), "decomposition of epsilon_total into
        approximation, systematic, bias, and readout contributions".

        Returns:
            dict with keys 'statistical', 'systematic', 'offset', 'readout', 'total'.
        """
        statistical = alpha * l1_fhat / np.sqrt(n_accuracy_blocks)
        systematic = (1.0 - alpha) * f_l2_norm
        offset = R * (1.0 - alpha) * (1.0 - (4.0 * n_accuracy_blocks) / (2 ** n_qubits))
        readout = 4.0 * R * readout_p
        total = statistical + systematic + offset + readout
        return {
            "statistical": statistical,
            "systematic": systematic,
            "offset": offset,
            "readout": readout,
            "total": total,
        }
