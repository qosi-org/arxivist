"""Measurement post-processing: raw circuit counts -> grouped probabilities -> scalar output.

Implements: Architecture Plan module `models/measurement.py`.
Paper sections: Section 3.3 (noisy probabilities / projector definition), Eq. (2.2)
(QNN scalar output), Section 4.1 "Measurement" (outcome grouping rule), Eq. (4.2)
(shot-noise statistical error).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeasurementProcessor:
    """Groups raw computational-basis measurement counts into P0..P3 and computes
    the scalar QNN output f^R_{n,theta}(x).
    """

    def group_counts(
        self, counts: dict[str, int], n_accuracy_blocks: int, n_qubits: int
    ) -> np.ndarray:
        """Group raw bitstring counts into the 4 aggregated probabilities P0..P3.

        Per Section 4.1 "Measurement": for each outcome o in {0, ..., 2^n_qubits - 1},
        write o = 4*k + m with k in {0,...,n_accuracy_blocks-1}, m in {0,1,2,3}.
        Only outcomes with k < n_accuracy_blocks contribute (states with
        k >= n_accuracy_blocks are padding artefacts from rounding up to a power of two).

        P_m = (1 / N_shots) * sum_{o: o mod 4 == m, o < 4*n_accuracy_blocks} count(o)

        Args:
            counts: dict mapping bitstrings (as produced by Qiskit, e.g. '01011') to
                integer shot counts.
            n_accuracy_blocks: n (number of accuracy blocks).
            n_qubits: total number of qubits in the circuit.

        Returns:
            probs, shape [4], the four grouped probabilities P0, P1, P2, P3.
        """
        total_shots = sum(counts.values())
        assert total_shots > 0, "counts dict has zero total shots"

        raw = np.zeros(2 ** n_qubits, dtype=np.float64)
        for bitstring, count in counts.items():
            # Qiskit bitstrings are big-endian by qubit index; convert to integer outcome.
            outcome = int(bitstring.replace(" ", ""), 2)
            if outcome < len(raw):
                raw[outcome] = count

        probs = np.zeros(4, dtype=np.float64)
        max_valid_outcome = 4 * n_accuracy_blocks
        for outcome in range(min(max_valid_outcome, len(raw))):
            m = outcome % 4
            probs[m] += raw[outcome]

        return probs / total_shots

    def qnn_output(self, probs: np.ndarray, R: float) -> float:
        """Compute the scalar QNN output f^R_{n,theta}(x) = R[1 - 2(P1 + P2)]  (Eq. 2.2).

        Args:
            probs: array of shape [4], the grouped probabilities P0..P3.
            R: output scaling factor (Section 4.1: R = ceil(1.1 * max_i P_i)).

        Returns:
            The scalar QNN output.
        """
        assert probs.shape == (4,), f"Expected shape [4], got {probs.shape}"
        p1, p2 = probs[1], probs[2]
        return R * (1.0 - 2.0 * (p1 + p2))

    def shot_noise_stderr(self, probs: np.ndarray, R: float, n_shots: int) -> float:
        """Statistical error on the QNN output price from finite shot count (Eq. 4.2).

        stderr = 2R * sqrt( (P1+P2)(1-P1-P2) / N_shots )

        Args:
            probs: array of shape [4], grouped probabilities P0..P3.
            R: output scaling factor.
            n_shots: number of measurement shots used.

        Returns:
            The estimated standard error of the QNN output.
        """
        assert probs.shape == (4,), f"Expected shape [4], got {probs.shape}"
        assert n_shots > 0, "n_shots must be positive"
        p_sum = probs[1] + probs[2]
        variance_term = max(p_sum * (1.0 - p_sum), 0.0)  # clamp for float safety
        return 2.0 * R * np.sqrt(variance_term / n_shots)
