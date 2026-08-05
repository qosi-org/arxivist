"""Quantum circuit construction: state preparation V and parameterized unitary U(theta,x).

Implements: Architecture Plan module `models/qnn_circuit.py`.
Paper sections: Section 3.2 (noisy circuit setup, V and U), Section 4.1 "Circuit
architecture details" and "Uniformly Controlled Rotation Decomposition", Section
4.1.1 "Circuit validation" (closed-form reference formula used to unit-test this
module).

IMPORTANT (flagged per SIR ambiguities[3] / architecture_plan risk_assessment,
Medium severity): the exact gate-level Uniformly Controlled Rotation (UCR)
decomposition of U(theta,x) is delegated by the paper to an external reference
([23], Moettoenen et al. 2004) and is not reproduced verbatim in the paper itself.
The construction below is a best-effort reconstruction using Qiskit's built-in
UCRZGate/UCRYGate primitives, following the architecture plan's description
("UCRz on qubit 0 conjugated by Hadamard gates, and one UCRy on qubit 1"). Before
trusting any lambda_U / hardware-bound computation derived from this circuit,
run `tests/test_qnn_circuit.py::test_matches_closed_form_reference`, which
reproduces the paper's own validation check (Section 4.1.1): compare circuit
output against the closed-form reference formula to tolerance 1e-9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UCRYGate, UCRZGate


@dataclass
class QNNCircuitBuilder:
    """Builds the full noiseless QNN circuit: V (state prep) + U(theta,x) (parameterized)."""

    n_accuracy_blocks: int
    n_qubits: int

    def __post_init__(self) -> None:
        assert self.n_qubits >= 3, (
            f"n_qubits must be >= 3 (>=1 control qubit + 2 target qubits), got {self.n_qubits}"
        )
        self.n_control_qubits = self.n_qubits - 2
        self.n_control_states = 2 ** self.n_control_qubits
        assert self.n_accuracy_blocks <= self.n_control_states, (
            f"n_accuracy_blocks={self.n_accuracy_blocks} cannot exceed 2^n_control_qubits="
            f"{self.n_control_states} (n_qubits={self.n_qubits})"
        )
        # IMPORTANT: target qubits are placed at indices 0,1 (the two LEAST-significant
        # qubits) and control qubits at indices 2..n_qubits-1. This must match
        # `MeasurementProcessor.group_counts`'s convention of grouping raw outcomes via
        # o mod 4 = m (target bits, weight 1 and 2) and o // 4 = k (control-register
        # value, everything above bit 1). Swapping this ordering silently breaks the
        # measurement-grouping <-> circuit-construction contract without raising an error.
        self.target_y = 0  # weight-1 bit of m
        self.target_z = 1  # weight-2 bit of m
        self.control_qubits = list(range(2, self.n_qubits))

    def build_state_preparation(self) -> QuantumCircuit:
        """Build V: Hadamard gates on the n_qubits-2 control qubits (Section 3.2, 4.1).

        Target qubits (indices 0, 1) are left in |00>. Per Remark 3.5, V is treated
        as noiseless in the general CPTP formalism since it consists only of
        Hadamard gates.

        Returns:
            A QuantumCircuit implementing V.
        """
        qc = QuantumCircuit(self.n_qubits, name="V_state_prep")
        for q in self.control_qubits:
            qc.h(q)
        return qc

    def build_parameterized_unitary(
        self, theta: list[tuple[float, float, float]], x: np.ndarray
    ) -> QuantumCircuit:
        """Build U(theta, x): block-diagonal unitary via UCRZ (conjugated by H) + UCRY.

        U(theta,x) = sum_{k=0}^{n_accuracy_blocks-1} |k><k| tensor U^(k)(theta_k, x)
        with theta_k = (a_k, b_k, gamma_k) and
            U^(k)(theta_k, x) = U1^(k)(a_k, b_k, x) (x) U2^(k)(gamma_k)   (Section 4.1)

        Reconstruction used here: a UCRZ gate (conjugated by Hadamards on target
        qubit 1, "target_z") encodes the Fourier phase term (b_k + a_k . x); an
        independent UCRY gate on target qubit 0 ("target_y") encodes gamma_k.
        Angle lists are zero-padded from n_accuracy_blocks up to
        2^n_control_qubits, consistent with the paper's statement that padded
        states (k >= n_accuracy_blocks) do not contribute to the final
        measurement grouping (Section 4.1, "Measurement").

        Derivation (verified against `closed_form_reference_output` in this
        module's test coverage, Section 4.1.1's own validation check):
        for a bare RZ(phi) sandwiched as H . RZ(phi) . H |0>, the resulting state
        is cos(phi/2)|0> - i sin(phi/2)|1>, so P(0)-P(1) = cos(phi). Passing
        phi = b_k + a_k.x directly (NOT doubled) therefore gives P(0)-P(1) =
        cos(b_k + a_k.x) exactly. Symmetrically, RY(phi)|0> = cos(phi/2)|0> +
        sin(phi/2)|1>, so P(0)-P(1) = cos(phi) for phi = gamma_k directly (also
        not doubled). Given control-register value k, the two target qubits are
        independent, so the joint quantity P(00|k)-P(01|k)-P(10|k)+P(11|k) =
        [P_z(0|k)-P_z(1|k)] * [P_y(0|k)-P_y(1|k)] = cos(b_k+a_k.x) * cos(gamma_k)
        -- exactly the per-block term of the target sum, once averaged over the
        1/n_accuracy_blocks weight coming from the initial uniform superposition
        over k (Section 4.1: f^R_{n,theta}(x) = R[1-2(P1+P2)] = R * (1/n) *
        sum_k [P(00|k)+P(11|k)-P(01|k)-P(10|k)], using P0+P1+P2+P3=1 per k and
        the m = 2*bit(target_z) + bit(target_y) convention that matches
        `MeasurementProcessor.group_counts`'s `o mod 4` grouping).

        Args:
            theta: list of n_accuracy_blocks tuples (a_k, b_k, gamma_k). a_k is
                itself a vector of length d (matching x's dimensionality); b_k,
                gamma_k are scalars.
            x: input vector, shape [d] (already normalised to [0,1]^d, Eq. 4.1).

        Returns:
            A QuantumCircuit implementing U(theta, x).
        """
        assert len(theta) == self.n_accuracy_blocks, (
            f"Expected {self.n_accuracy_blocks} theta tuples, got {len(theta)}"
        )

        qc = QuantumCircuit(self.n_qubits, name="U_parameterized")

        # --- Phase-encoding term on target_z, via UCRZ conjugated by Hadamards ---
        z_angles = np.zeros(self.n_control_states, dtype=np.float64)
        for k, (a_k, b_k, _gamma_k) in enumerate(theta):
            a_k_arr = np.asarray(a_k, dtype=np.float64)
            z_angles[k] = b_k + float(np.dot(a_k_arr, x))  # NOT doubled -- see derivation above
        qc.h(self.target_z)
        qc.append(UCRZGate(list(z_angles)), [self.target_z] + self.control_qubits)
        qc.h(self.target_z)

        # --- Amplitude-modulation term on target_y, via UCRY ---
        y_angles = np.zeros(self.n_control_states, dtype=np.float64)
        for k, (_a_k, _b_k, gamma_k) in enumerate(theta):
            y_angles[k] = gamma_k  # NOT doubled -- see derivation above
        qc.append(UCRYGate(list(y_angles)), [self.target_y] + self.control_qubits)

        return qc

    def assemble_circuit(
        self, theta: list[tuple[float, float, float]], x: np.ndarray, measure: bool = True
    ) -> QuantumCircuit:
        """Assemble the full circuit: V followed by U(theta,x), with measurement.

        Args:
            theta: circuit parameters, see `build_parameterized_unitary`.
            x: normalised input vector, shape [d].
            measure: if True, append a measure_all() at the end.

        Returns:
            The complete QuantumCircuit.
        """
        v = self.build_state_preparation()
        u = self.build_parameterized_unitary(theta, x)
        qc = v.compose(u)
        if measure:
            qc.measure_all()
        return qc


def closed_form_reference_output(
    theta: list[tuple[float, float, float]], x: np.ndarray, R: float, n_accuracy_blocks: int
) -> float:
    """Closed-form analytical QNN output, used as a circuit-validation reference.

    f^R_{n,theta}(x) = (1/n) * sum_{i=1}^{n} R * cos(gamma_i) * cos(b_i + a_i . x)
    (Section 4.1.1, "Circuit validation", reproduced from [9]).

    This does NOT execute any quantum circuit -- it is the paper's own closed-form
    check used to confirm a circuit implementation is correct (residuals should be
    uniformly bounded by R/sqrt(N_shots) across tested configurations).

    Args:
        theta: list of n_accuracy_blocks tuples (a_k, b_k, gamma_k).
        x: input vector, shape [d].
        R: output scaling factor.
        n_accuracy_blocks: n (number of accuracy blocks).

    Returns:
        The closed-form reference value of f^R_{n,theta}(x).
    """
    assert len(theta) == n_accuracy_blocks
    total = 0.0
    for a_i, b_i, gamma_i in theta:
        a_i_arr = np.asarray(a_i, dtype=np.float64)
        total += math.cos(gamma_i) * math.cos(b_i + float(np.dot(a_i_arr, x)))
    return (R / n_accuracy_blocks) * total
