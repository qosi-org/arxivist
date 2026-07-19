"""
CVA oracle assembly: full encoding unitary A_Theta, Grover iterate Q, and the
marked-subspace projector Pi_111.

Implements Section 2.2.1 (Eq. 33-37), Figure 1, and Appendix B (Eq. 71-72) of
arXiv:2607.12990.

    A = R_q R_p R_v G_P                       (composition order, Section 2.2.1)
    a_CVA = <xi| Pi_111 |xi>,  |xi> = A|0>       (Eq. 35-36)
    Q = -A S0 A^dagger Sf                       (Grover iterate, Appendix B)
    Q^k A|0> = cos(K*theta)|Psi0> + sin(K*theta)|Psi1>,  K = 2k+1   (Eq. 72)

SIR reference: architecture.modules "A_Theta", "Pi_111", "Q (Grover iterate)".
"""

from __future__ import annotations


from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, ZGate
from qiskit.quantum_info import Statevector


class CVAOracle:
    """Assembles the trained circuit blocks into the full CVA encoding unitary
    and constructs the amplified (Grover-iterated) circuits used by CABIQAE.

    Args:
        num_register_qubits: m + n, the size of the time-market register.
        num_ancillas: number of ancilla qubits (3: a_v, a_p, a_q for the
            full CVA oracle; for the reduced validation circuit A_test this
            module can equally be used with num_ancillas=0 and a
            user-supplied marked-bitstring convention).
    """

    def __init__(self, num_register_qubits: int, num_ancillas: int = 3) -> None:
        self.num_register_qubits = num_register_qubits
        self.num_ancillas = num_ancillas
        self.num_qubits = num_register_qubits + num_ancillas

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CVAOracle(register_qubits={self.num_register_qubits}, "
            f"ancillas={self.num_ancillas})"
        )

    def assemble(
        self,
        g_theta: QuantumCircuit,
        r_v: QuantumCircuit,
        r_p: QuantumCircuit,
        r_q: QuantumCircuit,
    ) -> QuantumCircuit:
        """Compose A_Theta = R_q(phi_q*) R_p(phi_p*) R_v(phi_v*) G_theta*.

        Args:
            g_theta: trained QCBM circuit on `num_register_qubits` qubits
                (parameters already bound).
            r_v: trained exposure CRCA circuit on
                `num_register_qubits + 1` qubits (parameters already bound);
                its last qubit is the a_v ancilla.
            r_p: trained discount CRCA circuit on `num_time_qubits + 1`
                qubits (parameters already bound); its last qubit is a_p.
            r_q: trained default-probability CRCA circuit on
                `num_time_qubits + 1` qubits (parameters already bound);
                its last qubit is a_q.

        Returns:
            The full 9-qubit (for the paper's instance) CVA encoding circuit
            A_Theta, with ancilla qubits ordered [a_v, a_p, a_q] appended
            after the time-market register.
        """
        qc = QuantumCircuit(self.num_qubits, name="A_Theta")
        register_qubits = list(range(self.num_register_qubits))
        a_v, a_p, a_q = (
            self.num_register_qubits,
            self.num_register_qubits + 1,
            self.num_register_qubits + 2,
        )

        qc.compose(g_theta, qubits=register_qubits, inplace=True)
        qc.compose(r_v, qubits=register_qubits + [a_v], inplace=True)

        # R_p and R_q act only on the time-register subset of `register_qubits`.
        num_time_qubits = r_p.num_qubits - 1
        time_qubits = register_qubits[:num_time_qubits]
        qc.compose(r_p, qubits=time_qubits + [a_p], inplace=True)
        qc.compose(r_q, qubits=time_qubits + [a_q], inplace=True)
        return qc

    def marked_amplitude_statevector(self, circuit: QuantumCircuit) -> float:
        """Compute a_CVA = <xi|Pi_111|xi> exactly via statevector simulation.

        Pi_111 projects onto all three ancillas (a_v, a_p, a_q) being in
        state |1>; the marked subspace is spanned by all |i>|j>|111>, not a
        single basis state of the full register (Eq. 35-36).

        Args:
            circuit: the (fully bound, no free parameters) A_Theta circuit.

        Returns:
            a_CVA in [0, 1].
        """
        state = Statevector.from_instruction(circuit)
        probs = state.probabilities_dict()
        # Qiskit bitstrings are little-endian left-to-right over all qubits;
        # the three ancillas are the *last* qubit indices, i.e. the leftmost
        # three characters of the returned bitstring.
        marked_mass = sum(
            p for bitstr, p in probs.items() if bitstr[: self.num_ancillas] == "1" * self.num_ancillas
        )
        return float(marked_mass)

    def _build_sf_phase_oracle(self) -> QuantumCircuit:
        """Build Sf = I - 2*Pi_good as a phase-flip circuit on the ancillas.

        Pi_good marks the subspace where all `num_ancillas` ancilla qubits
        are |1> (Eq. 35: Pi_111). This is implemented as a multi-controlled-Z
        (phase flip, no bitflip) acting on the ancilla qubits only, which is
        the "oracle" input expected by Qiskit's GroverOperator (a reflection
        about the marked/bad state, not a bit-flip marking oracle).
        """
        sf = QuantumCircuit(self.num_qubits, name="Sf")
        ancilla_qubits = list(
            range(self.num_register_qubits, self.num_register_qubits + self.num_ancillas)
        )
        if self.num_ancillas == 1:
            sf.z(ancilla_qubits[0])
        else:
            mcz = ZGate().control(self.num_ancillas - 1)
            sf.append(mcz, ancilla_qubits)
        return sf

    def grover_iterate(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Build Q = -A S0 A^dagger Sf (Appendix B).

        Constructs Sf directly as a phase oracle marking the all-ones
        ancilla subspace (Pi_111, Eq. 35), then hands it to Qiskit's
        GroverOperator together with the state-preparation circuit A. Note:
        Qiskit's GroverOperator `oracle` argument expects a *phase* oracle
        (a reflection about the marked state), not a bit-flip "good_state"
        marking -- hence Sf is built manually here rather than via the
        `good_state=` convenience kwarg (which does not exist in this
        qiskit version's GroverOperator API).

        Args:
            circuit: the A_Theta circuit (state-preparation oracle "A").

        Returns:
            A QuantumCircuit implementing one application of Q.
        """
        sf = self._build_sf_phase_oracle()
        grover_op = GroverOperator(oracle=sf, state_preparation=circuit)
        return grover_op

    def amplified_circuit(self, circuit: QuantumCircuit, k: int) -> QuantumCircuit:
        """Build Q^k A|0> for Grover power k (K = 2k+1 oracle queries, Eq. 72).

        Args:
            circuit: the A_Theta circuit.
            k: Grover power (k=0 returns the unamplified circuit, i.e. direct
                circuit sampling).

        Returns:
            QuantumCircuit implementing Q^k A applied to |0>.
        """
        full = QuantumCircuit(self.num_qubits)
        full.compose(circuit, inplace=True)
        if k > 0:
            q_op = self.grover_iterate(circuit)
            for _ in range(k):
                full.compose(q_op, inplace=True)
        return full

    def measure_ancillas(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Append classical measurement of the ancilla register only.

        Args:
            circuit: any circuit built on `self.num_qubits` qubits.

        Returns:
            Circuit with a ClassicalRegister of size `num_ancillas` measuring
            the top `num_ancillas` qubits (the ancillas).
        """
        measured = circuit.copy()
        measured.add_register(measured._create_creg(self.num_ancillas, "c"))
        ancilla_qubits = list(
            range(self.num_register_qubits, self.num_register_qubits + self.num_ancillas)
        )
        measured.measure(ancilla_qubits, list(range(self.num_ancillas)))
        return measured
