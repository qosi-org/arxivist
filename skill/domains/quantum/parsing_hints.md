# Domain: Quantum — Parsing Hints (Stage 1 Enrichment)

Load alongside `agents/01_paper_parser.md` when the detected domain is **Quantum**.

---

## Architecture extraction — Quantum-specific rules

**Circuit notation:** the architecture is a quantum circuit. Extract:
- Number of qubits (n)
- Circuit depth (total gate layers)
- Gate set used (standard: H, CNOT, RZ, RX, RY; note if custom gates are used)
- Whether the circuit is parameterised (variational) or fixed
- Entanglement structure: linear, all-to-all, hardware-efficient, brickwork

**Variational circuits (VQC/VQE/QAOA):** extract:
- Ansatz family (hardware-efficient, UCCSD, QAOA alternating operators)
- Number of variational layers (p for QAOA, L for VQC)
- Parameter initialisation strategy (random, FOURIER, identity block)
- Classical optimiser for variational parameters

**Hybrid quantum-classical:** identify clearly which parts run on quantum hardware vs
classical hardware. The interface (measurement basis, shot count, parameter shift rule)
is the critical implementation detail.

**"Depth" in quantum means circuit depth**, NOT neural network depth. Never conflate these.
Shallow circuits (depth < 20) can run on NISQ hardware; deep circuits require fault tolerance.

---

## Mathematical spec — Quantum-specific rules

Always extract:
- Hamiltonian H explicitly (for VQE/QAOA papers)
- Cost function / objective in terms of expectation values ⟨ψ|O|ψ⟩
- Parameter shift rule for gradient computation (if used)
- Noise model if the paper includes noise simulation
- Fidelity metric definition used in evaluation

---

## Training pipeline — Quantum-specific rules

**Classical optimisation loop:** extract:
- Optimiser (ADAM, COBYLA, SPSA, L-BFGS-B — quantum papers use different optimisers)
- Number of shots per circuit evaluation
- Number of optimisation steps / function evaluations
- Whether gradients are computed analytically (parameter shift) or numerically (SPSA)

**Hardware vs simulation:** note explicitly whether results are from:
- Classical statevector simulation (exact)
- Classical shot-based simulation (noisy sampling)
- Real quantum hardware (specify device: IBM, IonQ, Quantinuum, etc.)
- Noisy simulation with a hardware noise model

---

## Evaluation — Quantum-specific rules

Extract:
- Fidelity (state fidelity, gate fidelity)
- Energy error / approximation ratio (for VQE/QAOA)
- Circuit expressibility and entanglement capability if reported
- Classical simulation baseline comparison
- Hardware vs simulation result comparison if both are given
