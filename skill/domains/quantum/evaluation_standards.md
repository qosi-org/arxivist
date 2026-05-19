# Domain: Quantum — Evaluation Standards (Stage 6 Enrichment)

Load alongside `agents/06_results_comparator.md` when domain is **Quantum**.

---

## Deviation thresholds for quantum metrics

| Metric | Excellent | Good | Moderate | Notes |
|--------|-----------|------|----------|-------|
| State fidelity | ≤ 0.005 | ≤ 0.02 | ≤ 0.05 | Absolute (0–1 scale) |
| Energy error | ≤ 0.5% | ≤ 2% | ≤ 5% | Relative to exact ground state |
| Approximation ratio (QAOA) | ≤ 0.01 | ≤ 0.03 | ≤ 0.08 | Absolute (0–1 scale) |
| Circuit expressibility | ≤ 5% | ≤ 15% | ≤ 30% | Relative |

**Shot-based results have inherent statistical variance.** For results with < 8192 shots,
a deviation within 2 standard deviations of the shot noise estimate is Excellent regardless
of numerical difference.

---

## Root causes specific to quantum

1. **Simulation mode mismatch** — statevector (exact) vs shot-based vs noisy hardware
   produce fundamentally different results. Always check which mode the paper used.

2. **Noise model version** — hardware noise models from IBM/Google/IonQ change as devices
   are recalibrated. Noise models from different dates produce different results.

3. **Optimiser convergence sensitivity** — variational algorithms are sensitive to
   initialisation. Multiple random seeds must be run and aggregated.

4. **Barren plateaus** — gradients vanish exponentially with qubit count for random
   initialisations. If training doesn't converge, this is the likely cause for n > 10 qubits.

5. **Framework version** — PennyLane and Qiskit change circuit compilation and gradient
   rules across versions. Pin exact versions.
