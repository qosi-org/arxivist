# Domain: Quantum — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

---

## Critical reproducibility traps in Quantum papers

**Hardware calibration drift** — real quantum hardware is recalibrated daily. Results on
IBM or IonQ devices from the same circuit on different days can vary by 5–15% in fidelity.
Always flag if paper uses real hardware without specifying the calibration date.

**Shot count underreporting** — papers sometimes report the number of shots per circuit
evaluation but not the total shots used (shots × repetitions × optimisation steps). The
total computational budget affects result quality significantly.

**Noise model versioning** — hardware noise models (gate error rates, T1/T2 coherence times)
are snapshots. Using a noise model from a different date than the paper's experiments
produces different results. Flag if noise model provenance is not stated.

**Random initialisation sensitivity** — variational algorithms started from different random
parameters converge to different local optima. Papers that report a single run without
multiple seeds are not reporting reproducible results. Always implement multi-seed runs.

**Problem instance selection** — for combinatorial optimisation papers (MaxCut, portfolio
optimisation), the specific problem instance (graph, constraint matrix) determines results.
Always extract the exact instance used or its generation procedure.

**Classical simulation limits** — exact statevector simulation of n qubits requires 2^n
complex amplitudes. Above n=25, exact simulation is infeasible on a single machine. If the
paper uses n > 25 qubits with exact simulation, flag as requiring HPC resources.
