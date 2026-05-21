# Domain: Physics — Evaluation Standards (Stage 6 Enrichment)

Load alongside `agents/06_results_comparator.md` when domain is **Physics**.

---

## Deviation thresholds for physics metrics

| Metric | Excellent | Good | Moderate | Significant | Notes |
|--------|-----------|------|----------|-------------|-------|
| Relative L2 error | ≤ 5% | ≤ 15% | ≤ 30% | > 30% | Relative to reference |
| Energy drift | ≤ 1% | ≤ 3% | ≤ 8% | > 8% | Over full simulation |
| Convergence rate | ≤ 0.1 order | ≤ 0.2 order | ≤ 0.5 order | > 0.5 order | In log-log slope |
| Force MAE (MD) | ≤ 2% | ≤ 5% | ≤ 12% | > 12% | Relative |
| Wall-clock speedup | ≤ 10% | ≤ 25% | ≤ 50% | > 50% | Relative to paper |

**Physics errors are often regime-dependent.** A method that achieves low error in smooth
regimes may fail at shocks or boundary layers. Always note which regime the comparison is in.

---

## Root causes specific to physics

1. **Resolution mismatch** — PDE solvers are strongly resolution-dependent. If the user
   ran at a different spatial or temporal resolution than the paper, results will differ
   systematically. Check `spatial_resolution` and `timestep` against the paper's config.

2. **Reference solution quality** — papers compare against a "ground truth" that is itself
   a numerical solution at high resolution. If the reference was computed differently,
   relative errors are not comparable.

3. **Float32 vs float64** — long simulations accumulate numerical errors. If the paper used
   float64 and the implementation uses float32, energy drift and long-time accuracy will differ.

4. **Boundary condition implementation** — periodic, Dirichlet, Neumann, and mixed boundary
   conditions must be implemented exactly. A wrong BC type changes the solution fundamentally.

5. **Symmetry enforcement** — equivariant methods that enforce physical symmetries exactly
   (E(3), SE(3)) produce different results from methods that only approximately satisfy them.
   If the SIR flagged symmetry enforcement as assumed, check this first.

6. **Random seed for stochastic methods** — Monte Carlo and stochastic MD results require
   averaging over multiple runs. A single run comparison against a paper's averaged result
   will show apparent deviation even when the implementation is correct.
