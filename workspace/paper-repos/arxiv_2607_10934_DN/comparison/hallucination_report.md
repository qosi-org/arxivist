# Hallucination Report — arxiv_2607_10934

**Comparison date**: 2026-07-15T18:38:54Z

This report reviews the generated codebase (`paper-repos/arxiv_2607_10934/`) against
`sir-registry/arxiv_2607_10934/sir.json` for three classes of deviation.

## Structural hallucinations (components in code but NOT in the SIR)

None found. Every class in `architecture_plan.json -> module_hierarchy` maps to an SIR
`architecture.modules[]` entry, and no additional modules were introduced during code
generation.

## Parametric hallucinations (assumed hyperparameters that may be wrong)

### `n_steps (time-discretization resolution)`
- **Assumed value**: 1000
- **Severity**: Minor
- **Evidence**: Not specified anywhere in the paper (continuous-time throughout); terminal_price_error empirically shrinks as n_steps grows (0.13 at 100 steps -> 0.0026 at 20000 steps), consistent with an Euler-Maruyama discretization artifact rather than an implementation bug.
- **Suggested fix**: Increase --config training.n_steps for tighter terminal-price agreement if higher precision is needed; does not affect any of the four closed-form identity checks, which use independent fixed quadrature grids.

### `sigma(t) functional form for back_pedersen1998, m(t)/nu for cdf2016, C/sigma eigenvalues for common_eigenbasis`
- **Assumed value**: illustrative synthetic choices (sin-wave sigma(t); constant m=0.1, nu=0.2; random SPD C)
- **Severity**: Minor
- **Evidence**: The paper states these cases hold for ARBITRARY deterministic sigma(t) / m(t) / shared-eigenbasis (C, sigma); specific numeric choices are needed only to run a concrete demo and do not correspond to any paper-reported configuration (none exists, since this is a theory paper).
- **Suggested fix**: None needed for the paper's own claims (which are about arbitrary members of these classes); swap in any other valid (C, sigma) if project-specific numbers are needed.

## Omission hallucinations (SIR components missing or stubbed in code)

### Section 5.5 general scalar stochastic-liquidity BSDE solver
- **SIR location**: SIR mathematical_spec 'Stochastic maximum principle Hamiltonian (scalar stochastic liquidity)', implementation_assumptions[1]
- **Severity**: Significant
- **Suggested fix**: Implement a quadratic BSDE solver (deep BSDE / Han-Jentzen-E or LSMC) for eq. (5.44)-(5.45) if this case needs to be covered; currently only Sections 5.1, 5.2, 5.4, 5.6 are implemented.

### Section 5.7 general (non-commuting) matrix FBSDE
- **SIR location**: SIR architecture module GeneralMatrixFBSDE_OpenProblem, ambiguities[0]
- **Severity**: Minor
- **Suggested fix**: None available -- this is an open problem in the source paper itself, not an omission introduced by this pipeline. fbsde_stub.py documents this and raises NotImplementedError by design.

## Overall assessment

No Critical or Significant *structural* hallucinations were found. The two Minor
parametric items are illustrative-only numeric choices needed to run a concrete demo of
formulas the paper states hold for arbitrary members of a class (arbitrary deterministic
sigma(t), arbitrary shared-eigenbasis (C, sigma)) -- they do not misrepresent anything
the paper claims. The one Significant omission (Section 5.5's BSDE solver) is a scope
gap, not a fabrication: it is documented in the README and architecture plan risk
assessment rather than silently skipped. The one Minor omission (Section 5.7) is not a
gap introduced by this pipeline at all -- it is an open problem in the source paper.
