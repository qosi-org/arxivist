# Verification Log — arxiv_2607_10934

- **Comparison run timestamp**: 2026-07-15T18:38:54Z
- **ArXivist SIR version used**: 1
- **Architecture plan version used**: 1
- **Paper metrics identified (total claims in scope of Section 5 verification)**: 6
- **User/generated results matched**: 4
- **Metrics compared**:
  - terminal_covariance_identity (eq. 4.6) -- kyle1985
  - terminal_covariance_identity (eq. 4.6) -- back_pedersen1998
  - terminal_covariance_identity, Monte Carlo mean (eq. 4.6) -- collin_dufresne_fos2016
  - M*_0 agreement, eigenbasis-route vs direct-matrix closed form -- common_eigenbasis_bcel2020
- **User-reported config modifications**: none (default `configs/config.yaml`, seed=0, n_steps=1000)
- **Results file SHA256**: `8a0e41285b4b147cb79c573e94c0f988ac6f1eb801f8588287ac7e7d4b2c5cf3`
- **Manual review required**: True
- **Review reasons**:
  - overall_sir_confidence (0.58) is below the pipeline's 0.65 threshold
  - Section 5.5 and 5.7 paper claims are unmatched by design (out of scope / explicitly open problem), not because of an implementation failure
  - There is no independent human-run experiment to compare against -- these are self-verification checks of the generated code against the paper's own closed-form formulas, since the paper reports no numerical results table

## Notes on methodology deviation from the standard Stage 6 process

The standard Stage 6 process compares a human-run experiment's results against
`sir.json -> evaluation_protocol.reported_results`. That field is empty here because
the paper reports no numerical results table (it is a pure theory paper). This
verification run instead compares the repo's own simulated/computed outputs
(`run_verification.py`) against the paper's closed-form formulas directly, which is the
only meaningful notion of "reproducibility" available for this paper. This substitution
is documented in `architecture_plan.json -> risk_assessment` (Low severity item) and in
the SIR's `evaluation_protocol.special_conditions`.
