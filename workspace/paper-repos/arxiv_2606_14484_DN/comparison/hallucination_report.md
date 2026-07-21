# Hallucination Report
**Paper ID**: arxiv_2606_14484
**SIR version used**: 1 (overall confidence 0.64)
**Architecture plan version used**: 1

This paper has no trained model — it is a suite of Monte-Carlo simulations, calibrated closed-form
formulas, and multi-source on-chain reconciliations. There is no weight/checkpoint hallucination
risk in the usual sense. The relevant risk category here is **modeling-assumption hallucination**:
places where the paper describes a result narratively but not formulaically, and the code generator
had to invent a concrete method to fill the gap. All four instances found were already flagged by
the architecture plan's own risk assessment (not newly discovered here), and none rises to the
level of a structural or omission hallucination.

## Structural hallucinations
None found. The module hierarchy in `architecture_plan.json` (timeline forecast, Bitcoin/Ethereum
exposure models, mining-competitiveness model, mempool-race model, migration-race model,
top-20 readiness survey) maps one-to-one onto the SIR's `architecture` section. No extra components
were invented that aren't grounded in the SIR.

## Parametric hallucinations
These are hyperparameters/methods marked `# ASSUMED` in `config.yaml` where the paper gives a
narrative description but not an exact formula. Listed in descending order of impact on this
comparison run:

1. **Monte-Carlo distributional forms + mixture method** (severity: Significant)
   - `monte_carlo_distribution_form: uniform`, `survey_estimator_distribution_form: lognormal`,
     sample-level (not density-level) mixture — SIR confidence 0.4–0.45.
   - Evidence this matters: this is the direct, documented cause of the -26.95% deviation on
     P(CRQC by 2035), the one Significant metric miss in this comparison.
   - Suggested fix: expose `sensitivity_sweep()` output (already implemented) alongside the point
     estimate so users see the 8–24% band the paper itself reports, instead of one number that can
     land outside it.

2. **Mempool-race propagation-delay parameter** (severity: Minor)
   - `realistic_propagation_delay_minutes: 2.0` — back-calculated to hit the paper's stated ~30%,
     not independently derived (SIR confidence 0.4).
   - Evidence: your realistic success probability (28.7%) is within Good range (-4.33%) of the
     paper's 30%, so this assumption is not currently causing a problem, but it is worth flagging
     since the formula itself is invented, not sourced.

3. **Exposure-reconciliation method (Bitcoin `reconcile_sources`, Ethereum `bottom_up_estimate`)** (severity: Minor)
   - `reconciliation_method: simple_mean`; Ethereum `contract_fraction: 0.08` and
     `beacon_overcount_correction_factor: 2.3` — back-solved to land inside the paper's stated
     ranges rather than derived from the raw source measurements (SIR confidence 0.5).
   - Evidence: three of your four Moderate-severity deviations (P(CRQC by 2050) aside) — Bitcoin
     exposed-at-rest (+5.67%), Ethereum at-rest exposure (-5.83%) — sit in the range this assumption
     would be expected to produce.

4. **Physics-estimator calibration constants (Q0, R0, t0)** (severity: Minor)
   - Back-solved from the paper's 2026 boundary conditions rather than given directly
     (SIR confidence 0.45).
   - Evidence: contributes to, but is not solely responsible for, the P(CRQC by 2050) +5.33%
     deviation; effect is secondary to item 1 above.

## Omission hallucinations
None found. Cross-checking the SIR's `architecture.modules` list against `architecture_plan.json`'s
`module_hierarchy`, all modules are present and none are stubbed. The Table 3 readiness CSV
(`data/table3_readiness_ratings.csv`) was checked against the paper's Table 3 and matches on all
16 transcribed rows and ratings, so no transcription omission either.

## Overall Assessment

No structural fabrication or missing components. The one thing worth a user's attention is that the
pipeline's single Significant metric deviation (P(CRQC by 2035)) traces directly to a documented,
high-severity modeling assumption the paper leaves unspecified — this is an honest, flagged gap in
the reproduction, not a hidden error.
