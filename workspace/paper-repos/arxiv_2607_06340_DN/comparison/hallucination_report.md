# Hallucination Report

**Paper**: Signature-based identification of volatility models from path geometry
**Paper ID**: arxiv_2607_06340
**Report Date**: 2026-07-13

Cross-checks the generated repository against `sir.json` and `architecture_plan.json`,
and additionally documents the bugs found and fixed during Stage 4 testing
(prior to this comparison round) plus the one open, unresolved item.

**Summary: 0 structural hallucinations, 1 parametric issue (Medium, already
documented, not newly discovered), 0 omission hallucinations. 2 real code
bugs were found and fixed before this comparison; both are noted here for
the audit trail even though they predate this specific run.**

---

## Structural Hallucinations

**None found.** All 9 module files map cleanly to named SIR modules
(`HestonSimulator` -> `HestonPathSimulator`, `OUSimulator` ->
`OrnsteinUhlenbeckPathSimulator`, `RoughBergomiSimulator` ->
`RoughBergomiPathSimulator`, `SignatureComputer` ->
`TruncatedSignatureComputer`, `SignatureXGBClassifier` ->
`XGBoostClassifier`, `SignatureMLP` -> `NeuralNetworkBaseline`,
`ImportanceAnalyzer` -> `FeatureImportanceAnalyzer`). `ExperimentBuilder`
has no direct SIR module entry but is a straightforward orchestration layer
implementing the paper's 9 named experiments, not an invented component.

---

## Parametric Issues

### 1. Section 6.8 "comparable ranges" calibration — Medium, pre-existing, not new

- **Location**: `src/sig_vol_id/data/experiment_builder.py`, the
  `heston_ou_shared_dist` branch.
- **Evidence**: paper footnote 7 states Heston's `nu` and OU's `sigma` are
  drawn from "comparable ranges" without giving the exact calibration (SIR
  ambiguities, confidence 0.4 as tracked in `pipeline_state.json`).
  This implementation draws each independently from its own
  Feller-safe/config range for a given shared kappa/theta. This round's
  actual run (`6.8_low_nu`: 1.6% vs. paper's 69.8%) confirms this does not
  reproduce the paper's low-nu mechanism.
- **Classification rationale**: this is a parametric assumption the paper
  itself left underspecified, not an invented fact contradicting the paper
  -- hence Medium, not Critical, and explicitly not presented as fixed
  anywhere in the repo's docs.
- **Status**: open, flagged in README, notebook, and this comparison's
  root-cause analysis. Not touched in this report beyond documentation.

---

## Bugs Found and Fixed During Development (audit trail, pre-dates this comparison)

These are **not** hallucinations in the traditional sense -- they were
concrete code defects, found via failing unit tests and a notebook
execution producing a result in the wrong direction, and fixed before this
comparison round began. Logged here because they affect how much trust to
place in results from *before* the fix (none were shared with the user) vs.
*after* (all results in this comparison post-date both fixes).

### Bug 1: XGBClassifier predict() returned probabilities, not labels

- Passing `objective="multi:softprob"` and `num_class=n_classes` explicitly
  to `XGBClassifier` caused `.predict()` to return a `[n, n_classes]`
  probability array instead of `[n]` class labels, under xgboost 3.3.0.
- Found via a failing unit test (`test_fits_and_predicts_separable_classes`).
- Fixed by removing the explicit `objective`/`num_class` arguments and
  letting `XGBClassifier` auto-detect both from `y` at `fit()` time --
  verified to produce identical underlying boosting behavior with correct
  label output.

### Bug 2: 6.8 nu-override silently ignored when combined with shared-distribution

- The original `elif` chain in `experiment_builder.py` meant
  `heston_ou_shared_dist` and `nu_override`/`nu_fixed` were mutually
  exclusive branches -- so `6.8_low_nu`/`6.8_high_nu` (which need both)
  silently used only the shared-distribution branch, ignoring the nu
  override entirely.
- Found via notebook execution producing a result in the wrong direction.
- Fixed by restructuring the logic so nu overrides apply on top of
  whichever kappa/theta path was taken, matching the paper's "keeping all
  other parameters unchanged" description in Section 6.8.
- **Note**: fixing this bug did NOT fully resolve the low-nu direction
  (see Parametric Issue #1 above) -- the bug fix was necessary but not
  sufficient, since a separate, genuine calibration ambiguity remains.

---

## Omission Hallucinations

**None found.** All entrypoints (`train.py`, `evaluate.py`, `inference.py`,
`run_feature_importance.py`) are fully implemented, no stub methods, and
all 16 unit tests plus all 4 entrypoints were smoke-tested successfully
prior to release.

---

## Overall Assessment

The generated code is structurally complete and matches the SIR. The one
remaining gap (Section 6.8 low-nu calibration) is an honestly-flagged,
paper-level ambiguity rather than a code-generation defect, and this
round's actual run results (OU accuracy collapse, in particular) point to
sample size as the most likely explanation for the bulk of the remaining
deviation -- see `benchmark_comparison.md`'s root-cause analysis and
recommended actions for how to test that hypothesis directly.
