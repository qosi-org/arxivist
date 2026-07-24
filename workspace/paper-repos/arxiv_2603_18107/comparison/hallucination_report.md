# Hallucination Report — arxiv_2603_18107 (ARTEMIS)

**Paper**: ARTEMIS: A Neuro-Symbolic Framework for Economically Constrained Market Dynamics
**Analysis date**: 2026-07-19
**Basis**: comparison of `architecture_plan.json` and the generated `src/arxivist_artemis/`
code against `sir.json` (SIR version 1) and the paper text directly. This analysis does
**not** require user experimental results and was performed in full.

---

## Structural Hallucinations

Components/behaviors present in the generated code that deviate from, or are not grounded in,
the paper's stated methodology.

### H-1: Default config d_z/d_w mismatch silently zeroes the market-price-of-risk loss

- **Severity**: **Medium-High** (upgraded from Stage 5's initial "Medium" estimate after
  confirming the mechanism directly in code — see verification below)
- **Location**: `configs/config.yaml` (`model.artemis.d_z: 64`, `model.artemis.d_w: 16`);
  `src/arxivist_artemis/models/artemis.py::ARTEMIS.compute_losses` (line 159:
  `if sigma_coll.shape[-1] == sigma_coll.shape[-2]:`).
- **Evidence**: The market-price-of-risk loss (Eq. 8) is defined as an element-wise ratio
  `lambda(t) = sigma^-1 @ mu`, which is only mathematically well-defined when `sigma` is
  square (`d_z == d_w`). `compute_losses` correctly guards against the non-square case by
  falling back to `l_mpr = 0` rather than crashing — but this means that with the repository's
  **own shipped default config** (`d_z=64`, `d_w=16`), every training run silently computes
  `L_MPR = 0` for the entire run, with no warning printed anywhere.
- **Why this matters**: Table 3's own ablation (`A3_NoMPR`) shows that removing this exact
  loss term alone drops DSLOB directional accuracy from 64.89% to 56.82% — a real, paper-
  documented effect. Any user who trains ARTEMIS with the out-of-the-box config believing they
  are reproducing the paper's `A0_Full` configuration is, in fact, silently running something
  closer to `A3_NoMPR`, without any indication this happened. This was discovered during Stage
  5 notebook validation (Component 3's physics-losses demo raised an `AssertionError` on the
  same dimensional mismatch when called directly, rather than silently degrading) and is
  logged in `pipeline_state.json → stage5_validation.finding_carried_forward_for_stage6`.
- **Suggested fix**: Set `model.artemis.d_w` equal to `d_z` by default in `config.yaml` (or
  make `d_w` a derived/locked property equal to `d_z` rather than an independently-settable
  ASSUMED value), and/or have `compute_losses` emit an explicit warning (not just silently
  return 0) whenever `d_z != d_w` causes `L_MPR` to be skipped.
- **Not fixed in this report**: per this task's scope (Stage 6 only; do not modify Stages 1–5),
  this finding is documented here rather than patched in `config.yaml`/`artemis.py`.

### H-2: DSLOB is structurally unreproducible — its seed dataset is unnamed

- **Severity**: **High** (this is a paper-limitation finding, not an ArXivist implementation
  defect, but it is the single largest blocker to any real reproducibility assessment)
- **Location**: paper Section 3.4 ("The foundation of DSLOB is a real high-frequency limit
  order book dataset from which we extract 85 features...") — no name, citation, vendor, or
  access instructions are given anywhere in the paper for this source dataset.
- **Evidence**: `data/make_synthetic_seed_lob.py` was built specifically because no seed data
  could be identified; its own docstring and `data/README_data.md` both flag this explicitly.
- **Why this matters**: DSLOB underlies the paper's single headline result (64.96% DirAcc,
  the highest of all four datasets) and its entire 7-variant ablation study (Table 3) — the
  paper's strongest evidence for the value of the physics-informed losses. None of this can be
  exactly reproduced by anyone, including the original authors' collaborators, without access
  to information not present in the paper.
- **Not an ArXivist error**: this is a transparency gap in the source paper itself.

### H-3: OOM crash in evaluate.py / plot_figures.py (found and fixed in Stage 4)

- **Severity**: Low (already remediated; documented here for completeness of the audit trail)
- **Location**: `scripts/evaluate.py`, `scripts/plot_figures.py` (both carry `# BUGFIX` comments).
- **Evidence**: confirmed present via `grep -n "BUGFIX"` in both files.
- **Status**: Fixed during Stage 4 validation (batched forward passes instead of one
  full-dataset call). No outstanding action needed.

No other structural hallucinations were found. All 6 SIR architecture modules
(`LaplaceNeuralOperator`, `NeuralSDELatentDynamics`, `FeynmanKacPDEResidual`,
`MarketPriceOfRiskPenalty`, `SymbolicBottleneck`, `ConformalAllocationLayer`) map to
implemented classes with matching equations (verified against `sir.json → mathematical_spec`,
14/14 equations implemented — see Omission section below). The `KellyPortfolioLayer` (optional
Kelly-criterion QP) is correctly implemented as a separate, non-default-wired class not
referenced anywhere inside `ARTEMIS.__init__` — this correctly reflects the SIR's finding that
the paper never confirms this layer was used to produce any reported metric.

---

## Parametric Hallucinations

Hyperparameters marked `# ASSUMED` in the config, cross-referenced against `sir.json`'s
`implementation_assumptions` / `ambiguities`. All are explicit, documented config knobs — none
are silently hardcoded — but they remain candidate root causes for deviation in any future
real comparison.

| ID | Assumption | SIR confidence | Config location | Risk if wrong |
|----|------------|-----------------|-------------------|----------------|
| P-1 | Latent dimension d_z = 64 | 0.35 | `model.artemis.d_z` | Affects every module's capacity; inferred only from a Figure 3 caption, not confirmed as the actual training-time value |
| P-2 | Wiener dimension d_w = 16 | low (unspecified) | `model.artemis.d_w` | See H-1 above — also directly disables L_MPR at this specific value combined with d_z=64 |
| P-3 | SDE step count M = 100 | 0.3 | `model.artemis.n_sde_steps` | Affects Euler-Maruyama discretization error and simulated trajectory length; inferred only from a Figure 2 caption |
| P-4 | Loss weights lambda_1..4 | 0.3 | `model.artemis.loss_weights.*` | Paper only states these are "the same order of magnitude as the forecast loss" — any of a wide range of values would satisfy this description |
| P-5 | MPR Sharpe threshold kappa = 2.0 | 0.4 | `model.artemis.mpr_kappa` | Paper presents this as an illustrative "reasonable choice" for daily data specifically, not confirmed for the other 3 datasets/frequencies |
| P-6 | Symbolic basis library composition | 0.3 | `model.symbolic.basis_lags` | No enumeration given in paper at all; entirely an ArXivist-authored default |
| P-7 | Chronos-2 checkpoint name | low (unspecified) | `models/baselines/chronos2_wrapper.py::model_name` | Paper never gives an exact Hugging Face Hub identifier |
| P-8 | Conformal window/alpha | low (unspecified) | `model.conformal.*` | Not given in paper |
| P-9 | Batch size, epoch counts (except the 10-epoch DSLOB ablation) | low (unspecified) | `training.*` | Not given in paper for any model except the explicit "10 epochs" tied to Figure 6's ablation curves |

None of these were found to independently and definitively conflict with anything stated in
the paper text (they are genuine unresolved gaps in the source, not code errors), so none are
classified above Medium severity on their own — but P-1/P-2 combined are the direct cause of
H-1 above, which is why that finding is escalated.

---

## Omission Hallucinations

Components present in the SIR but absent or stubbed in the generated code.

**None found** for the core ARTEMIS architecture. Cross-checking every entry in
`sir.json → mathematical_spec` (14 equations) and `sir.json → architecture.modules` (6
modules) against the generated code:

| SIR equation/module | Implemented at |
|---|---|
| LNO kernel integral operator + Laplace-domain rational kernel | `models/laplace_neural_operator.py` |
| Neural SDE + Euler-Maruyama discretization | `models/neural_sde.py::NeuralSDE.simulate` |
| Diffusion Cholesky parameterization | `models/neural_sde.py::DiffusionNet` |
| Feynman-Kac PDE residual + PDE loss | `models/physics_losses.py::feynman_kac_residual, pde_loss` |
| Market price of risk + MPR loss | `models/physics_losses.py::market_price_of_risk, mpr_loss` |
| Consistency loss | `models/artemis.py::compute_losses` |
| Symbolic combination + L1 penalty + Gumbel-Softmax | `models/symbolic_bottleneck.py` |
| Distillation loss (Phase 2) | `scripts/train_model.py` (Phase 2 training loop) |
| Total composite loss | `models/artemis.py::compute_losses` |
| Conformal prediction interval | `models/conformal_allocation.py::AdaptiveConformalPredictor` |
| Kelly criterion portfolio QP | `models/conformal_allocation.py::KellyPortfolioLayer` |

All 5 baselines described in Section 5 (LSTM, Transformer, NS-Transformer, Informer, Chronos-2)
are implemented in `models/baselines/`. No `NotImplementedError`, `TODO`, or stub method
bodies were found outside `models/base`-style abstract declarations (there is no abstract base
class in this repo, unlike the bootstrap paper's repo, since ARTEMIS's modules are
heterogeneous by design).

**One intentional, documented omission**: **XGBoost**, named as a "6th baseline" in the
abstract, is not implemented. This is not an omission of something the SIR asked for — the SIR
itself flags this as a paper-internal inconsistency (`sir.json → ambiguities[0]`): XGBoost
never appears in Table 2 or any Section 5 baseline description, so there is nothing concrete
to implement it against. Implementing a 6th baseline with no reported numbers to validate
against would itself be a fabrication risk, which this repo explicitly avoids per the "do not
fabricate experimental results" instruction for this task.

---

## Additional Paper-Internal Inconsistencies (documented, not ArXivist errors)

These were found while parsing/generating code for this paper. They reflect issues in the
source paper itself, not defects introduced by ArXivist, but are catalogued here since they
directly affect what "faithful reproduction" can even mean for this paper.

1. **XGBoost baseline** (see above) — claimed in abstract, never reported.
2. **DSLOB feature count**: Section 5.1 states 59; Table 1 and Section 3.4 both state 85.
   `sir.json` and this repo use 85 (Table 1) as authoritative.
3. **Time-IMM directional accuracy**: the abstract/Introduction state ARTEMIS achieves 96.0%
   DirAcc on Time-IMM; Table 2 itself reports 0.860 (86.0%) for the identical ARTEMIS/Time-IMM
   cell. This discrepancy was noticed during Stage 4 code generation (not originally captured
   in the Stage 1 SIR) and documented in `README.md`'s "Reported results" section with an
   explicit footnote. **Recommendation for a future SIR re-parse**: add this as a fourth
   `ambiguities` entry alongside the XGBoost and DSLOB-feature-count findings.
4. **Table 2 vs Table 3 for DSLOB DirAcc**: Table 2 reports 64.96%; Table 3's `A0_Full` row
   (nominally the same full model) reports 64.89%. Already captured in `sir.json →
   ambiguities[4]` as within-expected-seed-variance, not a substantive conflict.

---

## Summary

| Type | Count | Critical | High | Medium | Low |
|------|-------|----------|------|--------|-----|
| Structural | 3 | 0 | 1 (H-2, paper limitation) | 1 (H-1) | 1 (H-3, already fixed) |
| Parametric | 9 | 0 | 0 | 0 | 9 (P-1..P-9, all pre-flagged SIR ambiguities) |
| Omission | 0 | 0 | 0 | 0 | 0 |
| Paper-internal inconsistencies (not ArXivist defects) | 4 | — | — | — | — |

**Overall assessment**: the generated implementation is a faithful, complete mapping of the
SIR — every claimed equation and module (6 modules, 14 equations, 5 baselines) is implemented
and was validated to execute correctly (Stage 4/5 placeholder-data smoke tests). The most
consequential finding (H-1) is a genuine configuration defect that silently disables part of
the paper's own ablation-validated mechanism when using the shipped defaults — this should be
fixed before any real comparison run is attempted. The largest reproducibility obstacle (H-2)
is not fixable at all by ArXivist or any downstream user, since it stems from the source
paper's own failure to identify its central dataset's provenance. The remaining parametric
gaps (P-1 through P-9) are consistent with this being a paper that omits nearly all of its
own hyperparameters — every one is exposed as an explicit, documented config knob rather than
a hidden default, per the architecture plan's Stage 3 policy.
