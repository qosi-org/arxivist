# Hallucination Report

**Paper**: Order Splitting and Liquidity Replenishment Are Jointly Necessary for the Square-Root Law of Market Impact
**Paper ID**: arxiv_2607_04280
**Report Date**: 2026-07-11

This report cross-checks the generated repository (`src/sqrt_law_abm/`) against the SIR
(`sir.json`) and architecture plan (`architecture_plan.json`) to identify components that
were invented, guessed, or omitted during code generation — as opposed to genuine
implementation-detail gaps in the paper itself, which are separately tracked as SIR
`ambiguities` / `implementation_assumptions`.

**Summary: 0 structural hallucinations, 3 parametric hallucinations (1 critical), 0 omission
hallucinations.** The generated architecture is a faithful, complete mapping of every SIR
module (`LimitOrderBook`, all 4 base agent types, `MetaorderReconstructor`,
`ImpactCurveFitter`, `CounterfactualAblationHarness`, all 3 theory predictors) — nothing was
invented that isn't traceable to a paper section. The issues below are unstated *parameters*
within otherwise-correct structures, not incorrect structures.

---

## Structural Hallucinations

**None found.** Every class in `src/sqrt_law_abm/models/` and `evaluation/metrics.py` maps
to a named SIR module:
- `LimitOrderBook` → SIR `architecture.modules[0]`
- `InstitutionalAgent`, `HFTMarketMakerAgent`, `RetailAgent`, `NewsAgent` → SIR
  `architecture.modules[1-4]`
- `MetaorderReconstructor`, `ImpactCurveFitter`, `CounterfactualAblationHarness`,
  `TheoryPredictorGGPS/FGLW/LOBWalking` → SIR `architecture.modules[5-10]`

`MomentumAgent` is the one class with no SIR module entry, but it is explicitly named in the
paper's Table 1 ("Momentum" scenario) and documented in the architecture plan's
`config_schema` — not a hallucination, just a paper-named component the SIR's module list
didn't enumerate separately (a minor SIR completeness gap, not a code-generation error).

---

## Parametric Hallucinations

### 1. HFT adaptive-spread formula — **CRITICAL**

- **Location**: `src/sqrt_law_abm/models/agents.py`, `HFTMarketMakerAgent._current_spread_ticks`
- **Evidence**: The paper states HFT agents post "at l = 50 price levels with adaptive
  spread" (Section 2.2) but gives no formula. The generated code invents a specific
  volatility-scaled heuristic: `base_spread_ticks * (1 + spread_vol_sensitivity * vol * 100)`
  with `base_spread_ticks=4`, `spread_vol_sensitivity=2.0`. Both the functional form and the
  constants are fabricated (SIR `ambiguities[0]`, confidence 0.4).
- **Why this is flagged critical now**: it directly coincides with your two Critical
  deviations (baseline δ=1.045 vs. paper's 0.549). This is the single highest-probability
  root cause identified in `benchmark_comparison.md`.
- **Suggested fix**: Treat this as a free parameter to calibrate, not a fact. Sweep
  `base_spread_ticks` and `spread_vol_sensitivity` against the paper's baseline δ=0.549 and
  Figure 9's reported ~4-tick baseline spread, rather than assuming the current defaults are
  correct.

### 2. Dirichlet splitting concentration parameter — Minor

- **Location**: `configs/config.yaml`, `model.institutional.dirichlet_concentration: 1.0`
- **Evidence**: Paper names "Dirichlet" as the splitting rule (Section 2.2) but never gives
  a concentration parameter. `1.0` (symmetric, uniform-over-simplex prior) was chosen as the
  most standard default (SIR `implementation_assumptions[3]`, confidence 0.55).
- **Why minor and not critical**: it affects the *shape* of within-metaorder child-size
  variance, a second-order effect compared to the HFT spread issue above. Unlikely to
  explain a 90%+ deviation on its own.
- **Suggested fix**: Lower priority than #1. Revisit once the HFT spread calibration is
  resolved and if residual deviation remains.

### 3. News-agent trigger process — Minor

- **Location**: `src/sqrt_law_abm/models/agents.py`, `NewsAgent` (`trigger_rate=0.001`,
  `size_multiplier=20.0`)
- **Evidence**: Paper says news agents "occasionally inject large market orders" to
  reproduce heavy-tailed returns, without quantifying frequency or size (SIR
  `ambiguities[1]`, confidence 0.35).
- **Why minor**: affects tail kurtosis of returns, not directly the impact-curve δ fit,
  which only uses institutional-agent metaorders.
- **Suggested fix**: Lowest priority for fixing your current δ mismatch; relevant mainly if
  you later try to reproduce the paper's reported κ=29.5 return kurtosis.

---

## Omission Hallucinations

**None found.** Every SIR module has a corresponding, non-stubbed implementation:
- No `NotImplementedError` or `TODO`/`pass`-only methods exist in the shipped
  `src/sqrt_law_abm/` code (verified by inspection and by the fact that
  `run_counterfactual_suite.py` executed all 8 scenarios without hitting a stub).
- The `gamma` (LOB-walking exponent) computation exists (`estimate_depth_profile_gamma` in
  `evaluation/metrics.py`) but is **not yet wired into `evaluate.py`'s summary output** —
  `evaluate.py` currently appends `float("nan")` as a placeholder for gamma per stock. This
  is a real gap worth noting, though it's a wiring omission in the evaluation script rather
  than a missing model component — the underlying LOB depth-profile data (`depth_profile()`)
  is available, it's just not sampled and passed through during a full run. Not counted
  above as a hallucination since the SIR module itself (`TheoryPredictorLOBWalking`) is
  fully implemented and unit-tested; only the live end-to-end wiring for reporting gamma
  from a full run is incomplete.
  - **Suggested fix if you want the gamma/LOB-walking comparison**: in
    `evaluate.py`/`run_counterfactual_suite.py`, sample `lob.depth_profile("sell", 50)`
    periodically during the run (e.g. once per day) and average, then call
    `estimate_depth_profile_gamma` on the result.

---

## Overall Assessment

The generated code is a **structurally complete and faithful** implementation of everything
the paper specifies explicitly. All identified issues are in the category the paper itself
left underspecified (SIR `ambiguities` / `implementation_assumptions`), not in code
generation quality. The #1 parametric hallucination (HFT spread rule) is the most likely
explanation for your current Critical deviations and should be the first thing you
calibrate — see `benchmark_comparison.md`'s Recommended Actions.
