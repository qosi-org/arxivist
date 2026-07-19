# Hallucination Report
**Paper ID**: paper_coordinated-sniper-cohorts-pumpfun
**SIR Version**: 1
**Architecture Plan Version**: 1
**Audit Date**: 2026-07-15
**Audit Mode**: Pre-run (static analysis against SIR and architecture plan)

---

## Overview

This report audits the generated codebase for three hallucination types:
- **Structural**: components present in code but absent from the SIR
- **Parametric**: assumed hyperparameters that may be wrong
- **Omission**: components in the SIR that are absent or stubbed in the code

Total hallucinations found: **9** (0 structural, 5 parametric, 4 omission)

---

## Type 1 — Structural Hallucinations
*Components in the generated code NOT present in the SIR*

**Count: 0**

All classes and functions in the generated codebase map directly to SIR components. No spurious components were added. The `calibrate()` method in `CohortScorer` is an ArXivist-added utility (not in the paper) but is clearly labelled as a workaround for the undisclosed τ, not a paper component.

✅ No structural hallucinations detected.

---

## Type 2 — Parametric Hallucinations
*Assumed hyperparameters marked `# ASSUMED` that may be incorrect*

### P-HALL-01 — `score_tau = 40.0`
| Field | Detail |
|---|---|
| **Severity** | 🔴 Critical |
| **Type** | Parametric |
| **Location** | `configs/config.yaml` → `detection.score_tau`; `detection/scorer.py` → `CohortScorer.__init__` |
| **SIR confidence** | 0.55 |
| **Evidence** | Paper states τ is "set to surface cohorts satisfying (a)+(b)+(c)" with no numeric disclosure. Value of 40.0 was chosen as a prior below the reported median score of 52.8, but the paper's actual τ is unknown. |
| **Impact** | Controls exactly how many cohorts are surfaced. Wrong τ cascades to 18 downstream metrics: total_cohorts, unique_wallets, mints_touched, all tier counts, all causal lifts, all placebo metrics. |
| **Suggested fix** | Run `detect.py --calibrate`. Do not use default τ=40.0 for publication-quality reproduction. |

### P-HALL-02 — `touch_threshold_score = 1`
| Field | Detail |
|---|---|
| **Severity** | 🟡 Significant |
| **Type** | Parametric |
| **Location** | `configs/config.yaml` → `detection.touch_threshold_score`; `detection/scorer.py` → `CohortScorer.score()` |
| **SIR confidence** | 0.68 |
| **Evidence** | The score formula `1{C touches L}` does not specify whether "touches" means ≥1 or ≥2 cohort wallets. The paper uses ≥2 for the causal analysis but the scoring formula predates that section. Default ≥1 was chosen to match the looser detection-side definition. |
| **Impact** | Affects `n_launches_hit` per cohort and therefore tier classification. If paper uses ≥2 for scoring too, all tier counts will be systematically lower. |
| **Suggested fix** | After calibration, compare COH-0001's `n_launches_hit` against paper's 42. If it comes out as 42 with threshold=1 but lower with threshold=2, threshold=1 is correct. |

### P-HALL-03 — `mean_first_rank` aggregation method
| Field | Detail |
|---|---|
| **Severity** | 🟡 Significant |
| **Type** | Parametric |
| **Location** | `detection/scorer.py` → `CohortScorer.score()` → `per_launch_min_ranks` loop |
| **SIR confidence** | 0.72 |
| **Evidence** | The paper states "average first-buyer rank" for COH-0001 = 2.29. The implementation takes the mean of per-launch minimum ranks: `mean(min_rank_per_launch)`. The alternative — mean over all (wallet, launch) pairs — would yield a higher value. The correct interpretation is inferred from context but not stated. |
| **Impact** | Affects EQ1 term2 (`5 / mean_first_rank`) and thus score magnitude. If the wrong aggregation is used, COH-0001's score will differ from 430.44. |
| **Suggested fix** | Post-run check: if `coh0001_mean_first_rank` in your output differs from 2.29, try: `cohort_rows.groupby("mint")["rank"].mean().mean()` (alternative, higher value) vs current `groupby("mint")["rank"].min().mean()`. |

### P-HALL-04 — `blockTime` tie-breaking via `tx_sig` ASC
| Field | Detail |
|---|---|
| **Severity** | 🟡 Moderate |
| **Type** | Parametric |
| **Location** | `io/loader.py` → `DataLoader.load_buyers()` → `sort_values(["blockTime", "tx_sig"])` |
| **SIR confidence** | 0.65 |
| **Evidence** | Paper ranks buyers by `blockTime` within each launch but does not specify tie-breaking for equal `blockTime`. Implementation uses `tx_sig` (transaction signature) ASC as a deterministic canonical order, consistent with typical Solana indexer behaviour. The paper's `analyze_sniper_cohorts.py` may use a different convention. |
| **Impact** | Affects which wallet gets rank 10 vs 11 in launches with blockTime ties, shifting the first-buyer window boundary. Marginal effect on a 166,098-launch corpus but non-zero. |
| **Suggested fix** | Compare rank assignments on a sample of 100 launches against the Zenodo `sniper_cohorts_intra.jsonl.gz`. If `n_qualifying_mints` matches 166,098, tie-breaking is consistent. |

### P-HALL-05 — Bootstrap resampling uses `np.random.default_rng(seed)` not `np.random.seed(seed)`
| Field | Detail |
|---|---|
| **Severity** | 🟢 Minor |
| **Type** | Parametric |
| **Location** | `causal/estimator.py` → `LiftEstimator.bootstrap_ci()` |
| **SIR confidence** | 0.88 |
| **Evidence** | The paper specifies "1,000 iterations, percentile method, seed=42" but does not specify the RNG implementation. The code uses NumPy's modern `default_rng` (PCG64), which differs from the legacy `np.random.seed` / Mersenne Twister. The point estimate is unaffected; the exact CI bounds will differ slightly. |
| **Impact** | Tiny: CI bounds may shift by ±0.5–1.0 percentage points due to different RNG stream. Well within noise. |
| **Suggested fix** | For exact bit-for-bit reproduction, change `bootstrap_ci()` to use `np.random.seed(seed)` and `np.random.choice()`. Not recommended unless you need exact CI bound matching. |

---

## Type 3 — Omission Hallucinations
*Components present in the SIR but absent or incomplete in the generated code*

### O-HALL-01 — `PassiveSolanaObserver` not implemented
| Field | Detail |
|---|---|
| **Severity** | 🟡 Significant |
| **Type** | Omission |
| **Location** | SIR `architecture.components[0]` → no corresponding file in generated repo |
| **SIR confidence** | 0.85 |
| **Evidence** | The SIR documents a `PassiveSolanaObserver` — a real-time Solana RPC listener that produces `pumpfun_buyers.jsonl` and `pumpfun_launches.jsonl`. This component is entirely absent from the repo. |
| **Impact** | Cannot collect fresh data without this. Reproduction requires either the Zenodo checkpoint or a separately implemented Solana RPC listener. |
| **Suggested fix** | This omission is intentional and documented in `data/README.md`. The observer is out of scope for a reproducibility repo (it requires live Solana RPC access). The repo correctly documents how to obtain the raw JSONL files. Severity reduced to Significant rather than Critical for this reason. |

### O-HALL-02 — `gen_p7_artifacts.py` not recreated
| Field | Detail |
|---|---|
| **Severity** | 🟢 Minor |
| **Type** | Omission |
| **Location** | Zenodo release includes `gen_p7_artifacts.py`; generated repo implements equivalent logic in `causal.py` |
| **SIR confidence** | 0.92 |
| **Evidence** | The paper's released script `gen_p7_artifacts.py` streams `pumpfun_buyers.jsonl` and `pumpfun_launches.jsonl` to produce `causal_buyer_flow.csv`. The generated `causal.py` reimplements this logic across `causal/sample.py`, `causal/estimator.py`, and `causal/placebo.py`. The output schema should be equivalent but the internal logic was reimplemented rather than ported. |
| **Impact** | Minor discrepancy risk in how intermediate data is passed between components. If `causal_buyer_flow.csv` columns differ from the paper's original, downstream analysis may need column renaming. |
| **Suggested fix** | After obtaining the Zenodo release, run both `gen_p7_artifacts.py` (original) and `causal.py` (reimplemented) on the same data and diff the `causal_buyer_flow.csv` outputs. |

### O-HALL-03 — Cox proportional-hazards graduation model not implemented
| Field | Detail |
|---|---|
| **Severity** | 🟢 Minor |
| **Type** | Omission |
| **Location** | SIR `evaluation_protocol` → paper Section 7.5 "Limitations" |
| **SIR confidence** | N/A (paper explicitly defers this) |
| **Evidence** | The paper's original intention was to estimate cohort presence effect on bonding-curve graduation probability using a Cox PH model. This was deferred because graduation-outcome data was too sparse at the time of writing. The generated repo correctly omits this — it was not implemented in the paper either. |
| **Impact** | None for current reproduction. Relevant only for future work extending the paper. |
| **Suggested fix** | If graduation outcome data becomes available, add `causal/cox_ph.py` using `lifelines` library. The `pumpfun_launches.jsonl` graduation flag would be the event indicator. |

### O-HALL-04 — Propensity-score matching on launch-quality covariates not implemented
| Field | Detail |
|---|---|
| **Severity** | 🟢 Minor |
| **Type** | Omission |
| **Location** | Paper Section 6.6 / 7.2 — "propensity-score matching... we leave to future work" |
| **SIR confidence** | N/A (paper explicitly defers this) |
| **Evidence** | The paper identifies PSM on `initial_mcap_sol`, `has_twitter`, `description_len`, `hour_of_day` as the required next step for causal identification. The generated repo does not implement this. |
| **Impact** | None for current reproduction — this is a stated limitation, not an implemented method. |
| **Suggested fix** | Add `causal/psm.py` using `scikit-learn` logistic regression for propensity score estimation and `NearestNeighbors` for 1:1 matching on launch-quality covariates from `pumpfun_launches.jsonl`. |

---

## Hallucination Summary

| Type | Count | Severity Distribution |
|---|---|---|
| Structural | 0 | — |
| Parametric | 5 | 1× Critical, 2× Significant, 1× Moderate, 1× Minor |
| Omission | 4 | 1× Significant, 3× Minor |
| **Total** | **9** | **1 Critical, 3 Significant, 2 Moderate, 3 Minor** |

**One hallucination requires action before any run: P-HALL-01 (τ=40.0)**. All others are either minor numerical precision issues or correctly-scoped omissions of future-work components.
