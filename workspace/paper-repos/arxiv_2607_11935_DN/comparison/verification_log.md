# Verification Log

**Paper ID**: arxiv_2607_11935
**Comparison run timestamp**: 2026-07-18T15:11:21Z

---

## Provenance

- **SIR version used**: 1 (`sir-registry/arxiv_2607_11935/sir.json`, `sir_version: 1`)
- **Architecture plan version used**: 1 (`sir-registry/arxiv_2607_11935/architecture_plan.json`, `plan_version: 1`, corrected post-hoc for 3 drift issues between the plan and the actually-built repo — see architecture_plan_summary.md)
- **Repo state at comparison time**: 46/46 tests passing, `ruff check` clean, both notebooks executed with zero errors prior to this comparison

## Input Data

- **Source**: user-pasted terminal/notebook output (two messages), not a file upload
- **Format**: plain text (printed `Simulation | Tipping t | beta lead | AR1 lead | Winner` table; printed `Table 1 row (synthetic Tropics)` dict; printed explore-notebook winner list)
- **SHA256 of concatenated user-provided results text**: `efa7f34215883e703c37475a830b0d4a61f985ca1206aacf276ba87754fd0935`
- **User-reported config modifications**: none stated. Results are consistent with default `configs/config.yaml` (seed=0, `use_synthetic_fallback: true`, default `evaluation.simulation` block).
- **Cross-check performed**: the six-system winner list from the *explore* notebook output
  (Fold bifurcation=AR1, Beta step change=beta, Beta linear decay=beta, Logistic map=tie,
  Stommel AMOC=beta, Critical slowing down=beta) was checked against the *reproduce*
  notebook's Table 3 output provided in the same turn. **They agree exactly**, which is
  expected since both use `seed=0` and the same default config — this confirms internal
  consistency of the two notebooks, but does not by itself validate against the paper
  (both notebooks share the same synthetic-data and simulation-parameter assumptions).

## Metrics Coverage

- **Total paper metrics available for comparison** (from `sir.json → evaluation_protocol.reported_results`): 15
- **Metrics the user's pasted output allowed us to match**: 6
  1. `|beta|` (Tropics)
  2. `sigma_beta` (Tropics)
  3. `r(beta, AR(1)_T)` (Tropics)
  4. `r(beta, MI)` (Tropics)
  5. `beta lead time over AR(1)` (Stommel AMOC)
  6. `beta lead time over AR(1)` (critical slowing down)
- **Unmatched** (user did not report Arctic or Monsoon region rows, nor the aggregate observational lead-lag range): 9
- **Additional qualitative-only comparisons** (no numeric paper ground truth to compute % deviation against, but a stated paper "winner" to check): Fold bifurcation, Logistic map, Beta step change, Beta linear decay — see `reproducibility_score.json → qualitative_winner_comparison`

## Computation Method

Reproducibility score computed exactly per `06_results_comparator.md`'s formula:

```
base_score = mean(1 - min(abs(pct_deviation) / 50, 1.0) for all matched pairs)
            = mean([0.996, 0.0, 0.0, 0.0, 0.026, 0.0])
            = 0.1703

sir_confidence_penalty = (1 - overall_sir_confidence) * 0.15
                       = (1 - 0.72) * 0.15
                       = 0.042

unmatched_penalty = (unmatched_count / total_paper_metrics) * 0.2
                  = (9 / 15) * 0.2
                  = 0.12

reproducibility_score = max(0, 0.1703 - 0.042 - 0.12) = 0.0083 -> rounded to 0.01
```

Score confidence classified as **Medium**: 6 metrics were matched (≥3, meeting the
"High" threshold on count alone), but the run used synthetic-fallback data and default
illustrative simulation parameters rather than the paper's real dataset — so while the
*measurement* of deviation is precise, what it says about the *paper's* reproducibility
(as opposed to this repo's synthetic-data behavior) is less certain. Downgraded from
High to Medium for this reason.

## Deviation Classification Audit

All 6 matched metrics' percentage deviations were computed as
`(user_value - paper_value) / paper_value * 100` and classified per the fixed
thresholds (≤2% Excellent, 2-5% Good, 5-15% Moderate, 15-30% Significant, >30%
Critical). Full arithmetic shown in `benchmark_comparison.md`. No manual adjustment or
rounding was applied to move any metric across a severity boundary.

## Hallucination Cross-Check

`architecture_plan.json`'s `module_hierarchy` was checked line-by-line against
`src/ews_kalman/`'s actual file tree and class definitions (not just filenames) to
confirm the "no omission hallucinations" finding in `hallucination_report.md`. All 10
planned modules have corresponding non-stub implementations with passing unit tests.

## Manual Review Flag

**Set to `true`.** Reason: the paper's central empirical claim (β orthogonal to AR(1))
is contradicted by this run's Tropics-region result. This is flagged for manual review
rather than presented as either "the paper doesn't reproduce" or "the code is broken" —
neither conclusion is warranted from a single synthetic-data run using seed=0. The
`hallucination_report.md`'s root-cause analysis indicates the synthetic data generator
is the most probable cause, but this has not been confirmed by comparing against a
same-analysis run on real NASA AIRS data, which is the necessary next step before
drawing any conclusion about the paper itself.

## Metadata Updated

- `sir-registry/arxiv_2607_11935/metadata.json`: `has_comparison_report` set to `true`,
  `last_updated` timestamp refreshed.
