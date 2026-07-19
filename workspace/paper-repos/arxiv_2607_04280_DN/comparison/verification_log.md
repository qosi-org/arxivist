# Verification Log

**Paper ID**: arxiv_2607_04280
**Comparison run timestamp**: 2026-07-11T00:00:00Z

---

## Inputs Used

- `sir.json` — version 1 (overall confidence 0.79)
- `architecture_plan.json` — version 1
- User results (pasted text, not a file upload):
  ```
  baseline:     delta=1.045  (paper: 0.549)
  no_splitting: delta=1.014  (paper: 0.324)

  At demo scale (60k steps vs paper's 1M), expect much noisier estimates than
  the paper's 20-stock averages -- run run_counterfactual_suite.py for the
  full-scale reproduction.
  ```
- SHA256 of user results input (normalized string `"baseline: delta=1.045; no_splitting: delta=1.014"`):
  `ae489aabe19a8ef1d6b7cbabf0ea247d0589d070aa0e04f41936beab8dcac2ba`

---

## Metric Matching

- Paper metrics available in `sir.json → evaluation_protocol.reported_results`: **14**
- User-reported metrics: **2** (`delta` for `baseline`, `delta` for `no_splitting`)
- Matched by (metric name, dataset): **2 / 2** of the user's reported values found a paper
  counterpart; **2 / 14** of the paper's total tracked metrics were addressed by this
  comparison round.
- Metrics named but not compared (no user data provided): `c`, `beta`, `alpha`, `gamma`,
  and δ for the other 6 ablation scenarios (`no_hft`, `price_limits`, `low_liquidity`,
  `momentum`, `uniform_split`, `front_loaded`), plus the 2000-stock cross-sectional mean.

---

## User-Reported Config Modifications

Inferred from the user's message and the notebook they ran
(`notebooks/reproduction_walkthrough.ipynb`), not from an explicit config diff:

- `n_steps`: 60,000 (vs. paper/config default 1,000,000) — **16.7x shorter**
- `warmup_steps`: ~2,000 (vs. paper/config default 50,000)
- `n_stocks`: 1 (vs. paper's 2000 for the full baseline, 20 for counterfactual ablations)
- `min_points_per_bin`: relaxed to 5 in the demo notebook path (vs. paper/config default 30)
  — this was a deliberate notebook-level relaxation documented in
  `notebooks/reproduction_walkthrough.ipynb`'s markdown, needed to get *any* fit at demo
  scale; it is not a hidden bug, but it does mean the fitted δ is less reliable than the
  paper's own binning standard.
- No indication the user modified any `# ASSUMED` parameters in `configs/config.yaml`
  (e.g. HFT spread rule, Dirichlet concentration) — all defaults as shipped.

---

## Comparison Computation Trace

```
Metric: delta, Baseline
  paper_value = 0.549
  user_value  = 1.045
  absolute_deviation = 1.045 - 0.549 = 0.496
  percentage_deviation = 0.496 / 0.549 * 100 = 90.35%
  severity = Critical (>30%)

Metric: delta, No splitting
  paper_value = 0.324
  user_value  = 1.014
  absolute_deviation = 1.014 - 0.324 = 0.690
  percentage_deviation = 0.690 / 0.324 * 100 = 212.96%
  severity = Critical (>30%)

base_score = mean(1 - min(90.35/50, 1.0), 1 - min(212.96/50, 1.0))
           = mean(1 - 1.0, 1 - 1.0)
           = mean(0, 0) = 0.0

sir_confidence_penalty = (1 - 0.79) * 0.15 = 0.0315
unmatched_penalty = (12 / 14) * 0.2 = 0.1714

reproducibility_score = max(0, 0.0 - 0.0315 - 0.1714) = max(0, -0.2029) = 0.0

score_confidence = "low"
  (rationale: only 2/14 paper metrics matched, and the user substantially
  modified scale/config relative to the paper's protocol — both conditions
  for "low confidence" per the comparator methodology.)
```

---

## SIR / Architecture Plan Versions Used

- SIR version: 1 (unchanged since initial commit — not modified as part of this comparison)
- Architecture plan version: 1 (unchanged)
- No modifications were made to `sir.json` or any generated source file during this
  comparison run, per the comparator's operating constraints.

---

## Manual Review Flag

**Requires manual review: Yes.**

Reason: both matched metrics are Critical deviations (>30%), and more importantly the
paper's core causal claim (splitting removal collapses δ) is not reproduced even
qualitatively at this scale — baseline and no-splitting results differ by only ~3%
against an expected ~41% drop. This combination (Critical deviation + missing qualitative
effect) warrants the user re-running at larger scale and/or recalibrating the flagged
HFT-spread parameter before further automated comparison is likely to be informative.

---

## Files Produced This Run

- `benchmark_comparison.md`
- `reproducibility_score.json`
- `hallucination_report.md`
- `verification_log.md` (this file)
- `sir-registry/arxiv_2607_04280/metadata.json` updated: `has_comparison_report: true`
