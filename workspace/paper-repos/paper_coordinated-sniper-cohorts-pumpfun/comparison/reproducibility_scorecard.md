# Reproducibility Scorecard
**Paper**: Coordinated Sniper Cohorts on Pump.fun (Kamat, 2026)
**DOI**: 10.5281/zenodo.20978741
**Audit Mode**: Pre-run (Option C — no user results submitted)
**Scorecard Date**: 2026-07-15

---

## Overall Score

| Scenario | Score | Grade | Interpretation |
|---|---|---|---|
| **Without `--calibrate`** | **0.693 / 1.000** | **C+** | Structurally correct but τ mismatch causes cascading metric failures |
| **After `--calibrate`** | **0.881 / 1.000** | **B+** | Strong reproduction; residual uncertainty in rank aggregation and placebo variance |
| **After τ + rank fix** | **~0.930 / 1.000** | **A−** | Near-full reproduction of all deterministic metrics |

---

## Score Breakdown by Metric Group

| Group | Metrics | Predicted Score | Bottleneck |
|---|---|---|---|
| Detection — tau-independent | ablation counts, median_size, max_launches | **0.97** | None |
| Detection — tau-dependent | total_cohorts, wallets, mints_touched, tier counts | **0.42 → 0.95** | τ (resolves with --calibrate) |
| Causal lifts (buyer count) | naive lift, CI, top-3 excl, tier strat | **0.71 → 0.92** | τ cascade (resolves) |
| Causal lifts (SOL inflow) | naive SOL lift + CI | **0.62 → 0.84** | τ + SOL skew (partially resolves) |
| Placebo metrics | Design 1, Design 2 lift + CI + n_treated | **0.55 → 0.68** | Structural (n=173 is low-power by design) |
| Score function (EQ1) | COH-0001 score, mean_first_rank | **0.80** | Rank aggregation ambiguity (SIR conf. 0.72) |

---

## Qualitative Findings Reproducibility

These are the paper's key *claims* rather than numeric metrics.
Each is assessed independently of τ.

| Claim | Location | Reproducible? | Confidence |
|---|---|---|---|
| Persistent wallet cohorts exist (co-occurrence above chance) | §4, §5 | ✅ Yes — deterministic, data-driven | High |
| Size distribution is right-skewed (pairs dominate) | §5, Fig 1 | ✅ Yes — structural property of co-occurrence graphs | High |
| Cohort activity is concentrated (top 1% → >9% of touches) | §5, Fig 2 | ✅ Yes — Lorenz curve shape is robust | High |
| Naive +132.3% buyer-count lift on cohort-touched launches | §6.3 | 🟡 Directionally yes; exact value τ-sensitive | Medium |
| Activity-matched placebo lift > real-cohort lift | §6.4 | ✅ Yes — this is a selection effect, not τ-sensitive | High |
| No CI overlap between real cohort and placebo | §6.4 | 🟡 Likely yes, but placebo CI width varies with n=173 | Medium |
| Non-monotone tier pattern (Premium < Standard) | §6.5 | 🟡 Directionally yes; exact values τ-sensitive | Medium |

---

## Pass / Fail Checklist (run this after your first real run)

Use this checklist to self-audit your results against the paper:

```
TIER 1 — Must match exactly (τ-independent, deterministic):
  [ ] ablation cutoff=2 raw components == 1,562
  [ ] ablation cutoff=3 raw components == 1,161
  [ ] ablation cutoff=5 raw components ==   737
  [ ] median_cohort_size == 2
  [ ] max_launches_hit >= 42  (on same 15-day corpus)

TIER 2 — Must match after --calibrate (τ-dependent):
  [ ] total_cohorts within ±50 of 1,012
  [ ] unique_cohort_wallets within ±200 of 2,965
  [ ] mints_touched_strict within ±500 of 5,411
  [ ] premium_tier_cohorts within ±5 of 22
  [ ] high_tier_cohorts within ±20 of 153

TIER 3 — Should match directionally (causal, τ-sensitive):
  [ ] naive_buyer_lift_pct within ±15pp of +132.3%
  [ ] naive_sol_lift_pct within ±25pp of +136.5%
  [ ] top3_excl_lift within ±15pp of +128.8%

TIER 4 — Key qualitative checks (most important):
  [ ] placebo_d2_lift > naive_cohort_lift  ← CRITICAL: this is the paper's central claim
  [ ] tier_premium_buyer_lift < tier_standard_buyer_lift  ← non-monotone pattern
  [ ] COH-0001 score within ±15 of 430.44
  [ ] COH-0001 mean_first_rank within ±0.5 of 2.29
```

---

## Recommended Run Order

```bash
# Step 1: Smoke test (no real data needed)
python detect.py --buyers data/sample_buyers.jsonl
# Expected: a handful of cohorts (synthetic data only)

# Step 2: Ablation smoke test (Tier 1 checks)
python detect.py --buyers data/pumpfun_buyers.jsonl --ablation
# Check: results/appendix_a_ablations.csv → 1562, 1161, 737

# Step 3: Calibrated detection (Tier 2 checks)
python detect.py \
    --from-intra data/sniper_cohorts_intra.jsonl.gz \
    --calibrate --calibrate-target 1012
# Check: total_cohorts ≈ 1012

# Step 4: Descriptive analysis
python analyze.py --cohorts results/sniper_cohorts.jsonl
# Check: Fig 1 right-skewed, Fig 2 Lorenz concentration

# Step 5: Full causal analysis (Tier 3 + 4 checks)
python causal.py \
    --buyers data/pumpfun_buyers.jsonl \
    --launches data/pumpfun_launches.jsonl \
    --cohorts results/sniper_cohorts.jsonl
# Check: placebo_lift > naive_lift, tier non-monotone

# Step 6: Update this scorecard with your real numbers
# (paste causal_buyer_flow_summary.txt back into ArXivist Stage 6 re-run)
```

---

## Limitations of This Pre-Run Audit

1. **No actual deviations computed** — all severity ratings are predictions based on SIR confidence scores and known implementation risks. Actual deviations may be larger or smaller.

2. **Corpus not reproducible without data** — the paper's 15-day window (2026-06-12 to 2026-06-26) cannot be exactly re-collected from Solana RPC because the chain state is historical. The Zenodo Stage-1 checkpoint (`sniper_cohorts_intra.jsonl.gz`) is the only exact-reproduction path.

3. **Placebo variance is epistemic, not aleatoric** — the wide predicted range for placebo metrics (±40–80%) reflects genuine statistical variability at n=173, not uncertainty about the implementation. Even running the paper's own `gen_p7_artifacts.py` twice with different seeds would produce different placebo point estimates.

4. **τ is the dominant unknown** — once τ is calibrated and confirmed, this report should be re-run as a **post-run audit** (Option A or B) with your actual `causal_buyer_flow_summary.txt` to get precise deviation percentages and a final reproducibility grade.
