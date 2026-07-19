# Benchmark Comparison Report

**Paper**: Order Splitting and Liquidity Replenishment Are Jointly Necessary for the Square-Root Law of Market Impact: A Counterfactual Dissection
**Paper ID**: arxiv_2607_04280
**arXiv**: 2607.04280
**Comparison Date**: 2026-07-11
**SIR Version Used**: 1

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **0.00** / 1.0 | Low | 14 | 2 |

**Interpretation**:
- 0.90–1.00: Excellent reproduction
- 0.75–0.89: Good reproduction with minor deviations
- 0.60–0.74: Partial reproduction — review moderate deviations
- 0.40–0.59: Significant reproduction gap — likely implementation issues
- < 0.40: Critical failure — fundamental mismatch

---

## Metric Comparison Table

| Metric | Dataset | Split | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------|-------------|------------|-----------|----------|
| delta | Baseline (pooled) | demo (1 stock, 60k/1M steps) | 0.549 | 1.045 | +90.3% | 🔴 Critical |
| delta | Ablation: No splitting | demo (1 stock, 60k/1M steps) | 0.324 | 1.014 | +212.9% | 🔴 Critical |
| c (impact coefficient) | Baseline / No splitting | demo | 0.98 / 0.22 | not reported | — | ⬜ Unmatched |
| delta_no_HFT | Ablation: No HFT | — | 0.386 | not run | — | ⬜ Unmatched |
| delta_price_limits | Ablation: Price limits | — | 0.529 | not run | — | ⬜ Unmatched |
| delta_low_liquidity | Ablation: Low liquidity | — | 0.543 | not run | — | ⬜ Unmatched |
| delta_momentum | Ablation: Momentum | — | 0.549 | not run | — | ⬜ Unmatched |
| delta_uniform_split | Ablation: Uniform split | — | 0.547 | not run | — | ⬜ Unmatched |
| delta_front_loaded | Ablation: Front-loaded | — | 0.502 | not run | — | ⬜ Unmatched |
| delta_mean_2000_stocks | Baseline (2000 stocks) | full | 0.539 | not run | — | ⬜ Unmatched |
| beta_GGPS_mean | Baseline (2000 stocks) | full | 1.16 | not reported | — | ⬜ Unmatched |
| alpha_FGLW_mean | Baseline (2000 stocks) | full | 2.00 | not reported | — | ⬜ Unmatched |
| gamma_LOB_walking_mean_delta_pred | Baseline (2000 stocks) | full | 0.36 | not reported | — | ⬜ Unmatched |
| delta_TSE_benchmark | Tokyo Stock Exchange | empirical | 0.489 | n/a (external reference, not reproduced) | — | ⬜ Unmatched |

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | 0 |
| 🟢 Good (2–5%) | 0 |
| 🟡 Moderate (5–15%) | 0 |
| 🟠 Significant (15–30%) | 0 |
| 🔴 Critical (>30%) | 2 |
| ⬜ Unmatched | 12 |

---

## The bigger problem than the raw numbers

Beyond the absolute deviation, there is a **qualitative failure** that matters more than either single number: the paper's entire thesis is that removing order splitting *collapses* δ (0.549 → 0.324, a **41% drop**). In your run, baseline (1.045) and no-splitting (1.014) are statistically indistinguishable — a **3% difference**, in the *same* direction the paper reports but two orders of magnitude too small to be meaningful at this scale. The ablation's core causal signal is not present in your results yet. That's the thing to fix first, not the absolute δ level.

---

## Root Cause Analysis

### delta on Baseline — 90.3% deviation

**Likely causes** (ordered by probability):

1. **Config mismatch — HFT adaptive-spread rule is an unstated assumption** (High)
   This is flagged as the single **High-severity risk** in `architecture_plan.json`. The paper never gives the market-makers' spread formula; the shipped default (`base_spread_ticks=4`, `spread_vol_sensitivity=2.0`) may be posting too wide/thin a book, making every metaorder walk further than it should even in the baseline case — which would inflate δ uniformly, exactly what you're seeing.
   Fix: try narrowing `model.hft.base_spread_ticks` (e.g. 1–2) and/or reducing `spread_vol_sensitivity`, then re-run `inference.py` to check if δ moves toward 0.5.

2. **Training convergence — scale is ~16x too short** (High)
   You ran ~60k steps vs. the paper's 1e6, and 1 stock vs. the paper's 2000 (or 20 for ablations). The paper explicitly validates convergence by checking δ agrees within ±0.01 between the first and second half of a full 1e6-step trajectory — your run has no such check and is far short of that regime.
   Fix: run `python train.py --config configs/config.yaml` (full config, no `--debug`) for at least one stock, or better, `run_counterfactual_suite.py` at full scale (20 stocks x 8 scenarios) for a proper cross-stock average.

3. **Data mismatch — too few metaorders for a stable fit** (Medium)
   At demo scale, only ~140 valid (Q_norm, I_norm) pairs typically survive filtering, against 20 bins requiring ≥30 points each (the paper's default). The demo notebook silently relaxes `min_pts` to 5 to get *any* fit, which makes the fitted δ much noisier and biased.
   Fix: increase `n_steps`/`n_stocks` until you comfortably clear `min_points_per_bin=30` per bin without relaxing it.

4. **SIR uncertainty — Dirichlet concentration and news-agent process are also assumed** (Medium)
   Both are flagged Medium-confidence in the SIR (`dirichlet_concentration=1.0`, `news.trigger_rate=0.001`). These affect the fine shape of the metaorder-size and child-count distributions, which feed both δ directly and the β/α tail exponents.
   Fix: lower priority than #1, but worth sensitivity-testing once #1 and #2 are addressed.

5. **Randomness — single seed, no cross-stock averaging** (Medium)
   A single stock at 60k steps has high sampling variance; the paper's reported numbers are 2000-stock (or 20-stock) means ± SE.
   Fix: run multiple seeds/stocks and report mean ± SE, not a single point estimate.

### delta on No-splitting Ablation — 212.9% deviation

**Likely causes** (ordered by probability):

1. **Same root cause as baseline (HFT spread assumption) dominates the signal** (High)
   Since baseline itself is already inflated to δ≈1.0, the "no splitting" scenario has no room to show its expected *drop* — both scenarios may be saturated in the same steep-impact regime for a different reason (an over-wide/under-replenished book) that has nothing to do with splitting. Fix #1 from baseline first; re-test the ablation after.

2. **Structural check needed — confirm `force_single_child` is actually forcing 1 child per metaorder** (Medium)
   The generated code implements "no splitting" via `child_count_range=(1,1)` (see `dataset.py` / `market.py`). This is structurally correct per the SIR, but with only ~140 metaorders total at demo scale, verify this is actually taking effect by checking `Nc` in a debug print — a silent bug here would also produce "no visible ablation effect."
   Fix: add a temporary assertion/print of `n_children` in `InstitutionalAgent.generate_metaorder` for a handful of metaorders and confirm they're all size 1.

3. **Insufficient scale to resolve a 41% relative effect** (Medium)
   Even if everything else were correct, distinguishing a 0.549→0.324 drop reliably needs the statistical power of the paper's 20-stock counterfactual design, not 1 stock at 1/16th the trajectory length.
   Fix: same as baseline fix #2 — run at full scale.

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 3 | 1 |
| Omission | 0 | 0 |

---

## Recommended Actions

Prioritized by expected impact on reproducibility score:

1. **Re-run at full or near-full scale first** (`train.py` without `--debug`, or at minimum ~500k steps / 5+ stocks) before tuning anything else — at 60k steps / 1 stock, no config fix can be reliably evaluated because the noise floor is too high to tell a real improvement from luck.
2. **Sweep `model.hft.base_spread_ticks` and `spread_vol_sensitivity`** (the flagged High-severity assumption) once you have a scale where the fit is stable — this is the single parameter most likely to bring baseline δ down toward 0.5.
3. **Verify the no-splitting ablation is structurally firing** (print `n_children` for a sample of metaorders) — cheap to check, rules out a silent bug before you spend compute re-running at scale.
4. **Run `run_counterfactual_suite.py` at full scale (20 stocks x 8 scenarios)** once (1) and (2) look reasonable, to get the paper's actual comparison design (cross-stock mean ± SE) rather than a single-stock point estimate.
5. Only after the above: compute the additional metrics currently unmatched (c, β, α, γ, and the other 6 ablation scenarios) so the next comparison round can score all 14 paper metrics instead of 2.

---

## Implementation Notes

*From the SIR — sections with confidence < 0.7 that may affect these results:*

- **`architecture.HFTMarketMakerAgent`** (confidence 0.4 in architecture_plan risk assessment; SIR ambiguities[0]): adaptive-spread formula not specified in the paper — implemented as a volatility-scaled heuristic. Most likely single cause of your Critical deviations.
- **`architecture.NewsAgent`** (SIR confidence 0.65 / ambiguities[1], confidence 0.35): trigger rate and size multiplier are assumed; affects tail heaviness (κ) but is a second-order effect on δ compared to the HFT spread.
- **`implementation_assumptions[3]`** (confidence 0.55): Dirichlet concentration parameter for the default splitting rule is assumed as 1.0 (symmetric); untested against the paper's actual child-order-size distribution shape.
- **`implementation_assumptions[4]`** (confidence 0.4): price-limit band width (±10%) and momentum lookback (30 ticks) are assumed — not relevant to your current two results, but will matter once you run the other 6 scenarios.

---

## Verification Log Summary

- Comparison run at: 2026-07-11T00:00:00Z
- User results hash: `ae489aabe19a8ef1d6b7cbabf0ea247d0589d070aa0e04f41936beab8dcac2ba`
- User-reported config modifications: demo-scale run (`N_STEPS_DEMO=60_000` vs. paper's 1,000,000; 1 stock vs. paper's 20/2000; `min_pts` relaxed to 5 vs. paper's 30) via `notebooks/reproduction_walkthrough.ipynb`
- Manual review required: **Yes** — both matched metrics are Critical deviations, and the ablation's core causal signal (splitting matters) is not yet visible; recommend re-running at scale before further tuning.

Full audit trail in `verification_log.md`.
