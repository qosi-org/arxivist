# Benchmark Comparison Report
**Paper**: Quantum Horizon — An evaluation of quantum computing as a threat to Bitcoin and Ethereum
**Paper ID**: arxiv_2606_14484
**Comparison Date**: 2026-07-21
**Reproducibility Score**: 0.826 (Medium confidence)

## Metric Comparison

| Metric | Paper Value | Your Value | Deviation | Severity |
|---|---|---|---|---|
| Logical qubits to break secp256k1 (frontier) | 1,350 | 1,325 (midpt of 1,200–1,450) | -1.85% | Excellent |
| Physical qubits to break secp256k1 (upper bound) | 500,000 | <500,000 | ~0.00% | Excellent |
| Best demonstrated physical qubits (2026) | 1,180 | 1,100 (midpt of 1,000–1,200) | -6.78% | Moderate |
| **P(CRQC by 2035)** | **0.167 (1-in-6)** | **0.122** | **-26.95%** | **Significant** |
| P(CRQC by 2040) | 0.300 | 0.308 | +2.67% | Good |
| P(CRQC by 2050) | 0.600 | 0.632 | +5.33% | Moderate |
| Bitcoin exposed at rest (% of supply) | 30.0% | 31.7% | +5.67% | Moderate |
| Bitcoin irreducibly-at-risk (M BTC) | 2.300 | 2.300 | 0.00% | Excellent |
| Bitcoin migratable exposure (M BTC) | 3.700 | 3.700 | 0.00% | Excellent |
| Ethereum at-rest exposure (midpoint) | 57.5% | 54.15% (midpt of 48.3–60.0%) | -5.83% | Moderate |
| Single quantum-machine hashrate @ 100 GHz (TH/s) | 21 | 20.69 | -1.48% | Excellent |
| Mempool-sniping realistic success probability | 30.0% | 28.7% | -4.33% | Good |
| Migration finish year, prompt 2026 start | 2030 | 2030 | 0.00% | Excellent |
| 2026 Bitcoin network hashrate (EH/s) | 860 | *not reported* | — | UNMATCHED |
| Ethereum dormant-and-exposed fraction | 0.1% | *not reported* | — | UNMATCHED |

Qualitative checks (not part of the numeric score, but verified against the paper):
- Top-20 quantum-readiness ratings (Table 3): your transcription and the CSV in `data/table3_readiness_ratings.csv` match the paper's ratings exactly for all 16 assets shown (XRP/Solana/Zcash at 4.0 down to Dogecoin at 1.5), including the "no coin reaches 5" claim. ✅
- Migration-race sweep: your 1-of-9 at-risk scenario count matches the paper's claim that only a severely-delayed start against an aggressive CRQC estimate is at risk. ✅

## Summary

Most headline numbers reproduce well: qubit-resource estimates, the Bitcoin exposure split, mining-hashrate figures, migration-race timing, and the Table 3 readiness ratings are all within 2% of the paper, several exactly. However, **the paper's single most quoted number — "about a one-in-six chance of a CRQC by 2035" (16.7%) — reproduces at 12.2% in your run, a 27% relative miss**, which is the one deviation this report cannot wave away as noise. Four other metrics (best-demonstrated qubits, P(CRQC by 2050), Bitcoin exposed-at-rest fraction, Ethereum at-rest exposure) land in the 5–7% "Moderate" band rather than "Excellent," consistent with these being reconciliations of multiple named data sources or a sampled Monte-Carlo forecast rather than closed-form values. Overall this is a good-but-not-perfect reproduction: the qualitative conclusions all hold (bounded threat, mining safe, migration wins the race, none of the top-20 is post-quantum), but the exact near-term probability the paper leads with is understated in your run.

## Root Cause Analysis

**P(CRQC by 2035): -26.95% (Significant)**
1. **SIR/implementation uncertainty (High probability)** — The architecture plan flags this as the single highest-severity risk in the pipeline: the paper does not specify the Monte-Carlo distributional forms for the physics estimator's swept parameters, the survey estimator's shape, or whether the two estimators are mixed at the sample level or the density level. The generated code fills all three gaps with `# ASSUMED` choices (uniform sweeps, a lognormal survey model, sample-level mixture) at SIR confidence 0.4–0.45. The paper itself notes the by-2035 figure is sensitive to the survey weight alone (8–24% across a 0.25–0.75 weight range), so a legitimate, differently-reasonable set of assumptions can land outside that band, and 12.2% is just below it. This is a modeling-assumption gap, not a bug.
2. **Randomness/seed (Medium probability)** — Monte-Carlo draws are stochastic; if `run_timeline_forecast.py` was run without a fixed seed, run-to-run variance alone could account for a few points of the gap, though not the full 27%.
3. **Config mismatch (Low probability)** — Worth double-checking that `configs/config.yaml`'s `survey_weight` was left at its default (0.5) and wasn't inadvertently changed before this run.

*Suggested fix*: re-run `run_timeline_forecast.py` with `--seed` fixed and confirm `survey_weight: 0.5` in `config.yaml`; then run `SystemicForecastModel.sensitivity_sweep()` (already implemented per the architecture plan) across weights 0.25–0.75 and report the by-2035 probability as a range rather than a point, matching how the paper itself frames it.

**Best demonstrated physical qubits: -6.78% (Moderate)**
1. **Not a modeled quantity (High probability)** — This is a reported hardware fact (IBM Condor, Atom Computing), not a calculation. Your 1,000–1,200 range vs. the paper's 1,180 suggests you may be using a slightly stale or rounded hardware figure. No fix needed beyond citing IBM Condor's exact 1,121-qubit spec if a tighter number is wanted.

**P(CRQC by 2050), Bitcoin exposed-at-rest %, Ethereum at-rest exposure midpoint: 5–7% (Moderate)**
1. **Reconciliation-method sensitivity (High probability)** — All three are outputs of a "combine several named sources" step (`reconcile_sources()`, `bottom_up_estimate()`) that the architecture plan explicitly marks as `ASSUMED: simple mean of sources` because the paper doesn't give an exact reconciliation formula. A simple mean vs. a source-quality-weighted average will shift these by a few percentage points, which is exactly the size of gap observed here. This is expected variance given the documented assumption, not an error.

## Recommended Actions

1. **Highest priority** — Re-run the timeline forecast with a fixed seed and report P(CRQC by 2035) as a range under the 0.25–0.75 survey-weight sweep, rather than a single point, to align with how the paper itself presents that number's sensitivity.
2. Verify `config.yaml`'s Monte-Carlo distributional-form and mixture-method settings haven't drifted from the documented `# ASSUMED` defaults.
3. For the three Moderate-severity reconciliation outputs, consider swapping `reconciliation_method` in `config.yaml` from `simple_mean` to a source-weighted variant and check whether that narrows the gap toward the paper's figures.
4. Report the two UNMATCHED metrics (2026 network hashrate, Ethereum dormant-and-exposed fraction) if you want full coverage of the paper's primary/secondary metric list.
