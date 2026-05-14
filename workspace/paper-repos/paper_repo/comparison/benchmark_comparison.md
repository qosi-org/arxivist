# Benchmark Comparison Report
**Paper**: Volatility Modeling in Markovian and Rough Regimes: Signature Methods and Analytical Expansions  
**Paper ID**: arxiv_2605_volatility_sig  
**arXiv**: 2507.23392v4  
**Comparison Date**: 2026-05-13  
**Reproducibility Score**: PENDING — no user results submitted yet  
**SIR Version**: 1  
**Architecture Plan Version**: 1  

---

## Status

No user results have been submitted. This report documents:
1. The paper's ground-truth metrics (from SIR `evaluation_protocol.reported_results`)
2. All known implementation assumptions and their confidence levels
3. Pre-identified hallucinations in the generated code
4. Expected deviation ranges when user results are submitted

Submit your results by running:
```bash
python train.py --config configs/config.yaml --experiment heston_uncorr
```
Then feed `results/heston_uncorr/results.json` to the Results Comparator.

---

## Paper Ground-Truth Metrics

### Experiment 1: Heston Uncorrelated (Section 5.1, Table 5.1)
| Contract (T, K) | e^ASV (paper) | e^SIG (paper) | SIG wins? |
|-----------------|---------------|---------------|-----------|
| 0.1, 90  | 0.00004 | 0.00127 | No  |
| 0.1, 95  | 0.00002 | 0.00007 | No  |
| 0.1, 100 | 0.00005 | 0.00009 | No  |
| 0.1, 105 | 0.00003 | 0.00069 | No  |
| 0.1, 110 | 0.00003 | 0.00078 | No  |
| 0.6, 90  | 0.00010 | 0.00024 | No  |
| 0.6, 95  | 0.00012 | 0.00021 | No  |
| 0.6, 100 | 0.00012 | 0.00005 | **Yes** |
| 0.6, 105 | 0.00012 | 0.00026 | No  |
| 0.6, 110 | 0.00010 | 0.00019 | No  |
| 1.1, 90  | 0.00011 | 0.00029 | No  |
| 1.1, 95  | 0.00012 | 0.00008 | **Yes** |
| 1.1, 100 | 0.00012 | 0.00055 | No  |
| 1.1, 105 | 0.00012 | 0.00031 | No  |
| 1.1, 110 | 0.00012 | 0.00069 | No  |
| 1.6, 90  | 0.00011 | 0.00089 | No  |
| 1.6, 95  | 0.00012 | 0.00014 | No  |
| 1.6, 100 | 0.00012 | 0.00029 | No  |
| 1.6, 105 | 0.00012 | 0.00008 | **Yes** |
| 1.6, 110 | 0.00011 | 0.00031 | No  |

**Loss at convergence**: 1.05 × 10⁻⁴  
**SIG wins**: 3/20 contracts  
**ℓ* key indicators**: ℓ[0] ≈ 0.2012 ≈ σ₀, ℓ[2] ≈ 1.085 (strong X_t linear term)

---

### Experiment 2: Heston Correlated ρ=-0.5 (Section 5.2, Table 5.2)
| Contract (T, K) | e^ASV (paper) | e^SIG (paper) | SIG wins? |
|-----------------|---------------|---------------|-----------|
| 0.1, 90  | 0.00046 | 0.00261 | No  |
| 0.6, 90  | 0.00024 | 0.00018 | **Yes** |
| *(see Table 5.2 in paper for all 20 entries)* | | | |

**Loss at convergence**: 1.46 × 10⁻³  
**Note**: Correlated case is harder — effects encoded in higher-order signature terms partially captured at N=3.

---

### Experiment 3: Rough Bergomi (Section 6, Table 6.1)
| Contract (T, K) | e^VIX (paper) | e^SIG (paper) | SIG wins? |
|-----------------|---------------|---------------|-----------|
| 0.1, 90  | 0.00104 | 0.00126 | No  |
| 0.1, 105 | 0.00058 | 0.00004 | **Yes** |
| 0.2, 90  | 0.00079 | 0.00026 | **Yes** |
| 0.2, 105 | 0.00044 | 0.00027 | **Yes** |
| 0.4, 90  | 0.00060 | 0.00019 | **Yes** |
| 0.4, 100 | 0.00003 | 0.00001 | **Yes** |
| 0.4, 110 | 0.00067 | 0.00015 | **Yes** |
| 0.6, 110 | 0.00061 | 0.00037 | **Yes** |
| *(see Table 6.1 for all 20 entries)* | | | |

**Loss at convergence**: 3.5 × 10⁻⁴  
**SIG wins**: 7/20 contracts  
**Calibration time**: 17–19 minutes (RTX 3080 Ti, shifted-exp fBM primary)

---

## Pre-Submission Deviation Thresholds

When you submit results, deviations will be classified as:

| Severity | Threshold | Expected cause if exceeded |
|----------|-----------|---------------------------|
| Excellent | ≤ 2% | Within MC noise |
| Good | 2–5% | Normal training variance |
| Moderate | 5–15% | Config mismatch (likely: nMC, T_steps, box_bounds) |
| Significant | 15–30% | Implementation deviation (likely: shuffle table, X0 ambiguity) |
| Critical | > 30% | Fundamental error (likely: VIX stub, wrong Q assembly) |

---

## Root Cause Pre-Analysis

### High-risk deviations to watch for

**1. Loss values 5–20× higher than paper (Moderate/Significant)**  
Most likely cause: `nMC` too low or `T_steps_per_unit` too low.  
Paper uses nMC=800k; our default is 800k but if reduced for speed the loss floor rises.  
Fix: confirm `nMC=800000` in config and full run completion.

**2. Correlated Heston loss >> 1.46e-3 (Significant)**  
Most likely cause: box constraint bounds [-10,10] are active, or N=3 truncation insufficient.  
The paper notes "higher-order terms may be required to fully capture dependence structure."  
Fix: try N=4 (at ~4× compute cost) or widen box_bounds to [-20,20].

**3. Rough Bergomi SIG errors worse than VIX at most contracts (Significant)**  
Most likely cause: rho/H mismatch between primary process (H=0.2) and market (H=0.1).  
This is an intentional feature of the paper but if the gap is too large, the fBM primary is not tracking the market.  
Fix: verify H=0.2 for primary, H=0.1 for market as in Section 6.

**4. ℓ[0] ≠ σ₀ in Heston uncorrelated (Moderate)**  
Most likely cause: X0 ambiguity — if x0_is_variance=false, the model is initialised differently.  
Paper reports ℓ[0] ≈ 0.201 ≈ σ₀=0.2 and ℓ[2] ≈ 1.085.  
Fix: ensure `x0_is_variance: true` and `X0: 0.1` in heston_primary config.

---

## Recommended Actions (Pre-Submission)

1. **Run a debug pass first** (`--debug` flag, nMC=2000) to confirm the full pipeline runs without errors before committing to an 800k-path run.
2. **Monitor Q-matrix Cholesky** — if warnings appear about paths needing extra regularisation, the Q assembly may have numerical issues. Expected: <0.1% failure rate.
3. **Check ℓ* indicators** after calibration:
   - Heston uncorr: ℓ[0] should be ≈ 0.20, ℓ[2] ≈ 1.08
   - If these differ by >10%, the optimizer converged to a wrong local minimum
4. **Save intermediate results** — signatures are expensive; use `--resume` to restart from a saved ℓ if the optimizer run is interrupted.
