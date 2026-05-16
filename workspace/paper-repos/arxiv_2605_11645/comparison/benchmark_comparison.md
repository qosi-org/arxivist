# Benchmark Comparison Report
**Paper**: GeomHerd: A Forward-looking Herding Quantification via Ricci Flow Geometry on Agent Interactive Simulations  
**Paper ID**: `arxiv_2605_11645`  
**Comparison Date**: 2026-05-13  
**Reproducibility Score**: N/A — Pre-run (no user results submitted yet)  
**Score Confidence**: Low (awaiting experimental results)

---

## Status: Pre-Run Analysis

No user experimental results have been submitted. This report provides:
1. Paper target values (from SIR) for each key metric
2. A priori deviation predictions based on known implementation assumptions
3. Full hallucination audit of generated code against the SIR

**To generate your results and populate this report:**
```bash
# Step 1: Generate CWS trajectories
for kappa in 0.5 0.8 1.2 1.8 2.5; do
    python run_detection.py --substrate cws --kappa $kappa --seeds 80 \
        --operating_point recall --output_dir results/detection/
done

# Step 2: Evaluate
python run_evaluation.py --results_dir results/detection/ --output results/eval_table.json

# Step 3: CCK regression
python run_cck_regression.py --data_dir results/detection/

# Step 4: Vicsek OOD
for eta in 0.5 1.0 1.6 2.0 2.5; do
    python run_detection.py --substrate vicsek --eta $eta --seeds 20
done
```

---

## Paper Target Metrics (from SIR `evaluation_protocol.geomherd_results`)

| Metric | Substrate | Config | Paper Value | 95% CI |
|--------|-----------|--------|-------------|--------|
| Median lead (κ̄⁺_OR) | CWS, 400 traj. | Recall-oriented (k=0.5, h=4) | **272 steps** | [236, 313] |
| Median lead (κ̄⁺_OR) | CWS, 400 traj. | Precision-oriented (k=2.0, h=4) | **178 steps** | [71, 407] |
| Recall_super (κ̄⁺_OR) | CWS precision | — | **0.04** | — |
| FAR_sub (κ̄⁺_OR) | CWS precision | — | **0.07** | — |
| Precision (κ̄⁺_OR) | CWS precision | — | **0.45** | — |
| Median lead (β⁻) | CWS, 400 traj. | CUSUM+Kendall τ=-0.4 | **318 steps** | [272, 344] |
| Recall_super (β⁻) | CWS contagion | — | **0.65** | — |
| FAR_sub (β⁻) | CWS contagion | — | **0.81** | — |
| AUROC (β⁻) | CWS contagion | — | **0.80** | — |
| γ₃ median (CCK regression) | CWS supercritical seeds | HAC Newey-West | **−0.0072** | [−0.00769, −0.00602] |
| Vicsek κ̄_OR(τ★) AUROC | Vicsek, 95/100 valid | At-event score | **0.99** | [0.98, 1.00] |
| Lead diff vs Lap-CSAD | CWS co-firing | Precision-oriented | **+153.8 steps** | [28.7, 297.5], p=0.03 |

---

## A Priori Deviation Predictions

### Fully Specified Components — Expected: Excellent (≤2%)

| Component | Confidence | Predicted Deviation |
|-----------|-----------|---------------------|
| AgentGraph construction (Eq. 1) | 0.97 | ≤ 2% — formula exact |
| ORC via LP-W1 (Eqs. 2-3) | 0.95 | ≤ 2% — POT library explicit |
| Sign decomposition κ± thresholds | 0.95 | ≤ 2% — ±0.1 explicit |
| CUSUM operating point params | 0.90 | ≤ 5% — (k, h) grid stated in Appendix D |
| Vicsek AUROC (≈0.99) | 0.82 | ≤ 5% — Vicsek mechanics standard |
| FSQ codebook (K=64) | 0.88 | ≤ 2% — 3D, Ld=4, K=64 explicit |

### Partially Specified — Expected: Moderate to Significant (5–30%)

| Component | Confidence | Predicted Deviation | Key Risk |
|-----------|-----------|---------------------|----------|
| CWS substrate (spin dynamics) | 0.75 | 15–30% | ASSUMED spin update + adjacency (Risk R5) |
| Median lead 272/178 steps | 0.80 | 15–30% | CWS mechanics + LLM agents withheld (R3) |
| β⁻ recall (0.65) | 0.65 | 10–25% | Kendall-τ threshold ASSUMED (R4) |
| γ₃ value (−0.0072) | 0.68 | 20–40% | Returns proxy != paper's actual per-asset returns |

### Low-Confidence Components — Expected: Critical (>30%)

| Component | Confidence | Predicted Deviation | Key Risk |
|-----------|-----------|---------------------|----------|
| τ_sing values | 0.60 | > 30% | Ricci flow update rule ASSUMED (Risk R1) |
| Kronos MAE | 0.45 | > 30% | Architecture absent (Risk R2 — STUB) |

---

## Summary

The implementation is architecturally faithful to the paper's pipeline for all fully-specified
components. The core geometric signal (κ̄_OR rising before herding, Vicsek transfer AUROC ≈ 0.99,
γ₃ < 0) is expected to reproduce. Absolute lead-time numbers (272/178/318 steps) are predicted
to show significant deviation (15–30%) primarily because: (1) the CWS substrate mechanics are
partially reconstructed from paper description; (2) LLM persona prompts are withheld and
replaced with rule-based fallback; (3) Ricci flow update rule is assumed multiplicative.
The Kronos head produces structurally valid outputs but absolute MAE numbers are not reproducible
from this implementation.

---

## Root Cause Analysis for Expected Significant Deviations

### Median Lead Time (Paper: 272 steps, Expected: 150–350 steps)

Likelihood-ordered root causes:

1. **CWS substrate approximation** (High)  
   Our implementation uses `tanh(f + κ·peer + η)` spin updates with Erdős-Rényi adjacency
   (p=0.3). The paper's Cividino-Sornette substrate is a full N-vector Ising model with
   specific coupling distribution. Different cascade onset timing will shift lead distribution.  
   *Fix*: Obtain Cividino et al. (2023) source code; match `sbase=0.6, spost=1.6` parameterization exactly.

2. **Rule-based vs LLM agents** (High)  
   LLM-persona agents produce richer subcritical action diversity → sharper κ̄_OR contrast at
   cascade onset → cleaner CUSUM trip. Rule-based agents produce different baseline variance,
   shifting baseline estimation in CUSUM.  
   *Fix*: Set `--llm_mode true` with `ANTHROPIC_API_KEY`; test persona prompt templates.

3. **CUSUM baseline window** (Medium)  
   W=35 snapshot samples × delta_t=10 = 350 simulator steps as baseline. If our trajectories
   are shorter or cascade occurs earlier, baseline estimate is noisier.  
   *Fix*: Confirm T_steps ≥ 500 for meaningful baseline establishment before cascade onset.

### γ₃ Magnitude (Paper: −0.0072, Expected: −0.003 to −0.015)

1. **Return series proxy** (High)  
   Our CCK regression feeds `order_parameter` as Rm and synthetic N_proxy=10 agents as CSAD
   proxy. Paper uses actual per-agent per-asset returns from full CWS price process with
   `beta_impact` and `sigma_xi` calibrated to match Cont (2001) stylized facts.  
   *Fix*: Store `info['returns']` from `CWSSubstrate.step()` and compute CSAD from actual returns.

2. **N_agents in CSAD** (Medium)  
   Paper uses N=66 agents × na=4 assets. Our proxy uses N_proxy=10. Different cross-section
   size changes CSAD variance and thus γ₃ scale.  
   *Fix*: Use the full 66-agent per-asset return cross-section.

---

## Recommended Actions (Priority Order)

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| P1 | Run full detection pipeline with 80 seeds × 5 κ values | Validates lead-time claims |
| P1 | Run Vicsek OOD (20 seeds × 5 η values) | High-confidence validation |
| P1 | Check γ₃ sign (should be < 0) | Proposition 1 sanity check |
| P2 | Store actual CWS per-asset returns; feed to CSAD | Fixes γ₃ magnitude |
| P2 | Verify CWS spin update against Cividino et al. (2023) | Fixes lead-time deviation |
| P3 | Enable `--llm_mode true` with persona prompts | Closes LLM agent gap |
| P3 | Test both Ricci flow variants (`multiplicative` / `additive`) | Validates τ_sing |
