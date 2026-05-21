# Hallucination Report
**Paper**: GeomHerd (arXiv:2605.11645)  
**Paper ID**: `arxiv_2605_11645`  
**Generated**: 2026-05-13  
**Auditor**: ArXivist Stage 6 (Results Comparator)

This report audits all generated code against the SIR for three hallucination types:
1. **Structural hallucinations** — components in the code NOT supported by the SIR
2. **Parametric hallucinations** — assumed hyperparameters that may be incorrect
3. **Omission hallucinations** — SIR components absent or stubbed in the generated code

---

## 1. Structural Hallucinations
*Components invented by the code generator that are not in the SIR and may be wrong.*

### H-S1 — Erdős-Rényi Adjacency Matrix in CWSSubstrate
**Severity**: Significant  
**Type**: Structural  
**Location**: `src/geomherd/simulation/cws_substrate.py::_init_adjacency()`  
**Code**:
```python
adj = self._rng.random((N, N)) < 0.3  # Erdős-Rényi random graph p=0.3
```
**Evidence**: The paper states agents interact via "neighbors weighted by coupling strength κ" but never specifies the adjacency topology. Erdős-Rényi with p=0.3 was inserted without SIR support. The Cividino-Sornette substrate likely uses a fully-connected or structured coupling matrix consistent with the N-vector Ising model.  
**Impact**: Affects cascade onset timing, order-parameter dynamics, and ultimately lead-time numbers. Median lead deviation is directly tied to this choice.  
**Suggested Fix**: (a) Obtain Cividino et al. (2023) code and match adjacency exactly; or (b) try fully-connected adjacency (`adj = np.ones((N,N)) - np.eye(N)`) and compare cascade statistics to paper's Figure 3.

---

### H-S2 — Rule-Based Agent Fallback Architecture
**Severity**: Minor  
**Type**: Structural  
**Location**: `src/geomherd/simulation/llm_agent.py::RuleBasedAgentFallback.decide()`  
**Evidence**: Paper does not describe rule-based traders at all — it uses LLM-persona agents exclusively. The rule-based fallback is a convenience for cost-free testing but is not in the SIR.  
**Impact**: Rule-based agents produce qualitatively different subcritical action diversity, shifting the CUSUM baseline estimate and potentially the precision/recall tradeoff.  
**Suggested Fix**: Mark clearly as a non-paper component. Validate that core geometry results (Vicsek AUROC, γ₃ sign) hold under rule-based fallback before comparing to paper numbers.

---

## 2. Parametric Hallucinations
*Assumed hyperparameters marked `# ASSUMED` that may differ from the paper.*

### H-P1 — Ricci Flow Update Rule (Risk R1 — HIGH SEVERITY)
**Severity**: Critical  
**Type**: Parametric  
**Location**: `src/geomherd/geometry/ricci_flow.py::DiscreteRicciFlow.run()`  
**Assumed value**: Multiplicative: `w(e) ← w(e) × (1 − η × κ(e))`, `η=0.01`  
**SIR state**: "Update rule not specified; only stopping criterion (neckpinch) given" (confidence 0.60)  
**Evidence**: Section 2.4 states only τ_sing = first neckpinch time; no iteration formula given. Common discrete Ricci flow variants (Lin-Lu-Yau, Ollivier, Forman) differ substantially in convergence behavior.  
**Impact**: τ_sing values will be wrong if the update rule is incorrect. This directly affects the Kronos forecasting head conditioning and any τ_sing-dependent metrics (Table 3 τ_sing row: precision=0.42, recall=0.03, median_lead=-93).  
**Suggested Fix**: (a) Contact authors; (b) compare τ_sing AUROC under both `multiplicative` and `additive` variants against Table 3 (τ_sing should give AUROC≈0.48); (c) the multiplicative rule is most consistent with standard graph Ricci flow literature.

---

### H-P2 — Kendall-τ Threshold τ_thresh = −0.4 (Risk R4)
**Severity**: Significant  
**Type**: Parametric  
**Location**: `src/geomherd/detection/cusum.py::KendallTauDetector.__init__()`; `configs/config.yaml`  
**Assumed value**: `tau_thresh = -0.4`, `W_tau = 20`  
**SIR state**: "Inferred from Table 3 row label 'τ_neg=−0.4'" (confidence 0.65)  
**Evidence**: Table 3 labels the β⁻ detector row as "(τ_neg=−0.4, CUSUM+slope, up)" — we parsed this as the Kendall-τ threshold, but it may refer to the CUSUM minimum-curvature threshold. W_tau is not stated anywhere.  
**Impact**: β⁻ recall (0.65) and AUROC (0.80) both depend on the OR-combination of CUSUM and Kendall-τ. Wrong τ_thresh shifts the β⁻ detection boundary.  
**Suggested Fix**: Sweep τ_thresh in {-0.6, -0.4, -0.2, 0.0, 0.2} on the CWS replay set; target β⁻ recall=0.65, AUROC=0.80, FAR_sub=0.81 (Table 3) to calibrate.

---

### H-P3 — Kronos Head Architecture (Risk R2 — HIGH SEVERITY)
**Severity**: Critical  
**Type**: Parametric  
**Location**: `src/geomherd/forecasting/kronos_head.py`; `configs/config.yaml::kronos`  
**Assumed values**: `d_model=64, n_layers=2, n_heads=4, tokeniser_codebook_size=512, context_len=64`  
**SIR state**: Architecture not specified in paper (confidence 0.45)  
**Evidence**: Paper says "Kronos-style discrete price tokeniser... feeds a transformer that consumes the GeomHerd triplet via AdaLN-Zero conditioning." No layer counts, hidden dims, or codebook size given.  
**Impact**: Cascade-window MAE numbers are not reproducible. The architecture may be significantly larger (typical Kronos models use d_model≥256, n_layers≥4).  
**Suggested Fix**: This is a structural stub. Contact authors or search for public Kronos implementation. The AdaLN-Zero conditioning mechanism is correctly implemented; only scale is unknown.

---

### H-P4 — CWS Spin Update Noise Parameters
**Severity**: Significant  
**Type**: Parametric  
**Location**: `src/geomherd/simulation/cws_substrate.py::__init__()`  
**Assumed values**: `sigma_f=0.5, sigma_eta=0.3, action_threshold=0.3, beta_impact=0.1, sigma_xi=0.02`  
**SIR state**: "Agent action driven by private signal + neighbor average + idiosyncratic noise; linear price impact rule" — no numeric values given (confidence 0.75)  
**Evidence**: Paper references Cividino et al. (2023) for substrate mechanics. Our noise parameters are uncalibrated guesses. Paper reports tail index α=5.75 and volatility-ACF slope β=0.27 — our substrate should be validated against these stylized facts.  
**Suggested Fix**: Check if simulated returns from our CWSSubstrate match Cont (2001) stylized facts: compute tail index (should be ≈5.75) and vol-ACF decay slope (should be ≈0.27). If not, recalibrate sigma parameters.

---

### H-P5 — VQ-VAE Price Tokeniser Architecture
**Severity**: Moderate  
**Type**: Parametric  
**Location**: `src/geomherd/forecasting/kronos_head.py::PriceTokeniser`  
**Assumed values**: `embed_dim=32, codebook_size=512`, 2-layer MLP encoder  
**SIR state**: "Frozen learned vector-quantiser on OHLCV sequences" — no architecture given (confidence 0.45)  
**Evidence**: Follows from the Kronos head stub; tokeniser architecture is equally unspecified.  
**Suggested Fix**: Same as H-P3 — treat entire Kronos + tokeniser block as a structural stub requiring author clarification.

---

### H-P6 — FSQ Codebook Placement
**Severity**: Minor  
**Type**: Parametric  
**Location**: `src/geomherd/geometry/vocabulary.py::FSQVocabularyTracker.__init__()`  
**Assumed value**: Uniform grid over [−1, 1]³  
**SIR state**: "3D, Ld=4, K=64 explicitly stated; grid placement not stated" (confidence 0.88)  
**Evidence**: Mentzer et al. (2024) FSQ places levels at `{−2, −1, 0, 1}` or `{−1.5, −0.5, 0.5, 1.5}` depending on the variant, not necessarily uniform in [−1,1].  
**Impact**: Minor effect on V_eff values (entropy is sensitive to codebook utilization distribution, not absolute positions). Expected deviation: ≤5%.  
**Suggested Fix**: Use FSQ paper's recommended level placement: {−2, −1, 0, 1} for Ld=4.

---

### H-P7 — LLM Persona Prompt Template
**Severity**: Moderate  
**Type**: Parametric  
**Location**: `src/geomherd/simulation/llm_agent.py::PersonaAgent.PERSONA_PROMPT_TEMPLATE`  
**Assumed value**: Generic 5-field prompt (risk_appetite, momentum_horizon, herding_tendency, prices, returns, majority_action)  
**SIR state**: "Intentionally withheld by authors" (confidence 0.55)  
**Evidence**: Paper states prompts are withheld to prevent adversarial replication. Our template is a best-effort approximation that may miss key behavioral cues (e.g., market sentiment framing, specific financial reasoning chain).  
**Impact**: Different prompt → different action distribution → different subcritical baseline → shifted CUSUM calibration.  
**Suggested Fix**: Consult co-authors or test multiple prompt variants; optimize persona prompts to match reported subcritical behavioral diversity statistics if available.

---

## 3. Omission Hallucinations
*Components present in the SIR but absent or only partially implemented.*

### H-O1 — Full CWS N-Vector Ising Mechanics
**Severity**: Significant  
**Type**: Omission  
**Location**: `src/geomherd/simulation/cws_substrate.py`  
**Missing**: Full Cividino-Sornette N-vector O(n) spin model mechanics, including: exact coupling topology, sbase/spost noise schedule, multi-asset correlated fundamentals, and calibrated price impact matching Cont (2001) stylized facts.  
**SIR state**: "Described at high level; full mechanics requires Cividino et al. (2023)" (confidence 0.75, Risk R5)  
**Evidence**: Paper Appendix B states `sbase=0.6, spost=1.6` but does not provide the full update equations. Simulated returns should reproduce α=5.75 (tail index), β=0.27 (vol-ACF slope), and martingale raw returns.  
**Suggested Fix**: Implement the stylized-facts validation in `run_detection.py` — compute tail index and vol-ACF slope and print as a sanity check at the top of each trajectory batch.

---

### H-O2 — 13F Fund-as-Agent Deployment Template
**Severity**: Minor  
**Type**: Omission  
**Location**: Not implemented (Appendix I)  
**Missing**: The quarterly 13F holdings graph described in Appendix I — funds as nodes, Jaccard overlap as edges, sliding window W=4 quarters, recovery of 2008Q4/2011Q3/2020Q1/2022Q1 stress periods.  
**SIR state**: Present in paper Appendix I; explicitly noted as "out of scope for body claims" (no quantitative claim made)  
**Evidence**: Paper states "no quantitative claim from this retrospective in the body." This is a template/demo appendix, not a primary result.  
**Suggested Fix**: Low priority. Can be added as `notebooks/13f_retrospective.ipynb` for completeness. Not required for primary result reproduction.

---

## Hallucination Summary Table

| ID | Type | Severity | Component | Confidence Impact |
|----|------|----------|-----------|-------------------|
| H-S1 | Structural | Significant | CWS adjacency (Erdős-Rényi p=0.3) | Lead-time deviation ±15-30% |
| H-S2 | Structural | Minor | Rule-based fallback agents | Baseline diversity different |
| H-P1 | Parametric | **Critical** | Ricci flow update rule (multiplicative assumed) | τ_sing >30% deviation |
| H-P2 | Parametric | Significant | Kendall-τ threshold (−0.4 assumed) | β⁻ recall ±10-25% |
| H-P3 | Parametric | **Critical** | Kronos head architecture (all dims assumed) | MAE >30% deviation |
| H-P4 | Parametric | Significant | CWS noise parameters (uncalibrated) | Lead-time deviation |
| H-P5 | Parametric | Moderate | VQ-VAE tokeniser architecture | Kronos MAE (secondary) |
| H-P6 | Parametric | Minor | FSQ level placement | V_eff ≤5% deviation |
| H-P7 | Parametric | Moderate | LLM persona prompt template | Subcritical action diversity |
| H-O1 | Omission | Significant | Full CWS N-vector Ising mechanics | Cascade timing accuracy |
| H-O2 | Omission | Minor | 13F fund-as-agent template (Appendix I) | No primary result impact |

**Total hallucinations**: 11 (2 structural, 7 parametric, 2 omission)  
**Critical**: 2 (H-P1: Ricci flow rule; H-P3: Kronos architecture)  
**Significant**: 4 (H-S1, H-P2, H-P4, H-O1)  
**Moderate/Minor**: 5

---

## Critical Hallucination Action Items

### H-P1 Action Plan (Ricci Flow Rule)
1. Test `multiplicative` vs `additive` variant on 10 CWS supercritical trajectories
2. Compare τ_sing AUROC — paper Table 3 reports AUROC=0.48 for τ_sing (near chance)
3. If multiplicative gives AUROC≈0.48, rule is consistent (near-chance is expected: τ_sing is a descriptor, not a binary classifier)
4. If AUROC is substantially different, try additive or contact authors

### H-P3 Action Plan (Kronos Architecture)
1. Acknowledge this is a stub — do not report Kronos MAE numbers as paper-comparable
2. Use as a structural validation only: verify the AdaLN-Zero conditioning mechanism compiles and runs
3. Contact authors for architecture specification or search for public Kronos codebase
4. Report relative ordering (GeomHerd-conditioned vs price-only AR) rather than absolute MAE
