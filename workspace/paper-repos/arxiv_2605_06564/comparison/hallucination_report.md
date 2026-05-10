# Hallucination Report
**Paper**: Dynamic Treatment on Networks (arXiv:2605.06564)  
**Paper ID**: arxiv_2605_06564  
**Generated**: 2026-05-09  
**ArXivist SIR Version**: 1

This report audits the generated implementation against the SIR for structural
hallucinations, parametric assumptions, and omissions.

---

## 1. Structural Hallucinations

Components added to the generated code that are NOT described in the paper.

| # | Component | File | Description | Severity | Evidence |
|---|-----------|------|-------------|----------|---------|
| S1 | `PlainDQNPolicy.get_raw_state()` returns K-dim state (not 2K) | `evaluation/baselines.py` | Paper doesn't specify what state Plain DQN uses. K-dim (observed only) was inferred as the natural ablation. | **Minor** | Paper says "offline RL framework without Ising augmentation" — using raw y_bar_t-1 is a reasonable but unverified interpretation. |
| S2 | `merge_small_communities()` as standalone utility | `utils/community_detection.py` | Extracted from Appendix E.3 text. The paper says "merged into largest" but doesn't show this as a separate function. | **Minor** | Matches paper text exactly; structural choice only. |

**Structural hallucination count: 2 (both Minor)**

---

## 2. Parametric Hallucinations

Hyperparameters that were assumed without explicit paper support.

| # | Parameter | Location | Assumed Value | Confidence | Evidence / Risk |
|---|-----------|----------|---------------|-----------|----------------|
| P1 | Q-network activation function | `configs/sbm_default.yaml`, `training/cql_trainer.py` | ReLU | **0.80** | Standard for CQL/d3rlpy. Not stated in paper. Risk: tanh or GELU could produce different Q-value magnitudes. |
| P2 | EMVS inner solver | `models/ising.py:fit_emvs()` | L1-penalized sklearn LogisticRegression | **0.62** | Paper describes EMVS as "alternating between posterior inclusion probabilities and weighted penalized logistic regression." True EMVS uses a specific EM update (Ročková & George 2014). **This is the highest-risk assumption in the codebase.** |
| P3 | HMC library | `models/ising.py:fit_mcmc()` | PyMC with NUTS | **0.65** | Paper cites Hoffman & Gelman (2014) — NUTS. Could be NumPyro or Stan. API differences may cause subtle sampling behavior differences. |
| P4 | SBM bin assignment method | `configs/sbm_default.yaml` | Spectral clustering | **0.78** | Paper uses 4-block SBM with K=4 bins. Ground-truth block identity likely used directly; spectral is a reasonable proxy. |
| P5 | LIR score formula | `evaluation/baselines.py:LIRPolicy` | degree(i) / mean_neighbor_degree(i) | **~0.60** | Liu et al. (2017) defines LIR but formula is not reproduced in the Q-Ising paper. This is a proxy that may not match exact baseline. |

**Parametric hallucination count: 5**  
**Critical risk: P2 (EMVS solver)** — this affects Stage 1 outputs and therefore all downstream state quality.

---

## 3. Omission Hallucinations

Components described in the SIR that are absent or stubbed in the generated code.

| # | Component | Where Expected | Status | Severity | Suggested Fix |
|---|-----------|---------------|--------|----------|--------------|
| O1 | True EMVS EM algorithm (Ročková & George 2014) | `models/ising.py` | **STUBBED** — L1-logistic proxy used | **Significant** | Implement the full E-step (compute posterior inclusion probs z_{ij} = p(γ_{ij} ≠ 0 \| data)) and M-step (solve weighted penalized logistic regression with spike-slab weights). Reference the EMVS R package source for the exact update equations. |
| O2 | Village experiment full pipeline | `train.py:run_village_experiment()` | **PARTIAL STUB** — imports and data loading only; core pipeline commented out | **Significant** | Mirror `run_sbm_experiment()` structure. The village pipeline is identical except for community detection and per-village iteration. Estimated effort: 30–50 lines of code. |

**Omission hallucination count: 2**

---

## 4. Overall Risk Assessment

| Risk Level | Count | Components |
|-----------|-------|-----------|
| 🔴 Critical | 0 | — |
| 🟠 Significant | 2 | EMVS solver (O1), village pipeline stub (O2) |
| 🟡 Moderate | 1 | LIR score formula (P5) |
| 🟢 Minor | 4 | S1, S2, P1, P3, P4 |

---

## 5. Suggested Fixes (Priority Order)

### Fix 1 (Highest Priority): Implement True EMVS [O1, P2]

Replace `fit_emvs()` in `src/q_ising/models/ising.py` with the genuine EM algorithm:

```python
def fit_emvs(self, panel):
    # E-step: z_{ij} <- p(gamma_{ij} != 0 | data, current theta)
    # Use posterior inclusion prob: z = sigmoid(log(c/(1-c)) + 0.5*log(v1/v0) + 0.5*gamma^2*(1/v0 - 1/v1))
    # M-step: solve weighted L2-penalized logistic regression per node
    # Weight for gamma_{ij}: w_{ij} = z_{ij}/v1 + (1-z_{ij})/v0
    # Repeat until convergence (2-3 steps per paper)
```

Reference: Ročková & George (2014), JASA, "EMVS: The EM Approach to Bayesian Variable Selection"

### Fix 2: Complete Village Pipeline [O2]

In `train.py:run_village_experiment()`, add the 3-stage pipeline body (identical to `run_sbm_experiment()` with village-specific data loading).

### Fix 3: Verify LIR Baseline [P5]

Check Liu et al. (2017) "A Fast and Efficient Algorithm for Mining Top-k Nodes in Complex Networks" for the exact LIR score definition and update `evaluation/baselines.py:LIRPolicy` accordingly.

### Fix 4 (Optional): PyMC → NumPyro migration [P3]

If PyMC installation causes issues (common on some platforms), an alternative is NumPyro which is lighter and JAX-based. The NUTS sampling interface is nearly identical.

---

*This report was generated without user training results. Re-run after training to update with empirical deviation analysis.*
