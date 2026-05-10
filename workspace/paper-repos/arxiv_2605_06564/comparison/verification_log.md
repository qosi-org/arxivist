# Verification Log
**Paper**: Dynamic Treatment on Networks (arXiv:2605.06564)  
**Paper ID**: arxiv_2605_06564  
**ArXivist Pipeline Run**: 2026-05-09T00:00:00Z

---

## Pipeline Execution Summary

| Stage | Status | Duration (est.) | Output |
|-------|--------|----------------|--------|
| Stage 1 — Paper Parser | ✅ Complete | — | `sir.json` (v1) |
| Stage 2 — SIR Registry | ✅ Complete | — | `global_index.json` updated |
| Stage 3 — Architecture Planner | ✅ Complete | — | `architecture_plan.json`, `architecture_plan_summary.md` |
| Stage 4 — Code Generator | ✅ Complete | — | Full repository in `paper-repos/arxiv_2605_06564/` |
| Stage 5 — Notebook Generator | ✅ Complete | — | 2 Jupyter notebooks |
| Stage 6 — Results Comparator | ✅ Complete (pre-run) | — | 4 comparison artifacts |

---

## SIR Provenance

| Field | Value |
|-------|-------|
| SIR version | 1 |
| Parsed from | Uploaded PDF (arXiv:2605.06564v1) |
| Paper date | 2026-05-07 |
| Overall SIR confidence | 0.83 |
| Lowest confidence component | `implementation_assumptions` (0.65) |
| Highest confidence component | `evaluation_protocol` (0.95) |

---

## Paper Metrics Inventory

| Metric | Source Location | Value Available | Comparison-Ready |
|--------|----------------|----------------|-----------------|
| Q-Ising mean adoption (SBM) | Figure 1 left | Approximate only (~0.20) | ⚠️ Approximate |
| Degree-Bin mean adoption (SBM) | Figure 1 left | Approximate only (~0.09) | ⚠️ Approximate |
| Q-Ising village reward (Table 2) | Table 2, all 42 villages | Exact (3 decimal places) | ✅ Ready |
| EMVS Ising AUC | Figure 5 | 0.762 | ✅ Ready |
| Q-Ising vs Degree-Bin correlation | Section 5.2 | −0.5 (approximate) | ⚠️ Approximate |

**Total paper metrics found**: 5  
**Exact values available**: 2  
**Approximate (figure-read) values**: 3

---

## User Results Comparison Log

**Status**: No user results submitted.  
**Action required**: Run `train.py` and submit output to trigger comparison.

### Expected User Result Format

```json
{
  "experiment": "sbm",
  "n_runs": 50,
  "H": 25,
  "seed": 42,
  "results": {
    "Random":    {"mean_reward": 0.XXX, "std_reward": 0.XXX},
    "Degree":    {"mean_reward": 0.XXX, "std_reward": 0.XXX},
    "LIR":       {"mean_reward": 0.XXX, "std_reward": 0.XXX},
    "DegreeBin": {"mean_reward": 0.XXX, "std_reward": 0.XXX},
    "PlainDQN":  {"mean_reward": 0.XXX, "std_reward": 0.XXX},
    "Q-Ising":   {"mean_reward": 0.XXX, "std_reward": 0.XXX}
  }
}
```

---

## Implementation Decisions Recorded

The following decisions were made during code generation that may affect reproducibility:

1. **EMVS**: Used L1-penalized sklearn LogisticRegression as proxy for EM algorithm.  
   *Risk*: Higher if true EMVS uses a different penalty structure or convergence criterion.  
   *Mitigation*: Abstract base class `IsingFitter` allows swapping in a correct EMVS.

2. **MCMC**: PyMC with NUTS assumed. Continuous relaxation of spike-and-slab used.  
   *Risk*: Lower; NUTS is specified by paper citation. Spike-slab relaxation is standard.

3. **Village pipeline**: Implemented as partial stub pending village data download.  
   *Risk*: None for SBM comparison; significant if village results are target.

4. **Bin assignment (SBM)**: Spectral clustering assumed.  
   *Risk*: If authors used ground-truth block membership directly (likely), spectral may introduce misclustering noise on small N.

5. **LIR baseline**: Proxy formula used.  
   *Risk*: Moderate — could affect baseline rankings but not Q-Ising's absolute performance.

---

## Verification Checklist

- [x] SIR parsed from paper PDF
- [x] SIR registered in global index
- [x] Architecture plan generated with confidence annotations
- [x] All SIR equations implemented with equation citations
- [x] All low-confidence components marked `# WARNING`
- [x] All assumed components marked `# ASSUMED`
- [x] STUB components documented in hallucination report
- [x] Notebooks runnable on synthetic data (no downloads required)
- [ ] User training results submitted
- [ ] Empirical comparison computed
- [ ] Reproducibility score finalized

---

*Log maintained by ArXivist Stage 6 — Results Comparator*
