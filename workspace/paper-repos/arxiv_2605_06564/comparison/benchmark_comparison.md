# Benchmark Comparison Report
**Paper**: Dynamic Treatment on Networks  
**Paper ID**: arxiv_2605_06564  
**Comparison Date**: 2026-05-09  
**Reproducibility Score**: N/A — no user results submitted yet  
**Status**: PRE-RUN BASELINE (awaiting user training output)

---

## How To Use This Report

Run training first, then submit results to the Results Comparator:

```bash
python train.py --config configs/sbm_default.yaml
python train.py --config configs/village_default.yaml
```

Then provide your `results/sbm_results.npy` or `results/villages/` output.

---

## Paper Target Metrics (from SIR evaluation_protocol)

### SBM Experiment (Section 5.1)

| Policy | Metric | Paper Value (approx) | Source |
|--------|--------|---------------------|--------|
| Q-Ising | Mean adoption rate (H=25) | ~0.20 (highest) | Figure 1 left, Section 5.1 |
| Plain DQN | Mean adoption rate (H=25) | ~0.15 | Figure 1 left |
| Degree-Bin | Mean adoption rate (H=25) | ~0.09 | Figure 1 left |
| Degree | Mean adoption rate (H=25) | ~0.07 | Figure 1 left |
| Random | Mean adoption rate (H=25) | ~0.08 | Figure 1 left |

*Note: Exact numeric values for SBM are not tabulated in the paper (only Figure 1 plots are shown). Values above are read from Figure 1 and are approximate.*

### Microfinance Villages (Section 5.2, Table 2 — 42 villages)

| Policy | Metric | Paper Value | Source |
|--------|--------|-------------|--------|
| Q-Ising | Mean reward (Village 0) | 0.061 ± 0.002 | Table 2 |
| Plain DQN | Mean reward (Village 0) | 0.057 ± 0.002 | Table 2 |
| Degree-Bin | Mean reward (Village 0) | 0.074 ± 0.002 | Table 2 |
| Q-Ising | Mean reward (Village 39) | 0.107 ± 0.002 | Table 2 |
| Plain DQN | Mean reward (Village 39) | 0.103 ± 0.002 | Table 2 |
| Q-Ising | Mean reward (Village 37) | 0.124 ± 0.002 | Table 2 |

*(Selected villages; full table in paper Table 2)*

### Model Quality

| Metric | Paper Value | Source |
|--------|-------------|--------|
| EMVS Ising AUC (pooled) | 0.762 | Figure 5 |
| Q-Ising improvement range | −20% to +50% over Degree-Bin | Figure 1 right |
| Modularity-improvement correlation | −0.5 | Section 5.2 |

---

## Metric Comparison Table (Placeholder — Update After Training)

| Metric | Dataset | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------------|------------|-----------|----------|
| Mean adoption rate (Q-Ising) | SBM H=25 | ~0.20 | *not yet run* | — | — |
| Mean adoption rate (Degree-Bin) | SBM H=25 | ~0.09 | *not yet run* | — | — |
| Ising AUC (pooled) | Microfinance villages | 0.762 | *not yet run* | — | — |
| Mean reward Village 39 (Q-Ising) | Microfinance | 0.107 | *not yet run* | — | — |

---

## Predicted Reproducibility Assessment

Based on SIR confidence scores and known implementation assumptions:

**Expected reproducibility: MODERATE (0.55–0.70)**

Rationale:
1. **CQL hyperparameters are fully specified** (Appendix E.2): alpha, discount, hidden layers, lr, batch size → expect low deviation here once d3rlpy is installed correctly.
2. **EMVS implementation is underspecified** (SIR confidence 0.62): The proxy L1-logistic regression will produce different coefficient magnitudes than the true EM algorithm. This may affect state quality and thereby Q-Ising performance.
3. **SBM results are in Figure 1 only** (no exact numbers published): Comparison against SBM targets will be approximate.
4. **Village experiment requires real adjacency data** from Harvard Dataverse; results depend on correct data loading.

---

## Recommended Actions Before Running

1. Verify `d3rlpy` version matches requirements.txt (`>=2.3`)
2. Verify `pymc>=5.0` if using ensemble/MCMC variant
3. Download village data: `python data/download_villages.py`
4. Use `--debug` flag for a fast smoke test before full training:
   ```bash
   python train.py --config configs/sbm_default.yaml --debug --dry-run
   ```
5. Set `--seed 42` to match paper's reproducibility intent

---

*Submit your results to update this report.*
