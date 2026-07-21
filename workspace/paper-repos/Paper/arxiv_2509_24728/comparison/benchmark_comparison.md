# Benchmark Comparison Report

**Paper**: Beyond Softmax: A Natural Parameterization for Categorical Random Variables  
**Paper ID**: arxiv_2509_24728  
**Authors**: Alessandro Manenti, Cesare Alippi  
**Venue**: ICML 2026 (PMLR 306)  
**Comparison Date**: 2026-07-16  
**Reproducibility Score**: N/A — Pre-run mode (no user results submitted yet)  
**Score Confidence**: N/A  
**ArXivist SIR Version**: 1  
**Mode**: PRE-RUN AUDIT — Paper targets recorded; user result columns show PENDING

---

## How to Use This Report

This report was generated **before** running experiments. It serves two purposes:

1. **Lookup table**: exact paper targets to compare against when you run training
2. **Prospective hallucination audit**: known implementation risks that could cause deviations

When you have results, paste them into the `Your Value` column and re-run Stage 6 with your numbers.

---

## Metric Comparison Tables

### Experiment 1 — Graph Structure Learning (Table 2, catnat-ν)

All values are reported as `mean ± std` over 10 seeds. Statistical significance: Welch's t-test, p < 0.05.

| θ* | Metric | Paper Value (catnat-ν) | Your Value | Abs. Dev. | % Dev. | Severity |
|----|--------|----------------------|------------|-----------|--------|----------|
| 0.1  | ES         | 7.425 ± 0.014  | PENDING | — | — | — |
| 0.1  | PP-MAE     | 0.3837 ± 0.0015 | PENDING | — | — | — |
| 0.1  | PP-MSE     | 0.617 ± 0.004  | PENDING | — | — | — |
| 0.1  | MAE(θ)     | 0.0052 ± 0.0003 | PENDING | — | — | — |
| 0.25 | ES         | 10.859 ± 0.012 | PENDING | — | — | — |
| 0.25 | PP-MAE     | 0.8201 ± 0.0012 | PENDING | — | — | — |
| 0.25 | PP-MSE     | 1.304 ± 0.003  | PENDING | — | — | — |
| 0.25 | MAE(θ)     | 0.0051 ± 0.0003 | PENDING | — | — | — |
| 0.5  | ES         | 14.937 ± 0.023 | PENDING | — | — | — |
| 0.5  | PP-MAE     | 1.2537 ± 0.0019 | PENDING | — | — | — |
| 0.5  | PP-MSE     | 2.466 ± 0.007  | PENDING | — | — | — |
| 0.5  | MAE(θ)     | 0.0061 ± 0.0006 | PENDING | — | — | — |
| 0.75 | ES         | 10.674 ± 0.012 | PENDING | — | — | — |
| 0.75 | PP-MAE     | 0.7969 ± 0.0015 | PENDING | — | — | — |
| 0.75 | PP-MSE     | 1.267 ± 0.004  | PENDING | — | — | — |
| 0.75 | MAE(θ)     | 0.0043 ± 0.0002 | PENDING | — | — | — |
| 0.9  | ES         | 7.340 ± 0.015  | PENDING | — | — | — |
| 0.9  | PP-MAE     | 0.3973 ± 0.0015 | PENDING | — | — | — |
| 0.9  | PP-MSE     | 0.607 ± 0.003  | PENDING | — | — | — |
| 0.9  | MAE(θ)     | 0.0023 ± 0.0001 | PENDING | — | — | — |

**Paper command to reproduce:**
```bash
python scripts/run_gsl_grid.py --parameterization natural --theta_star 0.5
# Repeat for theta_star in {0.1, 0.25, 0.75, 0.9}
```

**Key watchpoint**: MAE(θ) is the most sensitive metric and the one where catnat-ν shows the largest gains over softmax. If your MAE(θ) is not substantially lower than softmax's (~0.013), suspect a REINFORCE gradient flow issue or wrong score initialization. See hallucination report, H-03.

---

### Experiment 2 — Categorical VAE / MNIST (Table 3)

Metric: Negative Log-Likelihood estimated with **512 importance samples** (IWAE bound).  
Optimizer: Adam. Lower is better. Bold = statistically significant best (p < 0.05).

| N | K | Param | Dataset | Paper NLL | Your NLL | Abs. Dev. | % Dev. | Severity |
|---|---|-------|---------|-----------|----------|-----------|--------|----------|
| 10 | 8  | catnat-ν | MNIST        | 99.8 ± 0.4  | PENDING | — | — | — |
| 10 | 8  | softmax  | MNIST        | 100.9 ± 0.5 | PENDING | — | — | — |
| 10 | 16 | catnat-ν | MNIST        | 97.6 ± 0.2  | PENDING | — | — | — |
| 10 | 16 | softmax  | MNIST        | 98.1 ± 0.7  | PENDING | — | — | — |
| 10 | 32 | catnat-ν | MNIST        | 96.9 ± 0.4  | PENDING | — | — | — |
| 10 | 32 | softmax  | MNIST        | 98.6 ± 0.7  | PENDING | — | — | — |
| 20 | 8  | catnat-ν | MNIST        | 97.7 ± 0.2  | PENDING | — | — | — |
| 20 | 8  | softmax  | MNIST        | 97.8 ± 0.2  | PENDING | — | — | — |
| 20 | 16 | catnat-ν | MNIST        | 97.0 ± 0.4  | PENDING | — | — | — |
| 20 | 16 | softmax  | MNIST        | 97.5 ± 0.5  | PENDING | — | — | — |
| 20 | 32 | catnat-ν | MNIST        | 96.9 ± 0.4  | PENDING | — | — | — |
| 20 | 32 | softmax  | MNIST        | 98.2 ± 0.8  | PENDING | — | — | — |
| 10 | 8  | catnat-ν | Binary MNIST | 83.2 ± 0.5  | PENDING | — | — | — |
| 10 | 8  | softmax  | Binary MNIST | 84.9 ± 0.8  | PENDING | — | — | — |
| 20 | 16 | catnat-ν | Binary MNIST | 76.6 ± 0.3  | PENDING | — | — | — |
| 20 | 16 | softmax  | Binary MNIST | 78.1 ± 0.4  | PENDING | — | — | — |

**Paper command to reproduce:**
```bash
python train.py --experiment vae --parameterization natural --N 20 --K 16
python evaluate.py --experiment vae --checkpoint checkpoints/vae/best.pt
```

**Key watchpoint**: The expected catnat-ν advantage over softmax is 0.5–2.0 NLL points. Differences smaller than 0.3 are not statistically significant given 5-seed variance. If you see catnat-ν *worse* than softmax, suspect the Gumbel injection point (H-02) or VAE conv dims (H-05). See hallucination report.

---

### Experiment 3 — Reinforcement Learning (Table 4)

Metric: Final Episodic Return (mean ± std over 10 seeds, top-10 configs from 160-trial TPE search).  
Higher is better. Note: RL results have very high variance by design.

| Env | Param | Paper Return | Your Return | Abs. Dev. | % Dev. | Severity |
|-----|-------|-------------|-------------|-----------|--------|----------|
| Breakout  | softmax   | 398 ± 25   | PENDING | — | — | — |
| Breakout  | catnat-ν  | 406 ± 34   | PENDING | — | — | — |
| Seaquest  | softmax   | 1875 ± 312 | PENDING | — | — | — |
| Seaquest  | catnat-ν  | 2164 ± 533 | PENDING | — | — | — |

**Paper command to reproduce:**
```bash
python scripts/run_rl_tpe.py --env BreakoutNoFrameskip-v4 --parameterization natural --n_trials 160
python scripts/run_rl_tpe.py --env SeaquestNoFrameskip-v4 --parameterization natural --n_trials 160
```

**Key watchpoint**: RL results have 160-trial hyperparameter search baked in. Without this search, results are expected to be 15–40% lower. The reported numbers are the *top-10 config average* — if you skip the TPE search and use default hyperparameters, treat any deviation up to 40% as Moderate rather than Critical.

---

## Baseline Comparisons (Softmax, Sparsemax, catnat-σ)

The paper reports results for all four parameterizations. Reproducing the baselines is as important as catnat-ν for validating the comparison framework:

| Experiment | Metric | softmax | sparsemax | catnat-σ | catnat-ν |
|-----------|--------|---------|-----------|----------|----------|
| GSL θ*=0.5 ES | lower | 14.990 | 15.030 | 15.020 | **14.937** |
| GSL θ*=0.5 MAE(θ) | lower | 0.0132 | 0.0126 | 0.0101 | **0.0061** |
| VAE MNIST N=20 K=16 NLL | lower | 97.5 | 102.1 | 96.9 | **97.0** |
| RL Seaquest return | higher | 1875 | — | — | **2164** |

If your softmax baseline is substantially different from these, the issue is in training configuration, not the catnat implementation itself.

---

## Summary

No user results have been submitted yet. This report records the paper's target values from Table 2, 3, and 4 for all three experiments (GSL, VAE, RL), across all θ* settings, N/K configurations, and all four parameterizations. 

The prospective hallucination audit (see `hallucination_report.md`) identifies **8 hallucination risks** in the generated implementation, with 1 classified Critical (VAE decoder spatial dimensions), 2 Significant, and 5 Minor. The Critical hallucination has been patched in Stage 5; the remaining risks are documented with suggested verification steps.

When you have experimental results, re-run Stage 6 by providing your numbers in the format:
```json
{"metric": "NLL", "dataset": "MNIST", "N": 20, "K": 16, "parameterization": "natural", "value": 97.2, "seeds": 5, "optimizer": "adam"}
```

---

## Recommended Pre-Run Actions (Priority Order)

1. **[CRITICAL — do first]** Verify VAE decoder output resolution. Run `python train.py --experiment vae --debug --dry_run` and confirm `recon.shape == [B, 1, 28, 28]`. If not, the conv dims patch in `models/vae.py` needs further adjustment.

2. **[HIGH]** Before full GSL run, verify Energy Score implementation against a known value: with identical predictions and target, ES should be 0. Run `tests/test_losses.py::TestEnergyScore::test_zero_when_perfect_and_zero_variance`.

3. **[HIGH]** For the VAE experiment, verify the 512-sample IWAE bound implementation: `python evaluate.py --experiment vae --checkpoint ... --n_eval_samples 10` should produce a finite scalar. Scale to 512 only after confirming correctness.

4. **[MEDIUM]** Run the full test suite before starting long training: `python -m pytest tests/ -v`. All 4 test files must pass.

5. **[MEDIUM]** For RL: set `n_trials=20` for a quick sanity check before committing to the full 160-trial TPE search. Confirm episodic returns are non-zero after 1M timesteps.

6. **[LOW]** Compare your softmax baseline against the paper's reported softmax numbers first. If softmax matches within 5%, catnat-ν deviations are more reliably attributable to the parameterization itself.
