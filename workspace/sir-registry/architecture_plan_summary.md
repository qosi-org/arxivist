# Architecture Plan Summary — arxiv_2606_11859

**Paper**: Scenario Generation for Time Series and Curves: A Comparison of Nonparametric and Semiparametric Bootstrap
**Repo name**: `arxivist_bootstrap`

## Framework
Primary numerical stack is **numpy / pandas / scipy / statsmodels** — this is a statistical/econometric
methods paper, not a neural-network paper. `torch` is included only as an *optional* batched-simulation
engine for accelerating the 10,000-path Monte Carlo VAR recursions; every module also has a pure-numpy path.
No CUDA requirement, no HuggingFace, config via plain YAML.

## Module Hierarchy
- `data/loaders.py` — load & align the 4 historical series (Equity EUR, 1y rate, inflation, full yield curve)
- `data/transforms.py` — returns/differences transforms (Section 2)
- `models/base.py` — `BaseScenarioGenerator` abstract interface shared by all 3 methods
- `models/stationary_bootstrap.py` — geometric block bootstrap (Section 2)
- `models/var_bootstrap.py` — VAR(1) + residual resampling (Section 3)
- `models/nelson_siegel.py` — factor extraction & curve reconstruction (Section 4)
- `models/ns_var_bootstrap.py` — VAR(1) on NS latent factors (Section 5)
- `evaluation/metrics.py` — negative-increment stats, correlation, KL divergence (Section 6)
- `evaluation/arbitrage.py` — discount-factor monotonicity checks (Appendix B)
- `utils/config.py` — YAML config + seeding

## Entrypoints
1. `scripts/fit_models.py` — fit all three methodologies on historical data
2. `scripts/simulate.py` — generate N Monte Carlo paths for a chosen method
3. `scripts/evaluate.py` — reproduce Tables 1–11 comparison metrics
4. `scripts/plot_figures.py` — reproduce Figures 1–4

## Key Assumptions Flagged for Review (SIR confidence < 0.7)
| Assumption | Confidence | Config knob |
|---|---|---|
| Stationary Bootstrap mean block length | 0.4 | `model.stationary_bootstrap.mean_block_length` |
| Nelson-Siegel lambda calibration routine/value | 0.6 | `model.nelson_siegel.lambda_calibration` |
| KL-divergence KDE bandwidth | 0.4 | `evaluation.kl_kde_bandwidth` |
| i.i.d. vs. block residual resampling in VAR-Bootstrap | 0.65 | `model.var_bootstrap.residual_resampling` |
| Historical estimation sample window | 0.4 | `data.sample_window` |

All five are exposed as explicit, documented config parameters rather than hardcoded, per the
Stage 3 rule for SIR confidence < 0.6 ("design as easily swappable").

## Risk Assessment Highlights
- **High**: The paper's underlying data (MSCI Europe Net TR EUR, Italian BOT yields, HICP ex-tobacco,
  full IT yield curve) is proprietary and not distributable. The repo ships a synthetic-data generator
  so the full pipeline runs end-to-end without proprietary access, clearly labeled as illustrative only.
- **Medium** (×3): Nelson-Siegel lambda value unreported; KL-divergence bandwidth unreported; residual
  resampling granularity (i.i.d. vs. block) unreported. All mitigated via config flags, not silent defaults.
- **Low** (×2): Stationary Bootstrap block length unreported; numerical stability of NS loadings as
  tau → 0 (0/0 limit), mitigated with a small-tau Taylor fallback.

Full machine-readable plan: `architecture_plan.json`.
