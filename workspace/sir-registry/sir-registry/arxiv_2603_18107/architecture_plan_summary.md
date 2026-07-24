# Architecture Plan Summary — arxiv_2603_18107

**Paper**: ARTEMIS: A Neuro-Symbolic Framework for Economically Constrained Market Dynamics
**Repo name**: `arxivist_artemis`

## ⚠ Human Review Required
This paper carries forward from Stage 1/2 with `human_review_required: true`. SIR overall
confidence is 0.64 (below the 0.65 auto-proceed threshold). Key reasons, all documented in
`sir.json`: two paper-internal inconsistencies (an undocumented 6th baseline claim; a DSLOB
feature-count discrepancy), several core hyperparameters never given numeric values, and the
paper's own Appendix A.4.1 explicitly conceding its PDE no-arbitrage regularizer is
theoretically incomplete. None of this blocks generating a faithful code scaffold — it means
the scaffold necessarily contains more `# ASSUMED` config values than a typical paper, all
clearly flagged.

## Framework
**PyTorch** — the paper's description (Euler-Maruyama SDE backprop via reparametrisation,
mixed-precision training, Hugging Face for Chronos-2, cvxpylayers-style differentiable QP)
is unambiguously a PyTorch stack. CUDA required in practice (6 models × 4 datasets × 5 seeds,
Jane Street alone ~12M rows).

## Module Hierarchy (14 files)
- `data/preprocessing.py` — 4 dataset-specific pipelines (Jane Street, Optiver, Time-IMM, and
  a from-scratch DSLOB synthetic-seed generator, since the paper's actual DSLOB seed data is
  unnamed and unavailable)
- `models/laplace_neural_operator.py` — Module 1 (continuous-time encoder)
- `models/neural_sde.py` — Module 2 (drift/diffusion nets + Euler-Maruyama simulate())
- `models/physics_losses.py` — Feynman-Kac PDE residual + market-price-of-risk penalty
- `models/symbolic_bottleneck.py` — Module 3, plus an `export_formula()` method that fills a
  gap the paper itself leaves open (no example distilled formula is ever shown)
- `models/conformal_allocation.py` — Module 4 (adaptive conformal + optional Kelly QP)
- `models/artemis.py` — top-level assembly + two-phase training forward passes
- `models/baselines/{lstm,transformer,ns_transformer,informer,chronos2_wrapper}.py` — the 5
  actually-reported baselines (XGBoost excluded — see Risk Assessment)
- `evaluation/{metrics,statistical_tests}.py` — RMSE/RankIC/DirAcc/Weighted-R2 + Wilcoxon test

## Entrypoints
1. `scripts/prepare_data.py` — per-dataset preprocessing
2. `scripts/train_model.py` — train ARTEMIS or any one baseline
3. `scripts/run_ablation.py` — the 7-variant DSLOB ablation (Table 3)
4. `scripts/evaluate.py` — metrics + Wilcoxon significance
5. `scripts/plot_figures.py` — reproduce Figures 2–6

## Key Assumptions Flagged for Review (SIR confidence < 0.6)
| Assumption | Confidence | Config knob |
|---|---|---|
| Latent dimension d_z = 64 | 0.35 | `model.artemis.d_z` |
| SDE step count M = 100 | 0.3 | `model.artemis.n_sde_steps` |
| Loss weights lambda_1..4 | 0.3 | `model.artemis.loss_weights.*` |
| MPR Sharpe threshold kappa = 2.0 | 0.4 | `model.artemis.mpr_kappa` |
| Symbolic basis library composition | 0.3 | `model.symbolic.basis_lags` |
| PDE regularizer's theoretical completeness | 0.5 | n/a — documented caveat, not a config knob |

All are exposed as explicit config parameters, never hardcoded, per Stage 3 policy for SIR
confidence < 0.6.

## Risk Assessment Highlights
- **High** (×2): (1) essentially every ARTEMIS-specific hyperparameter is unreported by the
  paper; (2) DSLOB — the dataset behind the paper's headline result and entire ablation
  study — is synthetic, non-public, and generated from an unnamed "real" seed dataset that
  cannot be obtained.
- **Medium** (×3): the PDE regularizer's self-admitted theoretical gap; the XGBoost/
  feature-count paper-internal inconsistencies; Optiver's asynchronous-stream-fusion
  preprocessing is bug-prone and hard to validate without ground truth.
- **Low** (×2): Kelly-QP layer's actual usage in reported results is unconfirmed; symbolic
  basis library is unenumerated (mitigated with a documented default + `export_formula()`).

Full machine-readable plan: `architecture_plan.json`.
