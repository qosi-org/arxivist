# Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage

**Paper:** Junchi Shen — [arXiv:2607.20168](https://arxiv.org/abs/2607.20168) (July 2026)
**ArXivist-generated reproduction** — `paper_id: arxiv_2607_20168`

A controlled horse race on Chinese A-shares: a quantum fidelity kernel, a projected
quantum kernel, and a classical RBF control share identical training subsamples, solver,
and tuning budgets, so only the kernel is exchanged. On a point-in-time universe over 170
walk-forward windows, **no quantum advantage exists**. On a survivorship-screened universe
over 60 windows, the *same* kernel appears dominant — the paper's contribution is
dissecting how that reversal is manufactured.

## ⚠️ Before you start: the real dataset is not available

This paper's China A-share data is **proprietary** — the paper states materials are
"available from the author on request," with no public dataset URL or DOI. This repo
implements the full methodology and is smoke-tested end-to-end against a **synthetic**
data generator (`data/synthetic.py`), but **cannot reproduce the paper's actual numbers**
without you sourcing equivalent real data. See `data/README_data.md`.

## ⚠️ The highest-risk implementation choice: the `H_q` gate

Eq. (1)'s encoding circuit includes a term `H_q` that **the paper never defines**. We
implement it as Hadamard-conjugated `RZ` (the standard IQP-style convention), but this is
our interpretation, not a stated fact — confidence 0.6 in the SIR. It's a one-line,
config-driven swap (`quantum.hq_gate_interpretation` in `configs/config.yaml`) if you have
reason to believe otherwise. See `src/qkernel_finance/quantum/feature_map.py`'s docstring.

## Quick start

```bash
git clone <this-repo-url> && cd arxiv_2607_20168
pip install -r requirements.txt
pip install -e .
python run_study.py --config configs/config.yaml --study main --debug   # synthetic smoke test
python compare_models.py --results-dir results/
python run_bandwidth_geometry_diagnostic.py --debug
```

All three entrypoints have been **actually executed**, not just syntax-checked, against
the bundled synthetic data — see `comparison/verification_log.md` for the full record,
including one real bug (a pandas-version-dependent column-dropping issue in
`groupby().apply()`) found and fixed during that testing.

## Installation

**pip:** `pip install -r requirements.txt` (or `requirements-dev.txt`), then `pip install -e .`
**conda:** `conda env create -f environment.yaml && conda activate qkernel_finance`
**Docker:** `docker compose -f docker/docker-compose.yml up study` (or `notebook` for JupyterLab on :8888)

## Repository structure

```
src/qkernel_finance/
├── data/          characteristics.py, universes.py, interactions.py, synthetic.py
├── features/       top8_selector.py, bandwidth.py
├── quantum/         feature_map.py (Eq.1 circuit -- see H_q warning above), kernels.py
├── classical/        rbf_kernel.py, baselines.py (ridge/xgboost/mlp/nn3/poly2ridge)
├── models/           krr.py (closed-form solve + Nystrom extension)
├── evaluation/        walkforward.py, metrics.py, geometry.py (Eq.2)
└── utils/             config.py
run_study.py · compare_models.py · run_bandwidth_geometry_diagnostic.py
configs/config.yaml · data/README_data.md · notebooks/reproduce_arxiv_2607_20168.ipynb
docker/Dockerfile · docker/docker-compose.yml
```

## Expected results (Table 3, Sec 5 of the paper)

| Model | Mean IC | ICIR | t-stat | Hit rate |
|---|---|---|---|---|
| Poly(2) ridge (top-8) | 0.0499 | 0.272 | 3.55 | 0.629 |
| Ridge (top-8) | 0.0494 | 0.247 | 3.21 | 0.671 |
| QKRR fidelity | 0.0254 | 0.171 | 2.23 | 0.582 |
| KRR-RBF control | 0.0208 | 0.161 | 2.10 | 0.582 |
| QKRR projected | 0.0168 | 0.134 | 1.74 | 0.571 |

Fidelity QKRR vs. RBF control: ΔIC = +0.005 (p=0.42) — statistically indistinguishable.
After Holm family-wise correction, **no pairwise difference among 11 models is significant**
(smallest adjusted p=0.43).

## Implementation assumptions (things the paper does not fully specify)

See `sir-registry/arxiv_2607_20168/sir.json → implementation_assumptions[]` and
`→ ambiguities[]` for the complete, confidence-scored list.

| Assumption | Value used | SIR confidence |
|---|---|---|
| `H_q` gate in Eq. (1) | Hadamard-conjugated RZ | **0.6 — highest-risk item in this repo** |
| Projected kernel γ tuning | per-window grid search (assumed) | 0.5 |
| Nyström extension formula | standard Williams & Seeger (2001) | 0.55 |
| Subsample stratification scheme | proportional by date | 0.55 |
| NN3/MLP optimizer | Adam, lr=1e-3 | 0.5 |

## Reproducibility notes / known deviations

- **No real dataset**: this is the single largest barrier to a faithful reproduction (see
  `data/README_data.md`). Everything here is validated against synthetic data only.
- **`H_q` gate ambiguity**: see warning above.
- **Widened bandwidth grid** (Sec 8): the paper states "8 points spanning λ∈[0.01,1.6]"
  and names 5 of them explicitly (0.01, 0.2, 0.4, 0.8, 1.6); the remaining 3 interior
  points in `configs/config.yaml`'s `bandwidth_grid_widened` are our interpolation.
- **γ (projected kernel bandwidth)** has no stated tuning procedure in the paper, unlike
  λ and α; assumed tuned the same way.
- Real full-scale reproduction (170 + 60 windows, N≈37,800 Nyström extension, 8-qubit
  exact statevector simulation per window) is computationally nontrivial even without
  hardware — budget accordingly once real data is available.

## Citation

```bibtex
@article{shen2026quantum,
  title   = {Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage},
  author  = {Shen, Junchi},
  year    = {2026},
  eprint  = {2607.20168},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.PR}
}
```

## Generated by ArXivist

SIR and architecture plan live in `sir-registry/arxiv_2607_20168/` (`sir.json`,
`architecture_plan.json`, `architecture_plan_summary.md`). See `comparison/` for the
Stage 6 report — run against synthetic smoke-test data only (real data unavailable), so
treat it as a pipeline validation, not a verdict on reproducibility.
