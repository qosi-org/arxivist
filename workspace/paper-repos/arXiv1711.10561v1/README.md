# Physics-Informed Neural Networks (PINNs)

> **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017)**  
> *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*  
> arXiv: [1711.10561](https://arxiv.org/abs/1711.10561)

---

## What This Paper Does

PINNs are neural networks trained to approximate solutions to partial
differential equations (PDEs). Instead of solving PDEs on a discretised
grid, the network directly parameterises the continuous solution field
`u(t, x)`. Physical laws are enforced by adding a *physics residual loss*
`MSE_f = mean(|f(t,x)|²)` at randomly sampled collocation points, where
`f = u_t + N[u]` is the PDE left-hand side. This acts as a strong
regulariser, enabling accurate predictions from as few as 100 labelled
boundary data points — far less than classical numerical methods require.

Two model families are implemented:
1. **Continuous-time PINN** — learns `u(t,x)` directly; physics
   residuals computed via `torch.autograd.grad`.
2. **Discrete-time PINN** — embeds an implicit Gauss-Legendre
   Runge-Kutta scheme (up to 500 stages) to march in time with
   exponentially small temporal error.

---

## Compute Requirements

> **TL;DR:** Continuous PINN runs fine on a modern GPU in 1-2 hours. Discrete PINN at q=500 requires a GPU - do not attempt on CPU.

| Experiment | Hardware | Estimated Time | Notes |
|---|---|---|---|
| Burgers (continuous) | RTX 3050 6GB | ~1-2 hrs | Tested and working |
| Schrodinger | RTX 3050 6GB | ~2-3 hrs | - |
| Burgers discrete (q=500) | RTX 3050 6GB | ~4-8 hrs | **CPU: 24+ hrs, not feasible** |
| Allen-Cahn discrete (q=100) | RTX 3050 6GB | ~2-4 hrs | **CPU: 12+ hrs, not feasible** |

### Why the discrete PINN is slow on CPU

The discrete-time PINN at q=500 stages runs **500 sequential autograd.grad calls** inside every single L-BFGS closure evaluation. On CPU this costs ~3-5 seconds per iteration; at 50,000 total iterations that is 40-70 hours. On a GPU the same closure runs in ~0.1-0.2 seconds.

**Known limitation (vmap TODO):** DiscretePINN.compute_N_stages() currently loops over q stages sequentially. Replacing this with torch.vmap would reduce it to a single batched backward pass, cutting runtime by 5-10x. This is marked as a TODO in the source code.

**Workaround for CPU-only machines:**
- Run continuous PINN experiments (Burgers, Schrodinger) - fully feasible on CPU, just slower.
- Use --debug to smoke-test the discrete PINN at q=4 (~2 min on CPU).
- Skip full discrete runs or use a GPU / cloud instance for those experiments.

---

## Quick Start (3 commands)

```bash
# 1. Install
pip install -e . -r requirements.txt

# 2. Train Burgers (continuous) — ~60s on a single GPU
python train.py --config configs/burgers_continuous.yaml --verbose

# 3. Evaluate
python evaluate.py --config configs/burgers_continuous.yaml \
                   --checkpoint results/burgers_continuous/best_model.pt \
                   --plot
```

---

## Installation

**Conda (recommended):**
```bash
conda env create -f environment.yaml
conda activate pinns
pip install -e .
```

**pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+, PyTorch ≥ 2.0, CUDA 11.8+ (CPU training works
but is significantly slower for L-BFGS on large collocation sets).

---

## Experiments

### Burgers Equation (Continuous)
```bash
python train.py --config configs/burgers_continuous.yaml --verbose
python evaluate.py --config configs/burgers_continuous.yaml \
    --checkpoint results/burgers_continuous/best_model.pt --plot
```

### Schrödinger Equation (Continuous, complex)
```bash
# Pre-generate reference solution (run once)
python data/generate_reference.py --pde schrodinger --output data/schrodinger_ref.npy

python train.py --config configs/schrodinger.yaml
python evaluate.py --config configs/schrodinger.yaml \
    --checkpoint results/schrodinger_continuous/best_model.pt \
    --reference data/schrodinger_ref.npy --plot
```

### Burgers Equation (Discrete RK, q=500)
```bash
python train.py --config configs/burgers_discrete.yaml
python evaluate.py --config configs/burgers_discrete.yaml \
    --checkpoint results/burgers_discrete/best_model.pt --plot
```

### Allen-Cahn Equation (Discrete RK, q=100)
```bash
python data/generate_reference.py --pde allen_cahn --output data/allen_cahn_ref.npy
python train.py --config configs/allen_cahn.yaml
python evaluate.py --config configs/allen_cahn.yaml \
    --checkpoint results/allen_cahn_discrete/best_model.pt \
    --reference data/allen_cahn_ref.npy --plot
```

---

## Expected Results

| Experiment | Architecture | Data | Paper L2 | Notes |
|---|---|---|---|---|
| Burgers (continuous) | 9L × 20N, tanh | Nu=100, Nf=10k | **6.7×10⁻⁴** | ~60s on Titan X |
| Schrödinger | 5L × 100N, tanh | N0=50, Nb=50, Nf=20k | **1.97×10⁻³** | Complex-valued output |
| Burgers (discrete) | 4L × 50N, tanh | Nn=250, q=500, Δt=0.8 | **8.2×10⁻⁴** | Single time step |
| Allen-Cahn | 4L × 200N, tanh | Nn=200, q=100, Δt=0.8 | **6.99×10⁻³** | Periodic BC, sharp layers |

---

## Debug / Dry-Run

```bash
# Quick sanity check (tiny dataset, 10 L-BFGS iters)
python train.py --config configs/burgers_continuous.yaml --debug

# Validate setup without training
python train.py --config configs/burgers_continuous.yaml --dry-run
```

---

## Docker

```bash
docker-compose -f docker/docker-compose.yml run train
# or Jupyter notebook:
docker-compose -f docker/docker-compose.yml up notebook
```

---

## Repository Structure

```
src/pinns/
├── models/
│   ├── mlp.py               Shared MLP backbone (Xavier init, tanh)
│   ├── continuous_pinn.py   Continuous-time PINN
│   └── discrete_pinn.py     Discrete-time PINN (implicit RK)
├── pde/
│   └── operators.py         Burgers / Schrödinger / Allen-Cahn operators
├── data/
│   ├── sampling.py          Latin Hypercube Sampling + PDEDataset
│   └── exact_solutions.py   Reference solution generators
├── training/
│   ├── losses.py            MSE/SSE loss functions
│   └── trainer.py           L-BFGS training loops
├── evaluation/
│   └── metrics.py           Relative L2 error
└── utils/
    ├── rk_tableau.py        Gauss-Legendre RK tableau (Butcher coefficients)
    ├── autograd_helpers.py  Centralised PDE derivative computation
    └── config.py            YAML config loader + seed utility
```

---

## Reproducibility Notes

### Implementation Assumptions

The following details were not specified in the paper and have been assumed:

| Component | Assumption | Confidence |
|---|---|---|
| Weight initialisation | Xavier uniform (`gain = 5/3` for tanh) | 0.80 |
| RK tableau | Gauss-Legendre nodes via `numpy.polynomial.legendre.leggauss` | 0.82 |
| Physics loss weight | Equal weighting `λ_f = 1.0` | ~1.0 (standard) |
| Schrödinger BC collocation | Random Nb time points on boundary | 0.75 |
| Reference solutions | Python scipy pseudo-spectral (paper uses MATLAB Chebfun) | — |

### Known Deviations from Paper

- **Framework**: Paper uses TensorFlow 1.x (`tf.gradients`). This implementation
  uses PyTorch 2.x (`torch.autograd.grad`) with equivalent semantics.
- **Reference solutions**: Paper uses Chebfun (MATLAB) for Schrödinger and
  Allen-Cahn references. This repo uses Python pseudo-spectral solvers.
  Differences of <0.1% in the reference are expected.
- **q=500 RK tableau**: At very high stage counts, the Vandermonde system for
  computing the `A` matrix is numerically delicate. We use `numpy.linalg.lstsq`
  (SVD-based) in float64 for robustness. Validate `GaussLegendreTableau` at
  small `q` before running q=500.

### If Results Differ from Paper

1. Check RK tableau: `python -c "from pinns.utils.rk_tableau import GaussLegendreTableau; t = GaussLegendreTableau(4); print(t)"`
2. Try fewer L-BFGS iterations first to confirm training is descending.
3. Allen-Cahn at q=100 with 4L×200N can be memory-intensive — reduce Nn if OOM.
4. For Burgers discrete at q=500: start with q=32 to validate, then increase.

---

## Citation

```bibtex
@article{raissi2017physics,
  title   = {Physics Informed Deep Learning (Part I): Data-driven Solutions
             of Nonlinear Partial Differential Equations},
  author  = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal = {arXiv preprint arXiv:1711.10561},
  year    = {2017}
}
```

Original code: https://github.com/maziarraissi/PINNs  
This reproduction: ArXivist pipeline (paper_pinns_raissi2017)
