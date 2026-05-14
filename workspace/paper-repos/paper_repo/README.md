# volsig — Signature-Based Implied Volatility Calibration

> **Paper**: *Volatility Modeling in Markovian and Rough Regimes: Signature Methods and Analytical Expansions*
> **Authors**: Elisa Alòs, Òscar Burés, Rafael de Santiago, Josep Vives
> **arXiv**: [2507.23392v4](https://arxiv.org/abs/2507.23392) · *q-fin.MF* · May 2026

---

## What This Paper Does

This paper studies two complementary approaches to calibrating implied volatility surfaces from option prices:

1. **Analytical expansions** — Second-order asymptotic formulas for the Heston model (revisiting Alòs et al. 2015) and a *new* VIX-based closed-form calibration for the rough Bergomi model, both giving highly accurate results when the model is correctly specified.

2. **Signature-based models** — Volatility is expressed as a linear functional of the truncated path signature of a primary noise process (Heston variance SDE or fractional Brownian motion). This data-driven approach makes no parametric assumptions about volatility dynamics and adapts naturally to non-Markovian (rough) settings.

Key finding: in the Heston (Markovian) setting both approaches reach ~10⁻⁴ implied volatility errors; in the rough Bergomi (non-Markovian) setting signatures match or slightly outperform the analytical baseline, reflecting their ability to capture temporal dependencies.

---

## Quick Start

```bash
# Clone and install
git clone <repo_url> && cd volsig
pip install -e .

# Run the Heston uncorrelated experiment (Section 5.1)
python train.py --config configs/config.yaml --experiment heston_uncorr --debug

# Full run (nMC=800k, ~45-90 min with GPU)
python train.py --config configs/config.yaml --experiment heston_uncorr

# Evaluate a saved ℓ* vector
python evaluate.py --config configs/config.yaml \
                   --l_star results/heston_uncorr/l_star.npy \
                   --experiment heston_uncorr --plot

# Analytical Heston ASV calibration (Section 2.1)
python calibrate_heston_asv.py --config configs/config.yaml --experiment heston_uncorr

# VIX rough Bergomi calibration (Section 2.2)
python calibrate_rbergomi_vix.py --config configs/config.yaml
```

---

## Installation

### Option A — pip (CPU or CUDA)
```bash
pip install -e .
# For GPU signature computation (recommended):
pip install signatory==1.2.7.1.9.0   # requires matching PyTorch + CUDA
```

### Option B — conda
```bash
conda create -n volsig python=3.10
conda activate volsig
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
pip install signatory==1.2.7.1.9.0
```

### Option C — Docker (GPU)
```bash
cd docker
docker compose build
docker compose run train
docker compose up notebook   # starts Jupyter on port 8888
```

> **Note on `signatory`**: This library requires a specific PyTorch version match.
> If installation fails, remove it from `requirements.txt` — the code falls back to
> a pure NumPy Riemann-sum signature implementation (slower but correct).

---

## Experiments

| Script | Experiment | Paper Section | Expected Loss | Est. Runtime |
|--------|-----------|---------------|---------------|-------------|
| `train.py --experiment heston_uncorr` | Uncorrelated Heston | Section 5.1 | 1.05×10⁻⁴ | ~45 min (GPU, nMC=800k) |
| `train.py --experiment heston_corr`   | Correlated Heston ρ=-0.5 | Section 5.2 | 1.46×10⁻³ | ~60 min |
| `train.py --experiment rough_bergomi` | Rough Bergomi fBM primary | Section 6 | 3.5×10⁻⁴ | ~17-19 min |
| `calibrate_heston_asv.py`            | ASV analytical benchmark | Section 2.1 | — | <1 min |
| `calibrate_rbergomi_vix.py`          | VIX analytical benchmark | Section 2.2 | — | ~5 min |

All timings from the paper use a consumer desktop with 128 GB RAM and NVIDIA RTX 3080 Ti GPU.

---

## Expected Results

### Table 5.1 — Heston uncorrelated (ρ=0)
Both methods achieve errors in the range **10⁻⁵ to 10⁻⁴**.
Analytical ASV is slightly more accurate; SIG wins on 3/20 contracts.

### Table 5.2 — Heston correlated (ρ=-0.5)
Errors in **10⁻⁴ to 10⁻³**. Correlation partially captured at N=3.
Increasing to N=4 gives marginal improvement.

### Table 6.1 — Rough Bergomi
Both methods achieve **~10⁻⁴** errors. SIG outperforms VIX analytical on 7/20 contracts.
The non-Markovian structure of fBM is well captured by signatures.

### Calibrated ℓ* (from paper, for verification)

**Heston uncorrelated** (Section 5.1):
```
ℓ* = [0.201202133, 0.142660997, 1.08471290, -0.297312378, -0.0293435325,
      -0.0422317187, 9.25090162e-4, 0.293103687, -0.0143435573, -0.0134285652,
      -1.64737083e-3, -2.89883092e-3, -5.72798006e-4, -1.93045420e-3, -1.84406803e-4]
```
Note: ℓ[0]≈0.201 ≈ σ₀ and ℓ[2]≈1.085 confirms strong linear dependence on X_t (Heston structure).

**Rough Bergomi** (Section 6, shifted-exp primary):
```
ℓ* = [0.17273586, -0.29578964, -0.08071348, 0.40101573, -0.2974647,
      0.31988953, 1.40158411, 0.15016936, -0.05769989, 0.00999173,
      0.25021442, 0.02998332, -0.00789562, 0.12012242, 0.27102252]
```

---

## Module Structure

```
src/volsig/
├── models/
│   ├── signature_vol.py    ← Core pipeline (precompute → calibrate)
│   ├── heston.py           ← Heston SDE + ASV expansion (Section 2.1)
│   ├── rough_bergomi.py    ← Rough Bergomi + VIX calibrator (Section 2.2)
│   └── primary_process.py ← Heston variance SDE & fBM variants (Section 6)
├── signatures/
│   └── compute.py          ← Time augmentation, truncated signatures,
│                              shuffle products, Q-matrix (Section 3–4)
├── pricing/
│   ├── black_scholes.py    ← BS formula, IV inversion, Vega
│   └── mc_pricer.py        ← Signature MC pricer (Proposition 4.2)
├── calibration/
│   └── optimizer.py        ← L-BFGS-B wrapper, loss, weights (Section 4.2)
└── utils/
    ├── config.py           ← YAML config loading + seed_everything()
    └── plotting.py         ← 3D IV surface plots, error tables
```

---

## Key Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Signature truncation N | 3 | Paper: Section 4.3 |
| nMC | 800,000 | Paper: Section 4.3 |
| Optimizer | L-BFGS-B | Paper: Section 4.3 |
| Tolerance | 1×10⁻⁸ | Paper: Section 4.3 |
| Maturities | [0.1, 0.6, 1.1, 1.6] | Paper: Section 5.1 |
| Strikes | [90, 95, 100, 105, 110] | Paper: Section 5.1 |
| Weights γᵢ | inverse Vega | Paper: Section 4.2 |
| Interpolation | linear | Paper: Section 4.3 |
| VIX window Δ | 30 trading days | Paper: Section 2.2 |

---

## Reproducibility Notes

### Known deviations from the paper

1. **Euler step count** (`T_steps_per_unit: 252`): The paper states "Euler scheme" but does not specify the number of time steps. We assume 252 (daily). Changing this affects accuracy vs speed. **Mark: ASSUMED.**

2. **Box constraint bounds** (`box_bounds: [-10.0, 10.0]`): The paper mentions "box constraints to accelerate convergence" but does not give the bounds. **Mark: ASSUMED.** If your ℓ* components hit the bounds, widen them in `config.yaml`.

3. **Initialisation** (`l0_init: zeros`): Paper does not specify. **Mark: ASSUMED.** A warm start from a nearby ℓ may speed convergence.

4. **X0 interpretation** (Heston primary): The paper states X₀=0.1 for the Heston variance SDE. This is treated as the initial *variance* (not volatility). Set `x0_is_variance: false` in config if you prefer it as initial vol.

5. **VIX ATMI**: The rough Bergomi calibration (Section 2.2, Step 2) requires ATM implied volatility of VIX *options*, which needs nested simulation. The current implementation uses a proxy (short-T ATM equity IV). To use proper VIX options, implement `RoughBergomiModel.vix_option_atmi()`.

6. **GPU acceleration**: The paper uses a GPU-vectorised version of Peter Foster's signature code. This repo includes a pure NumPy fallback. For full performance, install `signatory` and set `device: cuda`.

### Validation against paper

To verify your installation reproduces the paper's numbers:
```python
# After running heston_uncorr experiment:
import numpy as np
l_star = np.load("results/heston_uncorr/l_star.npy")
paper_l_star = np.array([0.201202133, 0.142660997, 1.08471290, ...])
print(np.allclose(l_star, paper_l_star, atol=1e-3))  # should be True
```

Small differences are expected due to randomness in MC simulation.

---

## Citation

```bibtex
@article{alos2026volatility,
  title   = {Volatility Modeling in {Markovian} and Rough Regimes:
             Signature Methods and Analytical Expansions},
  author  = {Al\`os, Elisa and Bur\'es, \`Oscar and de Santiago, Rafael and Vives, Josep},
  journal = {arXiv preprint arXiv:2507.23392},
  year    = {2026},
  url     = {https://arxiv.org/abs/2507.23392}
}
```
