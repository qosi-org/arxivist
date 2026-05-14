# Architecture Plan — `volsig`
### Paper: Volatility Modeling in Markovian and Rough Regimes
*Alòs, Burés, de Santiago, Vives — arXiv 2507.23392v4*

---

## Framework
**Python 3.10+** · **NumPy/SciPy** · **PyTorch (GPU tensors)** · **signatory** (truncated signatures on GPU) · **SciPy L-BFGS-B** (calibration optimizer)

No deep learning training loop — this is a Monte Carlo simulation + convex optimization pipeline. PyTorch is used purely for GPU-accelerated tensor ops in signature computation.

---

## Module Map

```
src/volsig/
├── models/
│   ├── signature_vol.py     ← SignatureVolModel  (core: sig features → MC pricing → calibration)
│   ├── heston.py            ← HestonModel + ASV expansion (Alòs et al. 2015)
│   ├── rough_bergomi.py     ← RoughBergomiModel + VIXCalibrator (new method, Sec 2.2)
│   └── primary_process.py  ← PrimaryProcessSimulator (Heston SDE, fBM variants)
├── signatures/
│   ├── compute.py           ← SignatureComputer + ShuffleProductComputer
│   └── (time augment, truncated sig, Q-matrix assembly)
├── pricing/
│   ├── black_scholes.py     ← BS formula, IV inversion, Vega
│   └── mc_pricer.py         ← SignatureMCPricer (Prop 4.2 formula)
├── calibration/
│   └── optimizer.py         ← SignatureCalibrator (L-BFGS-B wrapper, loss, weights)
└── utils/
    ├── config.py            ← YAML → dataclass config loader
    └── plotting.py          ← 3D IV surface plots, error tables
```

---

## Key Data Flow

### Phase 1 — Offline Precomputation (run once per experiment)
```
W[nMC,T]  B[nMC,T]
    ↓
Z = ρW + √(1-ρ²)B              [nMC, T_steps]
X = Euler(primary SDE, W)       [nMC, T_steps]
X_aug = (t, X)                  [nMC, T_steps, 2]
    ↓
sig_N3  = Sig(X_aug, N=3)       [nMC, T_steps, 15]   ← per time step (for stochastic integral)
sig_N7  = Sig(X_aug, N=7)|_T    [nMC, 255]            ← terminal only (for Q matrix)
    ↓
stoch_int = Σ_t sig_N3[:,t,:] dZ_t   [nMC, 15]       ← Euler-Itô sum
Q(T)      = assemble_via_shuffle(sig_N7)   [nMC,15,15]
U(T)      = cholesky(-Q(T))               [nMC,15,15]
```

### Phase 2 — Online Optimization (L-BFGS-B loop)
```
ℓ ∈ ℝ¹⁵  (optimized variable)
    ↓
S̃_T(j) = S0·exp(-‖U(j)ℓ‖² + ℓᵀ stoch_int(j))   per path j
C(K,T,ℓ) = mean_j [max(S̃_T(j) − Ke^{-rT}, 0)]
loss = Σᵢ γᵢ (C_mkt[i] − C[i])²
    ↑ gradient → L-BFGS-B → update ℓ
```

---

## Signature Dimensions (2D time-augmented path, N=3)

| Level | Terms | Cumulative |
|-------|-------|------------|
| 0     | 1     | 1          |
| 1     | 2     | 3          |
| 2     | 4     | 7          |
| 3     | 8     | 15         |
| …     | …     | …          |
| 7     | 128   | 255        |

Q matrix is 15×15, assembled from 255-dim extended signatures via precomputed shuffle lookup table.

---

## Configuration (`configs/config.yaml`)

All hyperparameters from the paper with confidence annotations. Parameters with `# ASSUMED` require verification:
- `T_steps_per_unit: 252` — ASSUMED (paper omits Euler step count)
- `l0_init: zeros` — ASSUMED (paper omits initialization)
- `box_bounds: [-10.0, 10.0]` — ASSUMED (paper says "box constraints" without specifying bounds)

---

## Entrypoints

| Script | Purpose |
|--------|---------|
| `train.py` | Full signature calibration pipeline |
| `evaluate.py` | IV surface comparison + error tables |
| `calibrate_heston_asv.py` | Analytical Heston expansion (Section 2.1) |
| `calibrate_rbergomi_vix.py` | VIX-based rough Bergomi calibration (Section 2.2) |

---

## Risks (summary)

| ID | Severity | Issue |
|----|----------|-------|
| R1 | **High** | Shuffle product table for Q matrix — combinatorial, must be precomputed |
| R2 | **High** | `signatory` version compatibility with PyTorch/CUDA |
| R3 | **Medium** | fBM simulation — naïve Volterra O(T²); use Hybrid scheme |
| R4 | **Medium** | X0 ambiguity in Heston primary SDE |
| R5 | **Medium** | Box constraint bounds unknown |
| R6 | **Low** | VIX option pricing requires nested simulation |
| R7 | **Low** | Cholesky instability — add ε·I regularisation |
