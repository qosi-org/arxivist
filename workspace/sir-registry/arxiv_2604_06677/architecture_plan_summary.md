# Architecture Plan: Stochastic-Clock Barrier Options Library
## Paper: arxiv_2604_06677

### Framework Selection
**Language**: Python 3.10+  
**Framework**: NumPy / SciPy (NOT PyTorch — this is a mathematical finance library, not DL)  
**Key dependencies**: numpy, scipy, matplotlib, numba (optional JIT for inner loops)

**Reasoning**: The paper presents closed-form / semi-analytic pricing formulas relying on:
- 1D numerical integration (scipy.integrate.quad)
- ODE solvers for Riccati systems (scipy.integrate.solve_ivp)
- Monte Carlo with NumPy vectorization
- No tensor backpropagation needed

---

### Module Hierarchy

```
paper-repos/arxiv_2604_06677/
├── src/
│   └── stochastic_clock_barriers/
│       ├── __init__.py
│       ├── models/                         ← Clock Laplace transform families
│       │   ├── __init__.py
│       │   ├── base_clock.py               ← Abstract BaseClock class
│       │   ├── cir_clock.py                ← Integrated CIR (Section 4.1)
│       │   ├── sq_ou_clock.py              ← Squared OU (Section 4.2)
│       │   ├── markov_switching_clock.py   ← Markov switching (Section 4.3)
│       │   └── affine_jump_clock.py        ← Affine jump-diffusion (Section 4.4)
│       ├── pricing/                        ← Core pricing engines
│       │   ├── __init__.py
│       │   ├── single_barrier.py           ← UOP / DOC (Theorems 2.1, 2.2)
│       │   ├── double_barrier.py           ← DKO (Theorem 3.1)
│       │   ├── vanilla.py                  ← Vanilla via FFT/COS (Section 7.2)
│       │   ├── leverage_expansion.py       ← rho-expansion + Duhamel MC (Section 5)
│       │   └── pade_accelerator.py         ← Padé resummation (Section 5.8)
│       ├── calibration/                    ← Calibration workflow (Section 7)
│       │   ├── __init__.py
│       │   └── calibrator.py              ← 4-stage calibration engine
│       ├── monte_carlo/                    ← Reference MC pricer (Section 6.1)
│       │   ├── __init__.py
│       │   └── mc_pricer.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py                   ← Config loading, seed utility
│           └── quadrature.py              ← Shared quadrature helpers
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── reproduce_arxiv_2604_06677.ipynb
│   └── explore_arxiv_2604_06677.ipynb
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   └── README_data.md
├── scripts/
│   ├── reproduce_tables.py                ← Reproduces Tables 6.2–6.12
│   └── calibration_demo.py
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml
├── setup.py
└── README.md
```

---

### Module Specifications

#### `models/base_clock.py` — Abstract BaseClock
- `class BaseClock(ABC)`: abstract base for all clock families
  - `laplace(self, lam: np.ndarray, T: float) -> np.ndarray`: returns Φ_T(λ)
  - `conditional_laplace(self, lam: np.ndarray, t: float, T: float, y0: float) -> np.ndarray`: Φ_{t,T}(λ;y)
  - `d_laplace_dy(self, lam: np.ndarray, T: float) -> np.ndarray`: ∂_y Φ_{t,T}

#### `models/cir_clock.py` — IntegratedCIRClock
- `class IntegratedCIRClock(BaseClock)`: κ, θ, ξ, v0
  - Riccati ODE: dB/dt = κB + λ - (ξ²/2)B², dA/dt = κθB
  - `laplace(lam, T)`: exp(-A(T;λ) - B(T;λ)*v0)
  - `d_laplace_dv0(lam, T)`: = -B(T;λ) * Phi_T(lam)  [Eq 6.8]
  - Feller condition check on __init__

#### `models/sq_ou_clock.py` — SquaredOUClock  
- `class SquaredOUClock(BaseClock)`: α, σ, Y0
  - Riccati: dB/dt = 2σ²B² - 2αB + λ, dA/dt = σ²B
  - `laplace(lam, T)`: exp(-A(T;λ) - B(T;λ)*Y0²)
  - `d_laplace_dY0(lam, T)`: = -2*B(T;λ)*Y0 * Phi_T(lam)

#### `pricing/single_barrier.py` — SingleBarrierPricer
- `class SingleBarrierPricer`:
  - `price_uop(F0, K, H, T, r, q, clock) -> float`: Theorem 2.1, Eq 2.33
  - `price_doc(F0, K, L, T, r, q, clock) -> float`: Theorem 2.2, Eq 2.40
  - `_joint_survival_cdf(x, h, beta, clock, T) -> float`: Prop 2.2, Eq 2.30
  - Both use `scipy.integrate.quad` on compactified [0,1) domain
  - Phi_T cached on quadrature nodes

#### `pricing/double_barrier.py` — DoubleBarrierPricer
- `class DoubleBarrierPricer`:
  - `price_dko_call(F0, K, L, H, T, r, q, clock, N_terms=100) -> float`: Theorem 3.1, Eq 3.9
  - `price_dko_put(F0, K, L, H, T, r, q, clock, N_terms=100) -> float`: Eq 3.20
  - `_compute_An(n, k, l, h, beta) -> float`: Eq 3.5-3.8
  - `_compute_eigenfreqs(N, a) -> np.ndarray`: ω_n = nπ/a
  - `_compute_lambda_grid(N, a, beta) -> np.ndarray`: λ_n = (ω_n² + β²)/2

#### `pricing/leverage_expansion.py` — LeverageExpansion
- `class LeverageExpansion`:
  - `compute_coefficients(n_order, clock, contract_params, n_paths, n_steps) -> list[float]`
  - `price_rho(rho, coefficients, method='pade') -> float`
  - `_duhamel_mc_coefficient(n, prev_u_fn, clock, n_paths) -> float`: Eq 5.17
  - `_L1_operator(u_fn, x, y, clock) -> float`: mixed derivative a(y)*sqrt(v(y))*d²u/dxdy
  - `error_diagnostic(coefficients, rho, N) -> float`: Eq 5.25

#### `pricing/pade_accelerator.py` — PadeAccelerator
- `class PadeAccelerator`:
  - `compute_pade(coefficients, L, M) -> tuple[np.poly1d, np.poly1d]`: [L/M] Pade
  - `evaluate(rho) -> float`: evaluate Pade at rho
  - `pole_distance(rho_target) -> float`: distance to nearest real pole
  - `is_safe(rho_target, threshold=0.1) -> bool`: pole safety check

#### `monte_carlo/mc_pricer.py` — MonteCarloPricer
- `class MonteCarloPricer`:
  - `price(contract_params, clock_params, rho, n_paths, n_steps, seed) -> tuple[float, float]`
  - `_simulate_cir_path(params, n_steps, dt, Z) -> np.ndarray`: full truncation Euler (Eq 6.2)
  - `_simulate_sq_ou_path(params, n_steps, dt, Z) -> np.ndarray`: exact OU step (Eq 6.5)
  - `_apply_bridge_correction(x_path, barrier, dt) -> np.ndarray`: Broadie et al. 1997

---

### Configuration Schema (configs/config.yaml)

```yaml
contract:
  S0: 100.0
  K: 100.0
  H: 130.0       # upper barrier
  L: 70.0        # lower barrier
  T: 1.0
  r: 0.03
  q: 0.0

clock:
  family: "cir"  # options: cir, sq_ou, markov_switching, affine_jump
  cir:
    v0: 0.18     # initial variance
    kappa: 0.6   # mean reversion speed
    theta: 0.20  # long-run variance
    xi: 0.4      # vol-of-vol
  sq_ou:
    Y0: 0.4243
    alpha: 0.6
    sigma: 0.490
  rho: 0.0       # return-volatility correlation

pricing:
  single_barrier:
    n_quad: 200          # quadrature nodes
    quad_tol: 1e-10
    compactification: "tanh"  # ASSUMED: tanh or t/(1-t)
  double_barrier:
    n_series_terms: 100  # truncation of sine series
  leverage:
    expansion_order: 5
    pade_type: "[3/2]"
    n_paths_duhamel: 100000

monte_carlo:
  n_paths: 1000000
  n_steps_per_year: 2080
  bridge_correction: true
  seed: 42

calibration:
  optimizer: "L-BFGS-B"
  vanilla_weight: 1.0
  barrier_weight: 0.5
  regularization_lambda: 0.001
```

---

### Entrypoints

**`scripts/reproduce_tables.py`**
```
--config        path to config YAML [default: configs/config.yaml]
--table         which table to reproduce [6.2|6.3|6.4|6.5|6.6|6.7|6.10|6.11|all]
--output-dir    directory for output CSV/JSON
--seed          random seed [default: 42]
```

**`scripts/calibration_demo.py`**
```
--config        path to config YAML
--clock-family  cir|sq_ou
--rho           target correlation [default: -0.7]
```

---

### Dependencies

```
# requirements.txt
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
matplotlib>=3.7.0
numba>=0.57.0
pyyaml>=6.0
tqdm>=4.65.0
```

```
# requirements-dev.txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.0.285
jupyter>=1.0.0
ipywidgets>=8.0.0
```

---

### Risk Assessment

| Risk | Severity | Description | Mitigation |
|------|----------|-------------|-----------|
| Quadrature oscillation | Medium | Integrands sin(u·Δ)·Φ_T(u) oscillate for large Δ | Compactify + cache Phi_T; increase n_quad if needed |
| Riccati blow-up | Medium | CIR Riccati can explode if Feller violated (2κθ < ξ²) | Enforce Feller condition in __init__; use stiff solver |
| Pade pole near real axis | Medium | [L/M] Pade can have poles near [-0.9,0.9] | pole_distance check + fallback to Taylor |
| Duhamel MC variance | Medium | Estimator for u_n has high variance near barrier | Use antithetic variates + control variates |
| Expansion radius | Low | Taylor expansion unreliable for |ρ|>0.5 | Use Pade resummation; user warned |
