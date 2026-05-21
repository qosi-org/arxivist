# Architecture Plan — SPG-UVM
**Paper:** Stochastic Policy Gradient Methods in the Uncertain Volatility Model  
**Paper ID:** arxiv_2604_06670  
**Framework:** PyTorch 2.0+ / CUDA 11.8+

---

## Overview

This repository implements a **backward actor-critic stochastic policy gradient algorithm** for robust option pricing under the Uncertain Volatility Model (UVM). The algorithm runs N time steps backward in time; at each step it alternately trains a **Critic** (value network) and an **Actor** (policy network) using PPO.

Two policy classes are supported:
- **Continuous** (squashed Gaussian via C-vine): for uncertain correlations, d ≥ 2
- **Bang-bang** (factorized Bernoulli): for discrete volatility choices, d ≥ 1

---

## Module Layout

```
src/spg_uvm/
├── __init__.py
├── models/
│   ├── networks.py        — ActorNetwork, CriticNetwork (ELU MLP, 1 hidden layer, 32 units)
│   ├── policy.py          — ContinuousPolicy, BangBangPolicy
│   ├── vine.py            — CVineCorrelation (partial correlations → PSD matrix)
│   └── dynamics.py        — LogEulerScheme (multi-asset GBM step + path simulation)
├── payoffs.py             — GeoOutperformer, OutperformerSpread, BestOfButterfly, GeoCallSpread, CallSharpe
├── training/
│   ├── trainer.py         — SPGUVMTrainer (Algorithm 1 from paper)
│   ├── losses.py          — PPOLoss, CriticLoss, CorrelationPenalty (Huber)
│   ├── sampling.py        — StateSampler (log-normal mu_n, antithetic variates)
│   └── annealing.py       — SigmoidAnnealer (temperature & entropy schedule)
└── utils/
    ├── config.py          — UVMConfig dataclass + YAML I/O
    └── metrics.py         — PriceEstimator (actor price + 95% CI, relative error)
```

---

## Key Data Flows

### Continuous Policy Forward Pass
```
x [B,d]
  → LayerNorm → ELU(Linear, 32) → Linear → m_theta [B, d*(d+1)//2]
  → z ~ N(m_theta, λI)
  → tanh(z) [B, d*(d+1)//2]
     ├── z_sigma → affine rescale → sigma [B,d]  (volatility in bounds)
     └── z_rho  → C-vine → L [B,d,d] (Cholesky), rho [B,d,d] (correlation)
  → a = diag(sigma) @ L [B,d,d]  (used in log-Euler)
```

### Log-Euler Step
```
(x [B,d], a [B,d,d], xi [B,d]) → X_next = x * exp((r - ½diag(aaᵀ))*dt + a*√dt*xi)
```

### PPO Update (per epoch)
```
Frozen policy samples → (x, a, xi, z, latent_m_old) stored
New actor → m_new(x) → Gaussian likelihood ratio → clip → min with advantage → PPO loss
+ correlation penalty on mean action → total actor loss → Adam step
```

---

## Configuration

See `configs/default.yaml` for all hyperparameters. Key values:

| Param | Value | Confidence |
|-------|-------|-----------|
| Hidden units | 32 | 0.95 |
| M (MC samples) | 32768 | 0.97 |
| Minibatch | 1024 | 0.97 |
| E_first | 500 | 0.95 |
| E_subsequent | 10 | 0.95 |
| lr: 5e-3 → 1e-4 | sigmoid | 0.91 |
| PPO ε | 0.2 | 0.97 |
| β (correlation penalty) | 10 | 0.97 |
| δ (Huber threshold) | 0.05 | 0.97 |
| λ_init / λ_final | 1.0 / 0.01 | 0.85 |
| γ_init / γ_final | 0.01 / 0.0 | 0.82 |
| Adam β₁/β₂ | 0.9/0.999 | **ASSUMED** (0.85) |

---

## Entrypoints

```bash
python train.py --config configs/default.yaml --policy continuous --d 2 \
    --payoff geo_outperformer --device cuda

python evaluate.py --checkpoint checkpoints/step_0.pt --n_paths 524288 \
    --reference_price 13.75
```

---

## Implementation Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| C-vine for d≥4 (recursive formula) | Medium | Validate vs d=3 closed form; unit tests |
| Sigmoid annealing exact formula | Low | Configurable params; Figure 1 as reference |
| Call Sharpe augmented state | Medium | StatefulPayoff abstraction |
| Adam hyperparams not specified | Low | Use PyTorch defaults; expose in config |
| Near-singular L when \|y\|→1 | Low | Clamp tanh to (-1+ε, 1-ε) |
