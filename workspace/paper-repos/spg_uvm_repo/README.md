# SPG-UVM: Stochastic Policy Gradient Methods for the Uncertain Volatility Model

> **Paper:** Abbas-Turki L.A., Chassagneux J.-F., Lemor J.-P., Loeper G., Sananes S. (2026).  
> *Stochastic Policy Gradient Methods in the Uncertain Volatility Model.* arXiv:2605.06670.

---

## What This Paper Does

This paper addresses **robust option pricing under the Uncertain Volatility Model (UVM)**: given
that the true volatility (and potentially correlation) of a multi-asset portfolio is unknown but
bounded, what is the worst-case fair price of an option?

This reduces to a **stochastic control problem** — choosing the volatility matrix at each time
step to maximize the option payoff. The authors solve it with a **backward actor-critic
Stochastic Policy Gradient (SPG)** scheme, combining:

- **Discrete Dynamic Programming** (backward induction, N steps)
- **PPO** (Proximal Policy Optimization) for stable policy updates
- **Shallow neural networks** (one hidden layer, 32 units) for actor and critic
- **C-vine parameterization** of correlation matrices, enforcing positive semidefiniteness by construction
- **Two policy classes:** squashed Gaussian (continuous, for uncertain correlations) and factorized Bernoulli (bang-bang, for discrete volatility choices)

The method scales to d=80 assets in the fixed-correlation setting and d=5 with uncertain correlations, comparing favorably with tree-based (GTU) and neural network (NNU) benchmarks.

---

## Quick Start

```bash
# 1. Clone and install
git clone <this-repo>
cd spg_uvm
pip install -e .

# 2. Train on 2D geometric outperformer (default)
python train.py --config configs/default.yaml

# 3. Debug mode (fast sanity check)
python train.py --config configs/default.yaml --debug

# 4. Bang-bang policy, best-of butterfly (Table 2 of paper)
python train.py --config configs/best_of_butterfly.yaml --policy bangbang

# 5. High-dimensional geo call spread (d=10, Table 3)
python train.py --config configs/default.yaml --payoff geo_call_spread --d 10
```

---

## Installation

**Pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

**Conda:**
```bash
conda env create -f environment.yaml
conda activate spg-uvm
pip install -e .
```

**Docker:**
```bash
cd docker
docker-compose up train
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 11.8+ (GPU required for performance)

---

## Repository Structure

```
├── src/spg_uvm/
│   ├── models/
│   │   ├── networks.py     # ActorNetwork, CriticNetwork (ELU MLP, 32 units)
│   │   ├── policy.py       # ContinuousPolicy, BangBangPolicy
│   │   ├── vine.py         # CVineCorrelation (C-vine PSD parameterization)
│   │   └── dynamics.py     # LogEulerScheme (multi-asset GBM step)
│   ├── payoffs.py          # Option payoffs (5 types)
│   ├── training/
│   │   ├── trainer.py      # SPGUVMTrainer (Algorithm 1)
│   │   ├── losses.py       # PPOLoss, CriticLoss, CorrelationPenalty
│   │   ├── sampling.py     # StateSampler (log-normal, antithetic)
│   │   └── annealing.py    # SigmoidAnnealer (temperature / LR schedule)
│   └── utils/
│       ├── config.py       # UVMConfig, set_seed
│       └── metrics.py      # PriceEstimator (actor price + CI)
├── configs/
│   ├── default.yaml        # 2D geo-outperformer (Tables 1-2)
│   └── best_of_butterfly.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── train.py                # Main training entrypoint
└── evaluate.py             # Evaluation entrypoint
```

---

## Training Details

All hyperparameters are from Section 4.1.3 of the paper:

| Hyperparameter | Value | Confidence |
|---|---|---|
| Hidden units | 32 | 0.95 |
| MC paths M | 32,768 (2^15) | 0.97 |
| Minibatch size | 1,024 (2^10) | 0.97 |
| Epochs (first step) | 500 | 0.95 |
| Epochs (subsequent) | 10 | 0.95 |
| Learning rate | 5e-3 → 1e-4 (sigmoid) | 0.91 |
| PPO ε | 0.2 | 0.97 |
| Correlation penalty β | 10 | 0.97 |
| Huber threshold δ | 0.05 | 0.97 |
| Temperature λ | 1.0 → 0.01 (sigmoid) | 0.85 |
| Entropy coeff γ | 0.01 → 0.0 (sigmoid) | 0.82 |
| Antithetic variates | Yes | 0.97 |
| Transfer learning | Yes | 0.97 |

---

## Expected Results

From the paper (Section 4.2):

### Geo-outperformer (Section 4.2.1, d=2, σ∈[0.1,0.2], ρ∈[-0.5,0.5], T=1, r=0)

| Method | Price | CI |
|---|---|---|
| **SPG-UVM (ours)** | **13.75** | **±0.06** |
| GTU (Goudenège et al.) | 13.75 | — |
| NNU (Goudenège et al.) | 13.75 | — |

### Geo call spread (Section 4.2.3, d=80, fixed ρ=0)

| Method | Price | CI |
|---|---|---|
| **SPG-UVM (ours)** | **9.51** | **±0.00** |
| Reference (analytic) | 9.51 | — |

---

## Reproducibility Notes

The following components rely on **assumptions** where the paper does not specify:

1. **Sigmoid annealing formula** (confidence 0.72): Not explicitly given; we parameterize
   as `v(l) = v_final + (v_init - v_final) * sigmoid(k * (mid - l))`.
   Controlled by `exploration.sigmoid_steepness` in the config.

2. **Adam β₁, β₂** (confidence 0.85): Paper specifies Adam but not β₁/β₂.
   We use PyTorch defaults (0.9, 0.999). Expose via optimizer hyperparams if needed.

3. **C-vine for d ≥ 4** (confidence 0.88): Recursive formula from Joe (2006) / JKL (2009);
   d=3 closed form verified against paper. Unit tests for PSD + unit diagonal recommended.

4. **Critic network per step**: The paper's Algorithm 1 trains a fresh critic at each step n.
   This implementation does not serialize critic networks between steps; actor price
   is computed from the full forward simulation under the trained actor networks.

---

## Citation

```bibtex
@misc{abbasturki2026spguvm,
  title   = {Stochastic Policy Gradient Methods in the Uncertain Volatility Model},
  author  = {Abbas-Turki, Lokman A. and Chassagneux, Jean-Fran{\c{c}}ois and
             Lemor, Jean-Philippe and Loeper, Gr{\'e}goire and Sananes, Simon},
  year    = {2026},
  eprint  = {2605.06670},
  archivePrefix = {arXiv},
  primaryClass = {q-fin.CP}
}
```

---

*Generated by [ArXivist](https://github.com/anthropic/arxivist) from arXiv:2605.06670.*
