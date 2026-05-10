# Q-Ising Architecture Plan Summary
**Paper**: Dynamic Treatment on Networks (arXiv:2605.06564)  
**Paper ID**: arxiv_2605_06564  
**Stage**: 3 — Architecture Planner

---

## Framework

- **Primary language**: Python 3.10+
- **Stage 1 (Ising inference)**: NumPy, SciPy, scikit-learn (EMVS); PyMC with NUTS (MCMC)
- **Stage 2 (Offline RL)**: PyTorch via d3rlpy
- **Network utilities**: python-igraph (edge-betweenness community detection)
- **GPU**: Not required — paper uses Apple M1 CPU

---

## Module Hierarchy

```
q_ising/
├── train.py                        ← Main CLI entrypoint
├── evaluate.py                     ← Policy comparison CLI
├── simulate_sis.py                 ← Panel data generation CLI
├── configs/
│   ├── sbm_default.yaml            ← SBM experiment config
│   └── village_default.yaml        ← Microfinance village config
├── data/
│   └── download_villages.py        ← Harvard Dataverse download script
├── docker/
│   └── Dockerfile
├── notebooks/
│   ├── 01_sbm_experiment.ipynb
│   └── 02_village_experiment.ipynb
└── src/q_ising/
    ├── data/
    │   ├── network.py              ← NetworkData: adjacency, bins, neighbors
    │   ├── panel.py                ← ObservationalPanel: trajectory storage
    │   └── sis_simulator.py        ← SISSimulator: churn → seed → spread
    ├── models/
    │   ├── ising.py                ← DynamicIsingModel (EMVS + MCMC)
    │   └── state_constructor.py    ← StateConstructor: node → bin-level states
    ├── training/
    │   ├── cql_trainer.py          ← CQLTrainer wrapping d3rlpy CQL
    │   └── ensemble_trainer.py     ← EnsembleTrainer: P agents, majority vote
    ├── evaluation/
    │   ├── metrics.py              ← PolicyEvaluator, mean adoption rate
    │   └── baselines.py            ← Random, Degree, LIR, DegreeBin, PlainDQN
    └── utils/
        ├── config.py               ← IsingConfig, CQLConfig, ExperimentConfig
        ├── community_detection.py  ← igraph edge-betweenness + merging
        └── sbm_generator.py        ← Stochastic block model generator
```

---

## Data Flow Summary

### Stage 1: Bayesian Dynamic Ising Inference

```
Panel D = {y_0, (a_t, y_t)_{t=1}^T}
Adjacency M, Features X, Bin assignment {B_1..B_K}
         │
         ▼
DynamicIsingModel.fit_emvs()   (or .fit_mcmc())
  ├── For each node i:
  │     eta_it = beta_0_k + beta_1_k*[a_t==i] + beta_2_k*y_{i,t-1}
  │              + beta_3_k*[a_t in N_i] + sum_j(gamma_{k,mj}*y_{j,t-1})
  │     P(y_it=1) = sigmoid(eta_it)
  │     Spike-and-slab prior on gamma; Normal(0,10) on betas
  └── Returns theta_hat (EMVS) or {theta^(p)} (MCMC draws)
         │
         ▼
StateConstructor.build_all_states()
  ├── For each t: set a_t = empty, compute eta_it_counterfactual
  │     l_hat_0_it = sigmoid(eta_it_counterfactual)     [N]
  │     l_bar_0_tk = mean over bin B_k                  [K]
  │     y_bar_{t-1,k} = mean(y_{t-1,i} for i in B_k)   [K]
  └── s_t = concat(l_bar_0_t, y_bar_{t-1})              [2K]
```

### Stage 2: Offline CQL

```
Transitions D_RL = {(s_t, b_t, r_t, s_{t+1})}
  where r_t = mean(y_t)   [scalar]
        b_t = bin index of treated node a_t
         │
         ▼
CQLTrainer.train()
  Q-network: Linear(2K→256)→BN→ReLU→Drop(0.3)
             →Linear(256→256)→BN→ReLU→Drop(0.3)
             →Linear(256→K)
  Loss = Bellman error + alpha*conservative_penalty (alpha=0.1)
  Discount psi=0.8, lr=3e-4, batch=64, max_steps=30000
         │
         ▼
Policy: pi(s) = argmax_k Q(s, k)
```

### Stage 3: Ensemble (MCMC variant)

```
P=20 MCMC draws {theta^(p)}
  → P state sequences {s_t^(p)}
  → P CQL agents trained independently
  → Majority vote: pi_ens(s) = argmax_k sum_p 1[pi^(p)(s) == k]
```

---

## Key Hyperparameters

| Param | Value | Confidence | Source |
|-------|-------|-----------|--------|
| Spike variance v0 | 0.01 | 0.95 | Paper §3.1 |
| Slab variance v1 | 10.0 | 0.95 | Paper §3.1 |
| Inclusion scale c | 1.0 | 0.95 | Paper §3.1 |
| Beta prior tau^2 | 10.0 | 0.95 | Paper §3.1 |
| CQL alpha | 0.1 | 0.95 | Appendix E.2 |
| Discount psi | 0.8 | 0.95 | Appendix E.2 |
| Hidden layers | [256,256] | 0.92 | Appendix E.2 |
| Learning rate | 3e-4 | 0.95 | Appendix E.2 |
| Batch size | 64 | 0.95 | Appendix E.2 |
| Max steps | 30000 | 0.95 | Appendix E.2 |
| Dropout | 0.3 | 0.95 | Appendix E.2 |
| Ensemble agents | 20 | 0.90 | Paper §3.3 |
| MCMC draws | 200 | 0.90 | Appendix B |
| MCMC tune | 300 | 0.90 | Appendix E.3 |
| Q-network activation | ReLU | **0.80** | **ASSUMED** |
| EMVS solver | weighted L1-LR | **0.62** | **ASSUMED** |

---

## Implementation Risks

| Severity | Risk | Mitigation |
|----------|------|-----------|
| 🔴 High | EMVS inner-loop solver not specified | Abstract `IsingFitter` base class; default to scipy L-BFGS |
| 🟡 Medium | HMC convergence at large N | Diagnostics + EMVS-first default |
| 🟡 Medium | Village adjacency data access | `download_villages.py` for Harvard Dataverse |
| 🟡 Medium | d3rlpy API version drift | Pin version; thin CQL adapter |
| 🟢 Low | Ensemble wall-clock (~20 min) | Expose `--n-agents` flag |

---

*Generated by ArXivist Stage 3 — Architecture Planner*
