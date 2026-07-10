# Architecture Plan Summary — RL for Trade Execution with Market and Limit Orders

**Paper ID:** arxiv_2507_06345 | **Framework:** PyTorch (>=2.1) | **Config:** plain YAML

## Why PyTorch
The paper's implementation is explicitly based on the CleanRL PPO codebase (Huang et al., 2022),
which is PyTorch-native. Models are small 2-layer MLPs; the real cost is the CPU-bound limit
order book simulator run across 128 parallel environments, so GPU is optional.

## Module Hierarchy
```
src/rlte/
├── models/policy.py        LogisticNormalPolicy, DirichletPolicy (actor networks)
├── models/value.py         ValueNetwork (critic)
├── models/distributions.py LogisticNormalTransform (h, h^-1, variance schedule)
├── env/order_book.py       LimitOrderBook (FIFO price-level queues)
├── env/traders.py          NoiseTraders, TacticalTraders, StrategicTrader
├── env/execution_env.py    TradeExecutionEnv, VectorizedTradeExecutionEnv
├── training/trainer.py     ActorCriticTrainer (Algorithm 1)
├── training/losses.py      policy_loss (Eq.14), value_loss (Eq.15)
├── evaluation/benchmarks.py SubmitAndLeave, TWAP heuristics
├── evaluation/metrics.py   Evaluator (10,000-sim Monte Carlo evaluation)
└── utils/{config.py, features.py}
```

## Key Tensor Flows
1. **Policy forward+sample**: state `[B, state_dim]` → 2×tanh MLP → `mu [B,K=6]` → sample
   `x ~ N(mu, sigma_i*I)` → logistic-normal transform → action `a in S^6 [B,7]`.
2. **Value forward**: state → 2×tanh MLP → `V(s) [B,1]`.
3. **Env step**: action allocation → round to integer lots → cancel/submit orders →
   advance clock by `dt=15s` running trader agents → accumulate cash flow → normalize reward
   `r=(r̄-γ·p^b(0))/M` → next normalized state.
4. **Training iteration**: collect τ=1280 trajectories (128 envs × 10 traj) of length N=10 →
   advantage = MC return − V(s) (GAE λ=1) → Adam step (lr=5e-4) on policy loss (Eq.14) and
   value loss (Eq.15) → anneal Σ=σ_i·I per Eq.12 → repeat for H=400 iterations.

## Config Highlights (with confidence flags)
- `K=6` simplex dimension (confidence 0.95, explicit)
- `sigma_init=1.0 → sigma_final=0.1` linear schedule over H=400 steps (explicit)
- `lr=0.0005`, Adam (explicit); **`beta1/beta2` ASSUMED PyTorch defaults (0.9/0.999), confidence 0.6**
- `num_envs=128`, `tau=1280` trajectories, `N=10` steps/traj (explicit, Appendix B.1)
- **Eval seed ASSUMED, not stated in paper (confidence 0.4)** — documented, results should be
  reported as mean/std across multiple seeds rather than treated as exact reproductions.

## Entrypoints
- `train.py --config <yaml> --algorithm LN|DR --market ... --lots 20|60`
- `evaluate.py --checkpoint <path> --policy LN|DR|SL|TWAP --num-sims 10000`
- `inference.py --checkpoint <path>` — single-episode visualization (Fig. 1/2/6/7 style)

## Top Risks
| Severity | Risk | Mitigation |
|---|---|---|
| Medium | Rounding rule for simplex→lots under-specified | Configurable `RoundingStrategy`, sequential-floor default |
| Medium | Custom LOB simulator correctness/perf at 128-env scale | Deque-based FIFO queues + multiprocessing pool + unit tests vs. Figure 1 worked example |
| Medium | Long training time (~1.2–2h/config on 64-core/128-thread server) | `--quick-test` mode with reduced envs/iterations |
| Low | Value-net init gains not 100% re-confirmed | Reuse policy-net gains (0.01), flagged `# ASSUMED` |
| Low | Adam betas unstated | PyTorch defaults, config-overridable |
| Low | Random seeds unstated | Multi-seed evaluation reporting, link to official repo for cross-check |

Full machine-readable plan: `architecture_plan.json` (same directory).
