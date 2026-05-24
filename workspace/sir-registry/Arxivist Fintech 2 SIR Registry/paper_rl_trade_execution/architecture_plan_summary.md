# Architecture Plan — RL Trade Execution (Nevmyvaka et al., ICML 2006)

## Framework Decision
**Pure Python + NumPy** — no neural network, no GPU required. This is a tabular RL paper from 2006.
The algorithm is a backward-induction Q-learning hybrid with a lookup table as the central data structure.

---

## Module Hierarchy

```
src/rl_trade_execution/
├── env/
│   ├── order_book.py         ← OrderBook, OrderBookSnapshot: load & simulate INET data
│   ├── market_env.py         ← TradeExecutionEnv: state encoding, step(), reset()
│   └── market_features.py    ← MarketFeatureExtractor: bid-ask spread, imbalance, tx volume
├── agent/
│   ├── q_table.py            ← QTable: numpy array, incremental update, best_action()
│   └── policy.py             ← OptimalPolicy: wraps Q-table, act() interface
├── training/
│   └── trainer.py            ← BackwardInductionTrainer: main training loop
├── baselines/
│   ├── submit_and_leave.py   ← S&L baseline with optimized fixed limit offset
│   └── market_order.py       ← Immediate full market order baseline
├── evaluation/
│   └── metrics.py            ← trading_cost_bps(), relative_improvement(), aggregation
├── data/
│   └── loader.py             ← INETDataLoader: load order books, partition episodes
└── utils/
    └── config.py             ← ExperimentConfig from YAML
```

---

## Core Algorithm Flow

### Training (Backward Induction)
```
for t = T downto 0:
  for each order book snapshot in training data:
    extract and discretize market features o1..oR
    for each inventory level i in 0..I:
      for each action a in 0..L:
        simulate execution of i shares at price (ask - a)
        compute immediate cost cim
        look up best future cost from already-computed t+1 states
        update Q-table: c(x,a) = n/(n+1)*c(x,a) + 1/(n+1)*(cim + best_future_cost)
```

Key property: the independence of market variables from actions means we only need
`T × I × L` passes over the data — runtime is **independent of R** (number of market vars).

### Inference
```
given (t, i, current_order_book):
  features = extract_and_discretize(order_book)
  state = encode(t, i, features)
  action = argmin_a Q[state, a]
  submit limit order at (ask - action) for remaining shares
```

---

## State Space

| Component | Type | Values |
|-----------|------|--------|
| `t` | time remaining | 0..T (default T=8) |
| `i` | inventory units left | 0..I (default I=8) |
| `o1` bid-ask spread | discretized | {0,1,2} low/med/high |
| `o2` market order cost | discretized | {0,1,2} |
| `o3` signed tx volume | discretized | {0,1,2} |

Q-table size: `T × I × 3^R × L` — e.g. 8×8×27×21 ≈ 36K entries (trivially small).

---

## Key Config Parameters

```yaml
experiment:
  stock: AMZN          # AMZN | NVDA | QCOM
  V: 10000             # shares to execute
  H_minutes: 2         # horizon (2 or 8)

state_space:
  T: 8                 # time resolution
  I: 8                 # inventory resolution
  market_variables: [bid_ask_spread, immediate_market_order_cost]
  n_bins_market: 3     # ASSUMED: from paper description

action_space:
  L: 21                # ASSUMED: inferred from figures (confidence 0.60)
  action_min: -6       # passive (own book)
  action_max: 14       # aggressive (cross spread)
```

---

## Risk Summary

| Severity | Issue |
|----------|-------|
| 🔴 High | INET data is proprietary — synthetic generator needed for reproducibility |
| 🟡 Medium | Action space size L not explicitly stated in paper |
| 🟡 Medium | Market variable bin counts partially unspecified |
| 🟢 Low | Reward/cost formula is clearly described in basis points |
| 🟢 Low | Memory: Q-table is tiny, no scalability concerns |

---

## Entrypoints

- `train.py` — run full backward induction training for a given config
- `evaluate.py` — evaluate a saved policy on test episodes
- `simulate_episode.py` — visualize a single episode step-by-step
