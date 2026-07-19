# Data

This repository requires **no external dataset**. The paper's entire analysis
is built on a self-contained limit-order-book agent-based model (Section 2)
that generates its own simulated trade tape, price series, and daily
OHLCV as it runs.

## Running the simulation to produce "data"

```bash
python train.py --config configs/config.yaml --debug   # fast smoke test
python train.py --config configs/config.yaml            # full 2000-stock baseline
```

Outputs are written to `results/{scenario}/simulation_results.pkl`.

## External benchmark (not downloaded, just referenced)

The paper compares its simulated ⟨δ⟩ = 0.539 against an empirical benchmark
from the Tokyo Stock Exchange: ⟨δ⟩ = 0.489, ⟨c⟩ = 0.842, reported in:

> Y. Sato, K. Kanazawa, "Strict Universality of the Square-Root Law of Price
> Impact across Stocks: A Complete Survey of the Tokyo Stock Exchange,"
> Physical Review Letters 135(25), 257401 (2025).

Only these two summary numbers are used in this repository (baked into
`configs/config.yaml` under `evaluation.tse_benchmark_mean_delta` /
`tse_benchmark_mean_c`). The underlying per-stock TSE trade data is
proprietary and is not required to reproduce this paper's simulated results.
