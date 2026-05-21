# Data for SPG-UVM

This repository requires **no external datasets**.

All experiments use **synthetically generated Monte Carlo paths** simulated
directly from the multi-asset Black-Scholes model under the Uncertain
Volatility Model (UVM). Paths are generated on-the-fly during training
using the log-Euler scheme implemented in `src/spg_uvm/models/dynamics.py`.

## Synthetic Data Generation

At each training step n, the `StateSampler` draws asset price states X_n
from a log-normal distribution with diagonal covariance (Section 4.1.3).
The log-Euler scheme then simulates one step forward given a sampled action.

No download scripts are needed. Simply run:

```bash
python train.py --config configs/default.yaml
```

## Reference Prices

Reference prices for validation are taken from:
  - Goudenège, Molent, Zanette (2024), "Robust option pricing: The approach
    of the uncertain volatility model" (GTU and NNU methods).

These are listed in `configs/default.yaml` under `evaluation.reference_price`.
Set the appropriate value for your experiment before running.
