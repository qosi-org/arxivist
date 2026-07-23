# Reinforcement Learning for Trade Execution with Market and Limit Orders

**Authors:** Patrick Cheridito, Moritz Weiss (Department of Mathematics, ETH Zurich)
**arXiv:** [2507.06345](https://arxiv.org/abs/2507.06345)
**Official code:** https://github.com/moritzweiss/rlte

> ArXivist-generated reference implementation scaffold. This repo was produced by an
> automated paper-to-code pipeline (Paper Parser &rarr; SIR Registry &rarr; Architecture
> Planner &rarr; Code Generator &rarr; Notebook Generator). See **Reproducibility Notes**
> below for what is faithfully implemented vs. assumed/stubbed.

## What this paper does

The paper introduces a reinforcement learning framework for **optimal trade execution**
in a limit order book. An algorithm must sell `M` lots over a fixed horizon by choosing,
at each of `N` discrete decision steps, an allocation across a market order, limit orders
at several price levels, and "do nothing" -- represented as a point on a probability
simplex. Actions are sampled from a novel **logistic-normal distribution** policy, trained
with an actor-critic policy gradient. The method is evaluated in simulated limit order
book markets populated by noise, tactical (imbalance-reactive), and strategic (large
TWAP) traders, so that the algorithm's own orders generate realistic market impact.

## Quick Start

```bash
git clone <this-repo> && cd arxiv_2507_06345
pip install -e .
python train.py --config configs/noise_20lots.yaml --debug --dry-run   # sanity check
python train.py --config configs/noise_20lots.yaml                     # full training
python evaluate.py --policy LN --checkpoint runs/final_checkpoint.pt --market noise --lots 20
jupyter lab notebooks/reproduce_arxiv_2507_06345.ipynb
```

## Installation

**pip:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**conda:**
```bash
conda env create -f environment.yaml
conda activate rlte
pip install -e .
```

**Docker:**
```bash
docker compose -f docker/docker-compose.yml up train
docker compose -f docker/docker-compose.yml up notebook   # Jupyter on :8888
```

## Training

```bash
python train.py --config configs/noise_20lots.yaml
python train.py --config configs/noise_tactical_60lots.yaml --seed 1
python train.py --config configs/noise_tactical_strategic_20lots.yaml --debug   # quick smoke test
```

Pre-built configs cover all 6 paper configurations (3 markets x 2 position sizes) under
`configs/{market}_{lots}lots.yaml`.

**Note:** the paper reports ~1.2-2 hours per configuration on a 64-core/128-thread
server with an RTX 4090 (Appendix B.1, Table 7). Use `--debug` for a fast smoke test on
modest hardware.

## Evaluation

```bash
python evaluate.py --policy SL --market noise --lots 20             # heuristic baseline
python evaluate.py --policy TWAP --market noise --lots 20           # heuristic baseline
python evaluate.py --policy LN --checkpoint runs/final_checkpoint.pt --market noise --lots 20
```

Runs 10,000 Monte Carlo market simulations (paper default) and reports the expected
normalized reward and its standard deviation, matching Table 1's format.

## Expected Results (Table 1, paper-reported)

Expected normalized reward per lot (ticks relative to initial best bid); best value per row in **bold**.

| Market | Lots | SL | TWAP | DR | LN |
|---|---|---|---|---|---|
| Noise | 20 | 0.52 | -0.06 | 0.21 | **0.61** |
| Noise | 60 | -1.09 | -1.40 | **-0.71** | -0.72 |
| Noise & Tactical | 20 | 0.10 | 0.48 | 0.73 | **0.81** |
| Noise & Tactical | 60 | -3.36 | -0.96 | **-0.23** | -0.25 |
| Noise & Tactical & Strategic | 20 | -1.64 | -0.36 | 1.06 | **1.13** |
| Noise & Tactical & Strategic | 60 | -2.51 | -1.45 | 0.03 | **0.23** |

## Implementation Assumptions

The paper does not specify every implementation detail. Key assumptions made in this
scaffold (full list with confidence scores in `sir-registry`'s `sir.json` /
`architecture_plan.json`):

| Assumption | Basis | Confidence |
|---|---|---|
| Adam `beta1=0.9`, `beta2=0.999` | PyTorch default, not stated in paper | 0.6 |
| Simplex-to-lots rounding: sequential floor, level 0..K-1 | Paper describes rule at a high level only | 0.6 |
| Value network init: orthogonal, gain=0.01, all layers | Explicitly stated only for policy network | 0.65 |
| PyTorch as implementation framework | Paper bases code on CleanRL (PyTorch) PPO | 0.85 |
| Evaluation random seed | Not stated in paper | 0.4 |

## Reproducibility Notes

- **Faithfully implemented**: logistic-normal transform and policy gradient (Eq. 8-9),
  variance annealing schedule (Eq. 12), policy/value losses (Eq. 14-15), noise/tactical
  trader intensity models (Section 5.1-5.2, Appendix A.1 Table 3), SL/TWAP heuristics
  (Section 3.4), Dirichlet policy math (Appendix B.2).
- **Simplified / scaffold-quality**: `TradeExecutionEnv._apply_action`'s inventory and
  own-order bookkeeping is a simplified reference (see `NOTE` comments in
  `execution_env.py`) — a production-grade reproduction should track individual limit
  order fills between decision steps precisely via `LimitOrderBook`'s fill callbacks
  rather than the coarse level-based reconciliation used here.
- **Not implemented (stub)**: the Dirichlet (DR) end-to-end trainer is not wired into
  `train.py` (raises `NotImplementedError` with pointers to the relevant classes);
  `data/estimate_equilibrium.py` (long-run average order book shape estimation,
  Appendix A.2, Figure 8) is not implemented — `FeatureNormalizer` falls back to a flat
  placeholder shape.
- **Known divergence risk**: exact numeric reproduction of Table 1 is not guaranteed due
  to unstated random seeds and the simplified fill-bookkeeping noted above. Compare
  trends (LN > DR > TWAP > SL ordering, sign and rough magnitude of rewards) rather than
  exact digits.

## Citation

```bibtex
@article{cheridito2025rlte,
  title   = {Reinforcement Learning for Trade Execution with Market and Limit Orders},
  author  = {Cheridito, Patrick and Weiss, Moritz},
  journal = {arXiv preprint arXiv:2507.06345},
  year    = {2025}
}
```

## Repository Structure

```
.
├── src/rlte/
│   ├── models/       policy.py, value.py, distributions.py
│   ├── env/          order_book.py, traders.py, execution_env.py
│   ├── training/      trainer.py, losses.py
│   ├── evaluation/    benchmarks.py, metrics.py
│   └── utils/         config.py, features.py
├── configs/           6 market/lot-size YAML configs + base config.yaml
├── docker/            Dockerfile, docker-compose.yml
├── data/               README_data.md (no external data needed; simulated market)
├── notebooks/          reproduce_arxiv_2507_06345.ipynb
├── results/             evaluation output JSON goes here
├── comparison/           Stage 6 output goes here (not yet run)
├── train.py, evaluate.py, inference.py
└── requirements*.txt, environment.yaml, pyproject.toml
```
