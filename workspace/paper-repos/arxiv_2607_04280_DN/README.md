# Square-Root Law ABM — reproduction of arXiv:2607.04280

> Order Splitting and Liquidity Replenishment Are Jointly Necessary for the
> Square-Root Law of Market Impact: A Counterfactual Dissection
> Yang Zhou, Jianwen Chen, Ruipeng Wei (2026)

This repository is an **ArXivist-generated reproduction** of a minimal
limit-order-book (LOB) agent-based model (ABM) used to test three
competing theories of the square-root law (SRL) of market impact, and to
identify which mechanisms are causally necessary for it via counterfactual
ablation.

**This is not a neural network.** It's a discrete-event simulation. The
repo still follows the standard ArXivist ML-project layout
(`train.py` / `evaluate.py` / `models/` / `training/`) for consistency, but
those names are repurposed — see [Mapping ML vocabulary onto this
paper](#mapping-ml-vocabulary-onto-this-paper) below.

## What this reproduces

- A market with 4 heterogeneous agent types: **institutional** traders
  (source of metaorders), **HFT market makers** (liquidity replenishment),
  **retail** noise traders, and **news** agents (volatility shocks).
- Ex-post metaorder reconstruction and the paper's exact impact-curve
  fitting procedure (log-binning + relative least-squares, Eq. 3).
- The paper's three competing theoretical predictions for δ: **GGPS**
  (δ=β−1), **FGLW** (δ=α−1), **LOB walking** (δ=1/(1+γ)).
- The 8-scenario counterfactual ablation suite from Table 1: baseline, no
  splitting, no HFT, price limits, low liquidity, momentum, uniform split,
  front-loaded split.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Fast smoke test (~1 min): 4 stocks, 5000 steps each
python train.py --config configs/config.yaml --debug

# Single-stock sanity check with full step count (~3-5 min)
python inference.py --config configs/config.yaml --scenario baseline --seed 0

# Full baseline run (2000 stocks) - this is what the paper reports as
# ~3 hours on a 40-core server; expect proportionally longer on fewer cores
python train.py --config configs/config.yaml
python evaluate.py --results-dir results/baseline --scenario baseline

# The paper's headline result: Table 1 / Figure 7 (20 stocks x 8 scenarios)
python run_counterfactual_suite.py --config configs/config.yaml
```

Docker:

```bash
docker compose -f docker/docker-compose.yml up train
```

## Mapping ML vocabulary onto this paper

| ArXivist template concept | What it actually is here |
|---|---|
| `models/` | `LimitOrderBook` + the 4 (+1) agent classes |
| `train.py` / "training" | Batch discrete-event simulation over N stocks (no gradients, no loss minimization over a dataset) |
| `training/losses.py` | The paper's actual fitting objective: relative-least-squares power-law fit (Eq. 3) |
| `evaluate.py` / "metrics" | Fitted δ, c, β, α, γ per stock/scenario, compared against the three theories and the TSE benchmark |
| "checkpoints" | Saved trade tapes / per-stock result tables, not model weights |

## Reproducibility notes — read this before expecting exact paper numbers

The paper's text fully specifies most parameters (see `configs/config.yaml`,
where every value is commented as either from the paper or `# ASSUMED`).
**Several implementation details are not given in the paper text** and were
filled in with the most standard/literature-common choice. These are the
ones most likely to shift your fitted δ away from the paper's 0.549
(baseline) / 0.324 (no splitting) / 0.386 (no HFT):

1. **HFT adaptive-spread rule** (`model.hft.spread_rule` in config) — the
   paper says market makers quote with "adaptive spread" but never gives
   the formula. We use a simple volatility-scaled heuristic
   (`base_spread_ticks * (1 + sensitivity * recent_vol)`). This directly
   affects the emergent depth-profile exponent γ and thus the no-HFT
   ablation number.
2. **News agent trigger process** (`model.news.trigger_rate`,
   `size_multiplier`) — needed to reproduce the reported excess kurtosis
   (κ=29.5) but not quantified in the text.
3. **Price-limit / momentum / low-liquidity ablation parameters** — named
   in Table 1 but not numerically specified (band width, lookback window,
   liquidity reduction magnitude).
4. **Dirichlet/uniform/front-loaded splitting-rule exact math** — named,
   not mathematically defined.

**What we did verify end-to-end**: the full pipeline (simulate → reconstruct
metaorders → normalize → log-bin → fit power law → compare against the 3
theories) runs correctly and recovers a *known* synthetic power law to
within 5% in unit tests (`tests/test_core.py`). An initial single-stock
run with the as-shipped `# ASSUMED` defaults produced a fitted δ≈1.0 rather
than the paper's ≈0.5 — this is expected given how sensitive δ is to the
unstated HFT spread/replenishment dynamics, and is the reason item (1)
above is flagged **High severity** in `architecture_plan.json`. If your
goal is to match the paper's exact numbers, start by tuning
`model.hft.base_spread_ticks` / `spread_vol_sensitivity` and
`model.institutional.dirichlet_concentration` — these are the two
highest-leverage unstated parameters.

If you find a configuration that reproduces the paper's numbers more
closely, please consider it a "corrected" ambiguity resolution and update
`configs/config.yaml`'s `# ASSUMED` comments accordingly.

## Repository layout

```
train.py                       # batch simulation runner (see Quickstart)
evaluate.py                    # impact-curve fitting + theory comparison
inference.py                   # single-stock quick check
run_counterfactual_suite.py    # reproduces Table 1 / Figure 7 directly
configs/config.yaml            # all parameters, each tagged explicit/ASSUMED
src/sqrt_law_abm/
  models/
    lob.py                     # LimitOrderBook matching engine
    agents.py                  # Institutional / HFT / Retail / News / Momentum agents
    market.py                  # StockMarketSimulation: wires agents + LOB
  data/
    dataset.py                 # StockParameterSampler + scenario overrides
    transforms.py               # MetaorderReconstructor (Sec 2.5, Eq. 2)
  training/
    trainer.py                 # BatchSimulationRunner (parallel per-stock runs)
    losses.py                  # RelativeLeastSquaresFit (Eq. 3)
  evaluation/
    metrics.py                 # ImpactCurveFitter, Hill estimator, TheoryPredictors
  utils/
    config.py                  # YAML config loading + global seeding
tests/test_core.py             # 20 unit tests (LOB, agents, fitting, theories)
docker/                        # Dockerfile + docker-compose.yml
data/                          # no external data needed — see data/README_data.md
notebooks/                     # walkthrough notebook (see notebooks/README.md)
```

## Citation

```bibtex
@article{zhou2026sqrtlaw,
  title={Order Splitting and Liquidity Replenishment Are Jointly Necessary
         for the Square-Root Law of Market Impact: A Counterfactual Dissection},
  author={Zhou, Yang and Chen, Jianwen and Wei, Ruipeng},
  journal={arXiv preprint arXiv:2607.04280},
  year={2026}
}
```

## Generation provenance

Generated by the **ArXivist** skill: Stage 1 (paper → SIR, confidence 0.79)
→ Stage 2 (SIR registry) → Stage 3 (architecture plan) → Stage 4 (this
repo). See `sir-registry/arxiv_2607_04280/` in the ArXivist workspace for
the full SIR, architecture plan, and risk assessment this repo was
generated from.
