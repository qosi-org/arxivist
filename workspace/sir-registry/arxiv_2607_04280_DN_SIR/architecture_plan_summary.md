# Architecture Plan Summary — arxiv_2607_04280

**Paper:** Order Splitting and Liquidity Replenishment Are Jointly Necessary for the Square-Root Law of Market Impact: A Counterfactual Dissection (Zhou, Chen & Wei, 2026)

## Framework
PyTorch (as a vectorized array backend + seeding utility only — **no gradient training occurs**), Python 3.10+, CPU by default, plain YAML config. This is a discrete-event agent-based simulation, not a neural network; the ArXivist repo template's "model/training" vocabulary is repurposed accordingly (see mapping table below).

## How ArXivist's ML vocabulary maps onto this paper

| Template concept | What it actually is here |
|---|---|
| "Model" | `StockMarketSimulation` — a LOB + 4 agent types |
| "Training" | Batch discrete-event simulation across up to 2000 stocks (`BatchSimulationRunner`) |
| "Loss" | Relative least-squares impact-curve fit (Eq. 3) |
| "Evaluation metrics" | Fitted δ, c, β, α, γ per stock/scenario |
| "Checkpoints" | Saved trade tapes / per-stock results tables (not model weights) |

## Module hierarchy (10 files)
- `models/lob.py` — `LimitOrderBook`: price-time-priority matching engine
- `models/agents.py` — `InstitutionalAgent`, `HFTMarketMakerAgent`, `RetailAgent`, `NewsAgent`, `MomentumAgent`
- `models/market.py` — `StockMarketSimulation`: wires agents + LOB into one stock's run
- `data/dataset.py` — `StockParameterSampler`: draws the 2000 independent stock configs
- `data/transforms.py` — `MetaorderReconstructor`: ex-post grouping + Eq. 2 normalization
- `training/trainer.py` — `BatchSimulationRunner`: parallel per-stock + counterfactual suite runner
- `training/losses.py` — `RelativeLeastSquaresFit`: Eq. 3
- `evaluation/metrics.py` — `ImpactCurveFitter`, `TailExponentEstimator`, `TheoryPredictors` (GGPS/FGLW/LOB-walking)
- `utils/config.py` — YAML loading + global seeding

## Entrypoints
- `train.py` — run baseline or one named scenario across N stocks
- `evaluate.py` — fit impact curves, reproduce Table 1 / Figures 3–7
- `inference.py` — single-stock quick check
- `run_counterfactual_suite.py` — paper-specific: all 8 scenarios × 20 stocks → Table 1 + Figure 7

## Key risks (full detail in `architecture_plan.json.risk_assessment`)
1. **[High]** HFT adaptive-spread rule and news-agent trigger process are not numerically specified — implemented as swappable strategies, flagged in README.
2. **[Medium]** Price-limit / momentum / low-liquidity ablation parameters are named, not quantified — literature-standard defaults used, marked `# ASSUMED` in config.
3. **[Medium]** Dirichlet/uniform/front-loaded splitting rules are named, not mathematically defined — implemented with the most standard interpretation of each.
4. **[Low]** Convergence check (first-half vs. second-half δ, tolerance 0.01) is weaker than typical ML validation — reproduced as an explicit automated check in `evaluate.py`.
5. **[Low]** Matching-engine internals (cancellation policy, tie-breaking) are common-practice defaults, not stated — documented inline, γ reported as emergent rather than assumed.

## Dependencies
`numpy`, `torch` (array backend only), `pandas`, `scipy`, `matplotlib`, `pyyaml`, `joblib` (parallel per-stock runs), `tqdm`. Dev: `pytest`, `black`, `ruff`, `jupyter`.
