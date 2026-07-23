# Architecture Plan Summary — arxiv_2606_14484
**Quantum Horizon: An evaluation of quantum computing as a threat to Bitcoin and Ethereum**

## Framework
- **Primary**: PyTorch (schema-required field only — **not actually used**; this paper has no trained model).
- The real stack is **NumPy + SciPy + pandas** (Monte-Carlo sampling, deterministic calibrated formulas, tabular reconciliation).
- Python 3.10+, no CUDA (everything here is scalar/low-dimensional Monte-Carlo or closed-form — trivially CPU-fast).
- Config: plain YAML, with heavy `# ASSUMED` annotation since this paper describes several of its models narratively rather than as explicit formulas.

## Module Hierarchy (11 files)
| Module | Maps to paper section |
|---|---|
| `timeline/physics_estimator.py`, `survey_estimator.py`, `monte_carlo_forecast.py` | §3.2 systemic break-year forecast (Figure 2) |
| `mining/mining_competitiveness.py` | §2.2 Grover-vs-PoW mining model (Figure 1) |
| `exposure/bitcoin_exposure.py`, `ethereum_exposure.py` | §4, §5 exposure reconciliation (Figure 3) |
| `attacks/mempool_race.py` | §4.3 mempool-sniping race |
| `migration/migration_race.py` | §7.4 Mosca's inequality (Figure 4) |
| `survey/readiness_rating.py` | §6 top-20 readiness survey (Table 3, Figure 5) |
| `utils/config.py`, `utils/plotting.py` | Config loading + all 5 figures |

## Key Design Decision: This Paper Has No Trained Model
Six independent, loosely-coupled quantitative models plus one explicitly-qualitative survey (the paper itself calls Section 6 "a sourced field assessment, not a model"). No optimizer, no loss, no learned parameters anywhere.

## The Big Gap: Narrative Models Without Explicit Formulas
Three sub-models are described in the paper only narratively (ranges + headline outputs, no reproducible formula):
1. **Monte-Carlo distributional forms** — the paper sweeps parameter *ranges* (hardware doubling time, resource-requirement halving, fault-tolerance lag) and states output *percentiles*, but never the sampling distribution shape or draw count.
2. **Mempool-race timing model** — reports 41%/30% success probabilities without showing the underlying calculation.
3. **Exposure-reconciliation method** — both Bitcoin and Ethereum exposure models combine multiple named sources into a headline range without stating the averaging/weighting method.

All three are handled the same way: implement a reasonable, clearly-labeled `# ASSUMED` default (documented in `config.yaml` and `risk_assessment`), expose every tunable as a config parameter, and provide the paper's own sensitivity-check logic (e.g. the 0.25–0.75 survey-weight sweep) so the reproduction is honest about what's precise vs. reverse-engineered to hit a target number.

## Dependencies
`numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml` (+ pytest, black, ruff for dev).

## Entrypoints
- `run_timeline_forecast.py` — Figure 2, systemic break-year forecast + sensitivity sweep
- `run_mining_analysis.py` — Figure 1, mining competitiveness
- `run_exposure_analysis.py` — Figure 3, Bitcoin + Ethereum exposure + mempool race
- `run_migration_race.py` — Figure 4, Mosca's inequality scenarios
- `run_readiness_survey.py` — Table 3 + Figure 5, top-20 readiness
- `run_all.py` — all five in sequence

## Top Risks
1. **[High]** Monte-Carlo distributional form/mixture method unspecified → every choice exposed in config, paper's own sensitivity sweep reproduced.
2. **[High]** Mempool-race timing model backed into a target number → propagation-delay parameter exposed explicitly, not hidden.
3. **[Medium]** Exposure-reconciliation method unspecified → reconciliation strategy is a swappable named config field.
4. **[Medium]** Top-20 survey is pure analyst judgment → transcribed to a versioned CSV with a drift-detection unit test.
5. **[Low]** Physics-estimator calibration constants back-solved from stated boundary conditions → unit-tested against the paper's own stated ~2052 physics-mode estimate.

**Next**: Stage 4 (Code Generator) builds the full repository at `paper-repos/arxiv_2606_14484/` from this plan.
