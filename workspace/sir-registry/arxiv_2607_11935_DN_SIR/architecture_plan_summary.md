# Architecture Plan Summary — arxiv_2607_11935
**Eigenvector rotation precedes eigenvalue-based early-warning signals: a TVP-Kalman approach to detecting critical transitions**

## Framework
- **Primary**: PyTorch (schema-required field only — **not actually used anywhere**; this paper has no trained model).
- The real computational stack is **NumPy + SciPy + scikit-learn** (linear Kalman recursions, statistics, kNN mutual information).
- Python 3.10+, no CUDA (series length 284, simulations a few hundred timesteps — trivially fast on CPU).
- Config: plain YAML.

## Module Hierarchy (10 files)
| Module | Role |
|---|---|
| `kalman/transition_matrix.py` | Order-3 Taylor-expansion state-transition matrix `F` |
| `kalman/tvp_kalman.py` | **Core method**: TVP-Kalman forward filter + RTS smoother, supports `mode="loglog"` (real climate data) and `mode="linear"` (2 of the 6 simulated systems) |
| `ews/classical_ews.py` | AR(1), variance, permutation entropy, mutual information (rolling windows) |
| `ews/lead_lag.py` | Cross-correlation lead-lag + significance-threshold detection |
| `data/airs_loader.py` | NASA AIRS regional loader + synthetic fallback |
| `simulation/tipping_systems.py` | Six simulated tipping-point system generators |
| `evaluation/region_stats.py` | Table 1 + Table 2 reproduction |
| `evaluation/simulation_validation.py` | Table 3 reproduction |
| `utils/plotting.py` | Figures 1–3 |
| `utils/config.py` | YAML config + global seeding |

## Key Design Decision: Two Kalman Modes
The paper defines two of its six validation systems ("beta step change", "beta linear decay") via an **explicit linear** relationship `y = β(t)x + ε`, distinct from the log-log elasticity used for the real climate application. `TVPKalmanFilter` supports both modes. **This was caught as a real bug during testing** — applying log-log mode uniformly caused a spurious multi-hundred-timestep "detection" from the filter's own warm-up transient, not a genuine early-warning signal.

## No Training Pipeline
R and Q are fixed filter-design hyperparameters given directly in the paper (Section 2.2) — there's no optimizer, no loss, no learned parameters anywhere in this reproduction.

## Dependencies
`numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml` (+ pytest, black, ruff for dev).

## Entrypoints
- `run_observational_analysis.py` — Tables 1–2, Figures 1–2 (3 AIRS regions)
- `run_simulation_validation.py` — Table 3 (6 simulated systems)
- `run_all.py` — both in sequence

## Top Risks
1. **[High]** Kalman transition matrix `F` is named but not numerically specified in the paper → standard factorial-scaled Taylor matrix used, swappable.
2. **[High]** Six simulated systems' noise/forcing parameters unspecified → illustrative parameters reproduce the *qualitative* Table 3 pattern well, not exact numbers.
3. **[Medium]** Linear-vs-loglog Kalman mode mismatch → **found and fixed** during testing (see above).
4. **[Medium]** NASA AIRS requires manual Giovanni-portal pull → synthetic fallback + documented `download.sh`.
5. **[Low]** No trained model, by design → prominently stated in README to avoid confusion with the quantum-CVA reproduction's very different pattern.

**Next**: Stage 4 (Code Generator) is complete — full repo built, tested (46/46 passing), and verified end-to-end at `paper-repos/arxiv_2607_11935/`.
