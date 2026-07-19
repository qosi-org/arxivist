# TVP-Kalman Eigenvector-Rotation Early-Warning Signals — Reproduction of arXiv:2607.11935

Reproduction of **"Eigenvector rotation precedes eigenvalue-based
early-warning signals: a TVP-Kalman approach to detecting critical
transitions"** (Gildas Ngueuleweu Tiwang, University of Dschang, Cameroon,
2026).

The paper introduces a new early-warning signal (EWS) for critical
transitions: the time-varying elasticity β(t) = dlog(y)/dlog(x), estimated
via a TVP-Kalman filter, which has O(δθ) sensitivity to bifurcation-parameter
perturbations versus O(δθ²) for classical eigenvalue-based EWS (AR(1),
variance). β is shown to be orthogonal to AR(1) and to precede it by 14–24
months on NASA AIRS climate data, and by 39–153 timesteps on six simulated
tipping-point systems.

## What's implemented

| Paper section | Module |
|---|---|
| 2.2 (TVP-Kalman elasticity estimation) | `src/ews_kalman/kalman/tvp_kalman.py`, `transition_matrix.py` |
| 2.3 (classical rolling-window EWS) | `src/ews_kalman/ews/classical_ews.py` |
| 2.4, 3.2 (lead-lag / significance analysis) | `src/ews_kalman/ews/lead_lag.py` |
| 2.1 (NASA AIRS regional data) | `src/ews_kalman/data/airs_loader.py` |
| 2.4, Figure 3 (six simulated tipping systems) | `src/ews_kalman/simulation/tipping_systems.py` |
| Table 1, Table 2 | `src/ews_kalman/evaluation/region_stats.py` |
| Table 3 | `src/ews_kalman/evaluation/simulation_validation.py` |
| Figures 1–3 | `src/ews_kalman/utils/plotting.py` |

**Note: this paper has no trained model.** The Kalman filter's R and Q are
fixed design hyperparameters given directly in the paper (Section 2.2), not
learned/fit values — there is no training loop, optimizer, or loss
minimisation anywhere in this pipeline.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Observational analysis (Tables 1-2, Figures 1-2)
python run_observational_analysis.py --config configs/config.yaml --region all

# Simulated tipping-point validation (Table 3)
python run_simulation_validation.py --config configs/config.yaml --system all

# Both, in sequence
python run_all.py --config configs/config.yaml --output-dir results/
```

Or via Docker:
```bash
docker compose -f docker/docker-compose.yml up --build
```

## Repository layout

```
configs/config.yaml              All hyperparameters (# ASSUMED comments flag unstated values)
src/ews_kalman/
  kalman/                        TVP-Kalman forward filter + RTS smoother + Taylor transition matrix
  ews/                           Classical rolling-window EWS (AR1, variance, PE, MI) + lead-lag analysis
  data/                          NASA AIRS regional loader (+ synthetic fallback)
  simulation/                    Six simulated tipping-point system generators
  evaluation/                    Table 1/2/3 reproduction logic
  utils/                         Config loading, seeding, Figure 1-3 plotting
run_observational_analysis.py / run_simulation_validation.py / run_all.py   Entrypoints
tests/                           46 unit tests covering every module
data/                            NASA AIRS data requirements + synthetic-fallback docs
docker/                          Dockerfile + docker-compose.yml
```

## Reproducibility notes (read before trusting exact numbers)

This is a statistics/signal-processing paper, not a deep-learning paper —
there's no GPU, no training loop, and no learned parameters. The gaps that
matter here are in the paper's own level of numerical detail, not in model
training:

1. **The Kalman state-transition matrix F** is named only as "the Taylor
   expansion transition matrix of order 3" — no numeric entries given. This
   repo uses the standard factorial-scaled kinematic-chain matrix (see
   `transition_matrix.py` docstring), which is the conventional choice for
   this kind of model but is not verified against the author's exact
   implementation.
2. **The six simulated systems' noise magnitudes, forcing schedules, and
   simulation lengths are not specified numerically** (Section 2.4 gives
   only qualitative equations and bifurcation-parameter endpoints). Running
   `run_simulation_validation.py` with this repo's illustrative parameters
   reproduces the **qualitative pattern** of Table 3 well — fold bifurcation
   is won by AR(1) (β never detects, since coupling itself is unchanged),
   logistic map is a tie, and β wins clearly on Stommel AMOC and critical
   slowing down (coupling-degradation systems) — but exact lead-time values
   (e.g. "+39" or "+153" timesteps) are not guaranteed to match.
3. **Two of the six simulated systems use a linear relationship, not
   log-log.** The paper explicitly defines "beta step change" and "beta
   linear decay" as `y = β(t)x + ε` (Section 2.4) — a genuinely different
   observation model from the log-log elasticity used for the real T-q
   climate application (Section 2.2). `TVPKalmanFilter` supports both via a
   `mode="loglog"` / `mode="linear"` argument; applying log-log mode to
   these two systems (an earlier bug caught during testing) caused a
   multi-hundred-timestep filter warm-up transient that looked like a false
   "detection" — fixed by using linear mode for those two systems
   specifically.
4. **NASA AIRS data is freely available but requires an interactive/manual
   pull** (Giovanni portal) — see `data/README_data.md`. A synthetic
   fallback runs the full pipeline out-of-the-box, calibrated to roughly
   match the paper's reported |β| per region, but is not real climate data.
5. **Permutation entropy and mutual-information estimator specifics**
   (exact algorithm/library, k for the kNN-based MI estimator) are named
   but not fully specified; standard implementations (Bandt-Pompe ordinal
   patterns; scikit-learn's Kraskov-style kNN MI estimator) are used.

Full detail and confidence scores per section are in
`sir-registry/arxiv_2607_11935/sir.json` (Stage 1) and
`architecture_plan.json` (Stage 3, `risk_assessment` field).

## Testing

```bash
pytest tests/ -v          # 46 tests, all passing
ruff check src/ *.py       # clean
```

## Citation

```
Ngueuleweu Tiwang, G. (2026). Eigenvector rotation precedes eigenvalue-based
early-warning signals: a TVP-Kalman approach to detecting critical
transitions. arXiv:2607.11935.
```

This is an independent reproduction generated by the ArXivist pipeline; it
is not affiliated with the paper's author. See `data/README_data.md` for
NASA AIRS data-access details.
