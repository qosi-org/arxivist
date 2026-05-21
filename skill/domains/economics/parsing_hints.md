# Domain: Economics — Parsing Hints (Stage 1 Enrichment)

Load alongside `agents/01_paper_parser.md` when the detected domain is **Economics**.

---

## Architecture extraction — Economics-specific rules

Economics papers describe **identification strategies, structural models, and estimation procedures**,
not software architectures. The "architecture" section captures the model structure:

**Causal inference papers:** extract:
- Treatment variable and outcome variable (exact names and units)
- Identification strategy: IV, DiD, RDD, synthetic control, matching
- Instrument(s) used, with the relevance and exclusion restriction arguments
- Control variables included in the specification
- Fixed effects structure (individual, time, two-way)
- Standard error clustering strategy

**Structural models (DSGE, IO):** extract:
- Agent types (households, firms, government) and their objective functions
- Market clearing conditions
- Equilibrium concept (Nash, competitive, social planner)
- Calibrated vs estimated parameters — record which are which

**Time series / macro models:** extract:
- VAR order p, variable list, identification scheme (Cholesky, sign restrictions, external IV)
- ARIMA orders (p, d, q) and seasonal components
- Cointegration rank if tested

**The mathematical_spec section is the most important** in economics papers.
Extract every regression equation, structural equation, and moment condition explicitly.

---

## Training pipeline — Economics-specific rules

"Training" = estimation. Extract:
- Estimator type: OLS, IV/2SLS, GMM, MLE, Bayesian, simulation-based
- Software used: Stata, R (which packages), Python (statsmodels, linearmodels)
- Number of bootstrap or simulation replications
- Optimiser for structural estimation (if applicable)
- Weighting matrix for GMM
- Instrument list for IV — exactly as stated

---

## Evaluation — Economics-specific rules

Extract:
- First-stage F-statistic (for IV papers — below 10 is a weak instrument)
- p-values and significance levels used
- R² / adjusted R² for OLS
- Hansen J-statistic for GMM overidentification
- Robustness checks described (placebo tests, alternative specifications)
- Sample size (N) and time periods (T) for panel data
