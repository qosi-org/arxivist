# Domain: ML — Parsing Hints (Stage 1 Enrichment)

Load alongside `agents/01_paper_parser.md` when the detected domain is **ML** (classical/probabilistic).

---

## Architecture extraction — ML-specific rules

Classical ML papers describe **model families** not layer graphs. Reframe the architecture section:
- For tree-based models: extract max_depth, n_estimators, min_samples_leaf, subsample ratio
- For SVMs: extract kernel type, C, gamma, whether dual or primal formulation
- For Gaussian Processes: extract kernel family (RBF, Matérn, spectral), mean function, likelihood
- For Bayesian models: extract prior distributions with parameters, likelihood form, inference method
- For ensemble methods: extract base learner, combination strategy, diversity mechanism

**No tensor shapes for most ML models** — the "tensor semantics" section will be sparse or null.
Record feature dimensionality instead: `n_features`, `n_classes`, `n_samples` from the paper.

**Algorithm pseudocode is the architecture** in many ML papers. Extract it step by step into the
`mathematical_spec` as named equations, even if not in equation form.

---

## Mathematical spec — ML-specific rules

**Probabilistic models:** always extract:
- Full joint distribution `p(x, z)` or generative model specification
- Variational family `q(z|x)` for VI methods
- ELBO or other objective with all KL terms written out
- Inference procedure: MCMC (which sampler?), VI (which approximation?), MAP

**Kernel methods:** extract the kernel function k(x, x') explicitly with all parameters.

**Boosting:** extract the weak learner update rule, shrinkage rate, and stopping criterion.

**Regularisation:** always extract regularisation type AND coefficient:
- L1 (Lasso), L2 (Ridge), Elastic Net — extract α, λ, ρ
- Record whether regularisation applies to all parameters or specific ones

---

## Training pipeline — ML-specific rules

"Training" in ML papers often means fitting, not gradient descent. Capture:
- Fitting procedure: closed-form, iterative (EM, coordinate descent), or stochastic
- Convergence criterion: tolerance, max iterations
- Cross-validation scheme used to select hyperparameters
- Hyperparameter search method (grid, random, Bayesian optimisation)

If the paper uses **scikit-learn** style APIs, extract the exact class name and parameters.

---

## Evaluation — ML-specific rules

**Always extract:**
- Cross-validation strategy (k-fold, stratified k-fold, LOOCV, time-series CV)
- Whether results are mean ± std over multiple runs/folds
- Statistical significance tests used (paired t-test, Wilcoxon, etc.) and p-value thresholds
- Whether comparison is against the same train/test split or uses different seeds
