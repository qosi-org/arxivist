# Domain: ML — Architecture Patterns (Stage 3 Enrichment)

Load alongside `agents/03_architecture_planner.md` when domain is **ML**.

---

## Framework selection for ML papers

**Default framework: scikit-learn** for classical ML.
**Use PyTorch/JAX** only if the paper involves gradient-based training of parameterised models.
**Use Stan or PyMC** for Bayesian inference papers with MCMC.
**Use GPyTorch or GPflow** for Gaussian Process papers.
**Use XGBoost or LightGBM** directly for boosting papers — do not reimplement.

Never use PyTorch to reimplement a classical algorithm that has a mature scikit-learn implementation.

## Module hierarchy for ML

Replace the standard `src/{project}/models/` layout with:

```
src/{project}/
├── model.py          ← Main model class (sklearn-compatible where possible)
├── kernels.py        ← Kernel functions (for GP/SVM papers)
├── inference.py      ← MCMC samplers, VI routines (for probabilistic papers)
├── features.py       ← Feature extraction / preprocessing
├── data/
│   ├── dataset.py
│   └── cross_validation.py   ← CV strategy from the paper
├── evaluation/
│   └── metrics.py
└── utils/
    └── config.py
```

## Sklearn-compatible interface standard

All model classes must implement:
```python
def fit(self, X, y) -> Self
def predict(self, X) -> np.ndarray
def score(self, X, y) -> float  # using paper's primary metric
```

And optionally:
```python
def predict_proba(self, X) -> np.ndarray   # probabilistic models
def sample(self, X, n_samples) -> np.ndarray  # generative/Bayesian models
```
