# Domain: ML — Code Conventions (Stage 4 Enrichment)

Load alongside `agents/04_code_generator.md` when domain is **ML**.

---

## Implementation standards

**Reproducibility in ML is seed-sensitive in different ways than deep learning:**
```python
import numpy as np
from sklearn.utils import check_random_state

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    # No torch seeding unless the model uses PyTorch
```

**Cross-validation must be implemented exactly as described:**
- Stratified vs non-stratified matters for class-imbalanced datasets
- Time-series CV must never use future data — use `TimeSeriesSplit`
- Always fix the random state of the CV splitter

**Hyperparameter config:**
```yaml
model:
  # All parameters extracted from paper — mark ASSUMED where not stated
  n_estimators: 100       # ASSUMED: common default if paper silent
  max_depth: null         # null = sklearn default (unlimited)
  random_state: 42

evaluation:
  cv_folds: 5
  cv_strategy: stratified  # or time_series, group, etc.
  metric: accuracy
  n_seeds: 5              # number of random seeds to average over
```

**Requirements for ML papers** (replace deep learning stack):
```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
# Conditional on model type:
# xgboost>=2.0.0
# lightgbm>=4.0.0
# gpytorch>=1.11.0
# pymc>=5.0.0
```

**No Docker GPU requirement** — most ML models run on CPU. Use a lightweight base image:
`FROM python:3.11-slim`

## What Must NOT be done

- Do NOT use PyTorch DataLoader for tabular data — use numpy arrays or pandas DataFrames
- Do NOT implement sklearn algorithms from scratch — use the library
- Do NOT ignore statistical significance — always report mean and std over multiple seeds
