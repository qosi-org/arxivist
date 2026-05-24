# Architecture Plan — Empirical Asset Pricing via Machine Learning
## Gu, Kelly, Xiu — Review of Financial Studies 2020

---

## Framework
**PyTorch** (neural networks) + **scikit-learn** (trees) + **NumPy/SciPy** (linear methods)  
Python 3.10+ | CPU sufficient (CUDA optional) | No GPU strictly required

---

## Module Hierarchy

```
src/asset_pricing_ml/
├── data/
│   ├── dataset.py          ← StockReturnDataset, PanelDataSplit
│   └── features.py         ← FeatureBuilder (Kronecker z_it = x_t ⊗ c_it)
├── models/
│   ├── linear.py           ← OLSModel, ElasticNetModel, PCRModel, PLSModel
│   ├── glm.py              ← GeneralizedLinearModel (spline + group lasso)
│   ├── trees.py            ← GBRTModel, RandomForestModel
│   └── neural_net.py       ← FeedForwardNN, NeuralNetModel (NN1–NN5)
├── training/
│   ├── losses.py           ← HuberLoss, L2Loss, WeightedL2Loss
│   └── trainer.py          ← NNTrainer, LinearTrainer
├── evaluation/
│   ├── metrics.py          ← ReturnMetrics, PortfolioMetrics
│   └── portfolios.py       ← PortfolioConstructor
└── utils/
    └── config.py           ← Config, ModelConfig, TrainingConfig
```

---

## 13 Models Implemented

| Model | Type | Regularization |
|-------|------|----------------|
| OLS | Linear | None |
| OLS-3 | Linear | Manual predictor selection |
| ENet | Linear | Elastic net (λ, ρ) |
| PCR | Dimension reduction | K components via SVD |
| PLS | Dimension reduction | K components via SIMPLS |
| GLM | Nonlinear (no interactions) | Group lasso on spline terms |
| RF | Tree ensemble | Random feature subsets + bagging |
| GBRT | Tree ensemble | Shrinkage ν + shallow trees |
| NN1 | Neural net [32] | L1 + early stop + batch norm + ensemble |
| NN2 | Neural net [32,16] | same |
| NN3 | Neural net [32,16,8] | same — **best performer** |
| NN4 | Neural net [32,16,8,4] | same |
| NN5 | Neural net [32,16,8,4,2] | same |

---

## NN3 Forward Pass (Primary Model)

```
z_it: [B, 920]  ← input features (Kronecker macro×char + industry dummies)
x1 = ReLU(BN(z  @ W0))  → [B, 32]
x2 = ReLU(BN(x1 @ W1))  → [B, 16]
x3 = ReLU(BN(x2 @ W2))  → [B, 8]
r_hat = x3 @ W3          → [B, 1]   ← predicted excess return
```

Ensemble: average predictions across 10 random seeds (count assumed — not stated in paper).

---

## Feature Construction

```
c_it [N, 94]  → cross-sectional rank → [-1, 1]  (replace missing with median)
x_t  [9]      → 8 Welch-Goyal macro predictors + constant
z_it [N, 920] = Kronecker(x_t, c_it_ranked) + 74 industry dummies
```

Publication lags enforced: monthly=1mo, quarterly=4mo, annual=6mo.

---

## Training Protocol

- **3-way temporal split**: Train 1957–1974 | Val 1975–1986 | Test 1987–2016
- **Annual refit**: training window grows +1 year each refitting; validation rolls forward
- **Hyperparameter tuning**: grid search over validation sample only
- **Optimizer**: Adam with learning rate shrinkage (Kingma & Ba 2014)
- **Regularization**: L1 penalty + early stopping + batch normalization + multi-seed ensemble

---

## Key Assumed Hyperparameters ⚠️

| Parameter | Assumed Value | Confidence | Note |
|-----------|--------------|-----------|------|
| Batch size | 512 | 0.45 | Not stated in paper |
| Ensemble seeds | 10 | 0.50 | Not stated |
| Early stopping patience | 5 epochs | 0.55 | Not stated |
| Adam lr | 0.001 | 0.65 | Standard default |
| L1 lambda grid | [1e-4, 0.01, 0.1] | 0.52 | Not stated |
| Spline knots | Quantile-based | 0.62 | Not specified |

---

## Evaluation Pipeline

1. **R²_oos**: benchmarked vs zero forecast (NOT historical mean)
2. **Diebold-Mariano test**: pairwise model comparison, Bonferroni-adjusted
3. **Variable importance**: R² reduction + SSD (sum of squared partial derivatives)
4. **Portfolio Sharpe ratios**: bottom-up forecasts + decile-sorted long-short
5. **Factor alphas**: FF5 + momentum 6-factor model

---

## Data Access ⚠️ HIGH RISK

The paper uses:
- **CRSP** monthly returns (subscription via WRDS)
- **94 stock characteristics** from Green, Hand, Zhang (2017) (partially public via Jeremiah Green's site)
- **Macro predictors** from Welch & Goyal (2008) (available free from Amit Goyal's website)

A **synthetic data generator** will be provided for development without WRDS access.
