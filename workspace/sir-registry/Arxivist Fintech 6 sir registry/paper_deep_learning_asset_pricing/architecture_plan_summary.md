# Architecture Plan — Deep Learning in Asset Pricing
**paper_id**: paper_deep_learning_asset_pricing  
**Generated**: 2026-06-03

---

## Framework Selection

**Primary**: PyTorch 2.1+, Python 3.10+, CUDA 11.8+  
**Reasoning**: Finance/tabular domain; LSTM + custom loss is natural in PyTorch. No JAX/HuggingFace needed.  
**Config management**: YAML + OmegaConf (lightweight, no Hydra overhead)

---

## Module Hierarchy

```
paper-repos/paper_deep_learning_asset_pricing/
├── src/dlap/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_encoder.py        # StateMacroRNN, ConditionalMacroRNN
│   │   ├── sdf_network.py         # SDFNetwork (FFN omega), LoadingNetwork (FFN beta)
│   │   ├── conditional_network.py # ConditionalNetwork (adversarial g)
│   │   └── gan_model.py           # GANAssetPricingModel — top-level orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # CRSPDataset, unbalanced panel handling
│   │   └── transforms.py          # Cross-sectional quantile normalization, macro transforms
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py              # MomentConditionLoss, ExplainedVariationLoss
│   │   └── trainer.py             # GANTrainer — 3-step adversarial training loop
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py             # SharpeRatio, ExplainedVariation, CrossSectionalR2, VariableImportance
│   └── utils/
│       ├── __init__.py
│       └── config.py              # Config loading, seed utility, panel weighting
├── configs/
│   └── config.yaml
├── train.py
├── evaluate.py
├── inference.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── download.py
│   └── README_data.md
├── notebooks/
│   └── reproduce_paper_deep_learning_asset_pricing.ipynb
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml
└── README.md
```

---

## Tensor Flow — GAN Forward Pass

```
FORWARD PASS: GANAssetPricingModel
  macro_series: [T, 178] float32  ← input (macroeconomic time series)
  firm_chars:   [T, N, 46] float32  ← input (firm characteristics, quantile-normalized)
  returns:      [T, N] float32  ← input (excess returns R^e_{t+1})

  # SDF branch
  h_t = StateMacroRNN(macro_series)   → [T, 4]        # LSTM hidden states
  sdf_in = cat([h_t.expand, firm_chars]) → [T*N, 50]  # broadcast h_t to each stock
  omega = SDFNetwork(sdf_in)           → [T, N]        # SDF weights per stock
  F_t = (omega * returns).sum(dim=-1)  → [T]           # SDF factor (tangency portfolio)
  M_t = 1 - F_t                        → [T]           # SDF values

  # Adversarial branch
  h_t_g = ConditionalMacroRNN(macro_series) → [T, 32]
  cond_in = cat([h_t_g.expand, firm_chars]) → [T*N, 78]
  g = ConditionalNetwork(cond_in)            → [T, N, 8]  # 8 moment conditions

  # Loss
  loss = MomentConditionLoss(M_t, returns, g, panel_weights)  → scalar

  return omega, F_t, M_t, g, loss

FORWARD PASS: LoadingNetwork (separate)
  sdf_in: [T*N, 50]  (same as SDFNetwork input)
  beta_hat = LoadingNetwork(sdf_in)  → [T, N]  # proportional to E_t[R^e * F_{t+1}]
```

---

## Risk Assessment

| Risk | Severity | Description | Mitigation |
|------|----------|-------------|------------|
| LSTM hidden size ambiguity | Medium | Paper doesn't state internal LSTM units | Config flag `lstm_hidden_size`; default = output states |
| Training iterations | Medium | No epoch/step count given | Early stopping on validation SR with patience=20 |
| Data access | High | CRSP is proprietary; FRED-MD is public | Synthetic data generator for unit tests; data README |
| g() normalization | Low | Exact normalization unclear | L2 normalize; config flag to try alternatives |
| Memory: T*N panel | Medium | ~10k stocks * 600 months = 6M entries | Process in time-batches; configurable chunk_size |
| Unbalanced panel weighting | Low | sqrt(T_i/T) weighting specified | Implemented directly from Eq. (3) |
