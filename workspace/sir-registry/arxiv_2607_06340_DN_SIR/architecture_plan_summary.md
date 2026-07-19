# Architecture Plan Summary — arxiv_2607_06340

**Paper:** Signature-based identification of volatility models from path geometry (Burés & De Santiago, 2026)

## Framework
PyTorch for batched path simulation (optional GPU), `iisignature` for truncated path signatures, **XGBoost** as the actual classifier (matches the paper's real method — not a repurposed template this time). Plain YAML config.

## Module hierarchy (9 files)
- `simulators/heston.py`, `ou.py`, `rbergomi.py` — the 3 volatility path generators
- `features/signatures.py` — time-augmented truncated path signature (orders 3/4/5)
- `models/xgb_classifier.py` — XGBoost wrapper with the paper's exact hyperparameters (lr=0.05, depth=6, 500 estimators)
- `models/nn_baseline.py` — PyTorch MLP (128-64-32) for the Section 6.7 robustness check
- `evaluation/importance.py` — built-in gain importance + permutation importance
- `data/experiment_builder.py` — builds each of the paper's 9 named experiments from one config
- `utils/config.py` — YAML + seeding

## Entrypoints
- `train.py --experiment {5.1,5.2,5.3,6.1,6.2,6.3,...}` — run any of the paper's experiments
- `evaluate.py` — confusion matrix + accuracy from saved results
- `inference.py` — classify a small fresh batch
- `run_feature_importance.py` — reproduces Figure 6.4

## Key risks
1. **[High]** Rough Bergomi is simulated via Cholesky-based exact fBM instead of the paper's GPU hybrid scheme — mathematically equivalent, much slower at scale. Flagged clearly; swap-in point documented for a production run.
2. **[Medium]** Signature computation uses CPU-only `iisignature` instead of the paper's GPU-adapted code — correct, but slow at 250k-path scale. `signatory` (GPU) is the documented upgrade path.
3. **[Low]** Heston discretization scheme and correlation rho are unstated in the paper (rho argued irrelevant since only the variance path is used downstream).
4. **[Low]** NN baseline hyperparameters (Sec 6.7) are unstated; literature defaults used, low priority since it's a one-off comparison, not a headline result.

## Dependencies
`numpy`, `torch`, `iisignature`, `xgboost`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`. Dev: `pytest`, `black`, `ruff`, `jupyter`.
