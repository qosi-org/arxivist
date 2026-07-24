# Verification Log — Caduceus (arxiv_2403_003234)

## Run metadata
- Architecture verification: 2026-07-24, local CPU (torch 2.5.0+cpu) + Colab
- Downstream fine-tuning: 2026-07-24, Colab T4 GPU (torch 2.11+cu128, transformers 4.44.2, mamba-ssm)
- SIR version used: 1
- Manual review required: No

## Part A — architecture verification (executed)

```
pytest tests/ -q      # 7 passed (local CPU and Colab)
```
Direct equivariance probe:
```
MambaDNA output mean-abs: 0.0608
RC-equivariance gap (max abs): 0.00e+00     # Theorem 3.1 holds exactly
```

## Part B — downstream fine-tuning (executed)

Commands:
```
python data/download.py --benchmark genomic_benchmarks --task human_nontata_promoters
python train.py --config configs/config.yaml
```

Model load:
```
kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3
params = 0.47M | device=cuda | amp=True
LOAD REPORT: mamba_rev.{in,out}_proj MISSING (intentionally tied for Caduceus-Ph; newly init) — expected
```

Result:
```
epoch 1  acc 0.8443
epoch 5  acc 0.9141
epoch 10 acc 0.9480   <- best
[done] best validation metrics: {'accuracy': 0.9479743192384326}
```

Paper (Caduceus-Ph, Table 1): **0.946**. Deviation **+0.21%** (Excellent / matched).

## Environment note (reproducibility)

Colab's very new transformers broke the Hub's Caduceus modeling code during load-finalize
(`tie_weights()` / `all_tied_weights_keys` API changes). Resolved by pinning **transformers==4.44.2**
(now reflected in requirements.txt / setup.py / environment.yaml as `>=4.38,<4.46`). A best-effort
compat shim also lives in `src/caduceus/__init__.py` for newer versions. mamba-ssm / causal-conv1d
were built from source against torch 2.11 with `--no-build-isolation`.

## Config
- Stock `configs/config.yaml` (Caduceus-Ph, lr 1e-3 `# ASSUMED`, mean-pool, RC aug + conjoin, 10 epochs).
- No user modifications recorded.

## Integrity
- The reported 0.9480 is the actual `[done]` output from the user's Colab run. No fabricated metrics.
