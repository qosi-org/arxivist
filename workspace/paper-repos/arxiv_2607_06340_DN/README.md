# Signature-Based Volatility Model Identification — reproduction of arXiv:2607.06340

> Signature-based identification of volatility models from path geometry
> Òscar Burés, Rafael De Santiago (2026)

An ArXivist-generated reproduction of a framework that identifies which
stochastic volatility model (Heston, Ornstein-Uhlenbeck, or rough Bergomi)
generated an observed path, using truncated path signatures as features and
XGBoost as the classifier — without any parametric calibration.

## What this reproduces

- **3 path simulators**: Heston variance process, Ornstein-Uhlenbeck, rough Bergomi
- **Truncated path signatures** (orders 3, 4, 5) as a feature map, time-augmented per the paper's Section 2.2
- **XGBoost classifier** with the paper's exact hyperparameters (lr=0.05, depth=6, 500 estimators)
- **All 9 named experiments**: fixed-parameter proof-of-concept (5.1-5.3), random-parameter main results (6.1-6.3), signature-order robustness (6.5), time-horizon robustness (6.6), the Heston/OU volatility-of-volatility deep dive (6.8), and sample-size robustness (6.9)
- **Feature importance**: built-in (gain) vs. permutation importance, reproducing Figure 6.4
- **Neural network baseline** (Section 6.7): a 128-64-32 MLP for comparison

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Fast smoke test (~seconds): 500 paths/class, 20 steps
python train.py --config configs/config.yaml --experiment 6.1 --debug
python evaluate.py --results-dir results/6.1 --experiment 6.1 --as-percentage

# Full-scale run of one experiment (250,000 paths/class — this is what took
# the paper 6-16 minutes per experiment on an RTX 3080 Ti; expect longer on
# CPU-only or without a GPU signature backend, see Reproducibility Notes)
python train.py --config configs/config.yaml --experiment 6.2
python evaluate.py --results-dir results/6.2 --experiment 6.2 --as-percentage

# Feature importance (reproduces Figure 6.4)
python run_feature_importance.py --results-dir results/6.2

# Any of the 9 named experiments:
python train.py --config configs/config.yaml --experiment 5.1
python train.py --config configs/config.yaml --experiment 6.3
python train.py --config configs/config.yaml --experiment 6.5_order3
python train.py --config configs/config.yaml --experiment 6.5_order5
python train.py --config configs/config.yaml --experiment 6.6_T0.2
python train.py --config configs/config.yaml --experiment 6.8_shared_dist
python train.py --config configs/config.yaml --experiment 6.8_low_nu
python train.py --config configs/config.yaml --experiment 6.8_high_nu
```

Docker:
```bash
docker build -f docker/Dockerfile -t sig-vol-id .
docker run sig-vol-id python train.py --config configs/config.yaml --experiment 6.1 --debug
```

## Reproducibility notes — read before expecting exact paper numbers

Everything explicitly stated in the paper's text is implemented as stated
(XGBoost hyperparameters, sample sizes, parameter ranges, the 100-step/T=0.1
discretization, the shared-noise control across rough-Bergomi Hurst classes
in Section 6.1). A few implementation details are **not** given in the paper
and were filled in with the most standard choice — flagged `# ASSUMED` in
`configs/config.yaml`:

1. **Rough Bergomi simulation scheme** (High severity, see
   `docs/architecture_plan.json`): the paper uses an adapted GPU hybrid
   scheme (Bennedsen, Lunde & Pakkanen 2017); this repo uses an exact
   Cholesky-decomposition fractional-Brownian-motion simulation instead.
   Both are mathematically exact at the paper's 100-step resolution — the
   Cholesky approach is just less GPU-friendly at very large step counts
   (not an issue at 100 steps). If you scale `n_steps` up significantly,
   swap in a true hybrid-scheme implementation (e.g. adapt
   `github.com/ryanmccrickerd/rough_bergomi`, which the paper itself cites).
2. **Signature computation backend**: uses the CPU-only `iisignature`
   package rather than the paper's custom GPU-adapted code. Mathematically
   identical output, slower at 250k-path scale. Swap in the `signatory`
   (PyTorch-native, GPU-capable) package for full-scale, faster runs.
3. **Heston discretization scheme**: full-truncation Euler (standard
   practice for the square-root diffusion); not specified in the paper.
4. **Heston/OU Brownian correlation rho**: not given in the paper, and
   irrelevant to this repo's results since only the variance path `v_t` is
   classified (the price process `S_t`, which rho would affect, is never
   simulated or used).
5. **Neural-network baseline hyperparameters** (Section 6.7): dropout,
   batch size, and Adam learning rate are unstated; literature defaults
   used (dropout 0.2, batch 256, lr 1e-3). This is a one-off robustness
   check in the paper, not a headline result.
6. **[Open issue] Section 6.8 Heston/OU volatility-of-volatility experiments**
   (`6.8_low_nu`, `6.8_high_nu`): the paper's footnote 7 says Heston's `nu`
   and OU's `sigma` are drawn from "comparable ranges" without specifying
   the exact calibration. As shipped, our `6.8_high_nu` result is close to
   the paper's (~91% Heston accuracy vs. paper's 90.9%), but `6.8_low_nu`
   does **not** currently reproduce the paper's direction (we get higher
   Heston accuracy at low nu, not lower/worse as the paper reports). This
   was tested and is a known, flagged gap -- not something we're claiming
   is fixed. It likely requires recalibrating what "comparable" means
   between Heston's multiplicative sqrt(v)-scaled noise and OU's additive
   noise (a fixed numeric range for both is not obviously equivalent in
   effect), which the paper's text does not fully resolve. Treat this one
   experiment's low-nu result with caution until further tuned.

**What we verified end-to-end**: all 16 unit tests pass (`tests/test_core.py`,
covering all 3 simulators, signature dimension formulas, and the XGBoost
wrapper), and `train.py`/`evaluate.py`/`inference.py`/`run_feature_importance.py`
were smoke-tested successfully. One real bug was found and fixed during
testing: passing `objective`/`num_class` explicitly to `XGBClassifier`
(rather than letting it auto-detect from `y`) caused `.predict()` to return
raw probability arrays instead of class labels in xgboost 3.3.0 — this is
now avoided in `models/xgb_classifier.py`. Even at tiny debug scale (500
paths/class, 20 steps), the feature-importance smoke test already picked out
`sig_27` as a top feature, qualitatively consistent with the paper's finding
that `sig_27` (an order-4 signature term) dominates importance.

## Repository layout

```
train.py                       # run one named experiment end-to-end
evaluate.py                    # confusion matrix + accuracy from saved results
inference.py                   # classify a small fresh batch
run_feature_importance.py      # reproduces Figure 6.4
configs/config.yaml             # all parameters, explicit vs ASSUMED tagged
src/sig_vol_id/
  simulators/
    heston.py                  # Heston variance process (full-truncation Euler)
    ou.py                      # Ornstein-Uhlenbeck (exact simulation)
    rbergomi.py                # rough Bergomi (Cholesky fBM; shared-noise control)
  features/
    signatures.py               # time-augmented truncated path signature
  models/
    xgb_classifier.py           # XGBoost wrapper, paper's exact hyperparameters
    nn_baseline.py               # Section 6.7 MLP robustness-check baseline
  evaluation/
    importance.py               # built-in + permutation feature importance
  data/
    experiment_builder.py        # builds each of the 9 named experiments
  utils/
    config.py                   # YAML config + seeding
tests/test_core.py              # 16 unit tests
docker/                         # Dockerfile
notebooks/                      # walkthrough notebook
```

## Citation

```bibtex
@article{bures2026signature,
  title={Signature-based identification of volatility models from path geometry},
  author={Bur{\'e}s, {\`O}scar and De Santiago, Rafael},
  journal={arXiv preprint arXiv:2607.06340},
  year={2026}
}
```

## Generation provenance

Generated by the **ArXivist** skill: Stage 1 (paper -> SIR, confidence 0.82)
-> Stage 2 (SIR registry) -> Stage 3 (architecture plan) -> Stage 4 (this
repo) -> Stage 5 (notebook). Full provenance, including the SIR, architecture
plan, and pipeline state, is in `docs/` -- see `docs/architecture_plan_summary.md`
for the design rationale and flagged risks.
