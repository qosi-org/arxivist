# Changelog

## v1.0 (2026-07-12) — ArXivist initial generation
- Generated from SIR v1 / architecture_plan v1 for arXiv:2505.01575v3.
- Implements all 6 model families (PretrainedTransformer[+LNF], StandardTransformer,
  SERT[+LNF], EncoderOnlyTransformer), rolling-window training, OOS R2/MSE (corrected
  v3 methodology), Diebold-Mariano HAC test, and sign-signal/softmax-filtered
  backtesting (equal/value-weighted, static/dynamic transaction cost).
- Fixed during smoke-testing (pre-release): teacher-forcing shift-right logic in
  `training/trainer.py::_forward` originally concatenated tensors along the time axis
  instead of shifting in place, doubling sequence length and causing a shape-mismatch
  crash on the first backward pass. Corrected to an in-place `y_shifted[:, 1:, :] =
  y[:, :-1, :]` assignment. Verified via `pytest tests/test_smoke.py` (10/10 passing)
  and an end-to-end `python train.py --config configs/config.yaml --debug` run.
- Fixed during smoke-testing: `train.py --debug` mode's dimension-reduction override
  could produce a `d_model` not divisible by `num_heads`; now rounds `d_model` down to
  the nearest multiple of `num_heads`.
