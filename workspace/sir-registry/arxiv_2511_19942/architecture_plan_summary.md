# Architecture Plan Summary — Differential Smoothing (arxiv_2511_19942)

## What we're building
A small, Kaggle/Colab-friendly reproduction of the paper's **Countdown task** experiment: train a
small instruct LLM with GRPO (baseline) vs. DS-GRPO (the paper's fix) and compare Pass@K /
Solution Multiplicity between the two.

## Key scope-down decisions (read this before running anything)
- **Model**: Qwen2.5-0.5B-Instruct instead of the paper's Qwen2.5-3B-Instruct, loaded in 4-bit with
  LoRA adapters. This is a deliberate compute-budget substitution, not an oversight — documented as
  an assumption with confidence 0.3. If a bigger free GPU is available, bump to 1.5B or the paper's
  actual 3B.
- **Training steps**: ~300 steps instead of "until reward saturates" (paper gives no fixed count).
- **Library choice**: uses HuggingFace TRL's `GRPOTrainer` as a base rather than re-implementing
  GRPO from scratch — the paper's actual contribution (differential smoothing) is implemented as a
  focused hook on top of that, not the whole RL loop.

## What's genuinely novel here vs. boilerplate
The **only** code that encodes the paper's actual idea is `rewards/differential_smoothing.py`
(Eq. 4/6) and `training/grpo_advantage.py` (Eq. 5). Everything else — model loading, the Countdown
verifier, the training loop wrapper — is standard RL-fine-tuning plumbing.

## Biggest risks (see architecture_plan.json → risk_assessment for full list)
1. **Memory** — RL fine-tuning needs the policy model + reference copy + rollout samples all in
   memory at once; even scoped down this might be tight on a free T4/P100.
2. **Assumed hyperparameters** (LR, optimizer, step count) might need adjusting if training doesn't
   behave sensibly — plan includes a `--dry-run`/`--debug` path to catch this early, cheaply.
3. **Home-built Countdown verifier** — must be unit-tested before trusting any training signal from
   it, since the paper doesn't specify its exact validation logic.

## Entrypoints
`train.py --config configs/config.yaml --use_differential_smoothing [true|false]` — same script
trains either the baseline or the fix, so results are directly comparable.
`evaluate.py` and `inference.py` for post-training checks.
