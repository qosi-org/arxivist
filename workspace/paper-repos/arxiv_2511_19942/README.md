# Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning — reproduction

**Paper:** Gai, Zeng, Zhang, Raghunathan (2025). [arXiv:2511.19942](https://arxiv.org/abs/2511.19942)

## What this paper does (plain English)

RL fine-tuning of LLMs (e.g. via GRPO) tends to make models "sharpen" — collapse onto one narrow
answer pattern, even sometimes a wrong one, losing the diversity of correct approaches they had
before fine-tuning. This paper proves *why* this happens (a selection-and-reinforcement bias) and
proposes **differential smoothing**: shift the reward down a bit for correct trajectories the
model is already very confident about, and up a bit for incorrect trajectories it's unsure about.
This keeps correctness high while preventing the collapse.

## What's reproduced here — and what's scaled down

This repo reproduces the paper's **Countdown task** experiment (a synthetic arithmetic puzzle:
reach a target number from 3-4 given integers using +, -, *, /), comparing vanilla GRPO against
DS-GRPO (this paper's method).

**Scoped down from the paper for free-tier Colab/Kaggle GPU compute:**
- Model: **Qwen2.5-0.5B-Instruct** (paper used Qwen2.5-3B-Instruct) + LoRA + 4-bit quantization
- Training steps: ~300 (paper trains "until reward saturates," no fixed count given)
- These are documented, deliberate substitutions — see `sir-registry/arxiv_2511_19942/sir.json`
  (`implementation_assumptions`) and `architecture_plan.json` (`risk_assessment`) for full reasoning.

## Quick start

```bash
pip install -e .

# Baseline: vanilla GRPO
python train.py --config configs/config.yaml --use_differential_smoothing false

# This paper's method: DS-GRPO
python train.py --config configs/config.yaml --use_differential_smoothing true

# Quick sanity check before a full run (2 steps, tiny batch)
python train.py --config configs/config.yaml --debug

# Evaluate a trained checkpoint
python evaluate.py --checkpoint checkpoints/ds_grpo --config configs/config.yaml

# Try a single puzzle
python inference.py --checkpoint checkpoints/ds_grpo --numbers 4,7,2 --target 15
```

Or open `notebooks/reproduce_arxiv_2511_19942.ipynb` (added in Stage 5) for a guided walkthrough.

## Reproducibility notes / known deviations

- **Training hyperparameters (optimizer, learning rate, step count) are assumed defaults**, not
  stated in the paper (SIR confidence 0.35 on this section). We use AdamW at 1e-6 — standard for
  this class of RL fine-tuning, but unverified against the paper's actual choice.
- **The Countdown verifier is a from-scratch reimplementation** (paper doesn't specify its exact
  logic) — parses model output as a safe arithmetic expression (no `eval()` on model output) and
  checks it uses the given numbers exactly once and reaches the target.
- **Model size (0.5B vs. paper's 3B)** is a deliberate compute-budget substitution — the underlying
  mechanism should still be testable at this scale, but absolute numbers won't match the paper's.
- Run `train.py --dry-run` first to confirm all components build; `--debug` for a 2-step sanity
  check before committing to a full run — this catches hyperparameter/memory issues cheaply.

## Citation

```bibtex
@article{gai2025differential,
  title={Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning},
  author={Gai, Jingchu and Zeng, Guanning and Zhang, Huaqing and Raghunathan, Aditi},
  journal={arXiv preprint arXiv:2511.19942},
  year={2025}
}
```
