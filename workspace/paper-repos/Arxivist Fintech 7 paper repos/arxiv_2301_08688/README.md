# APEX LOB Trader

**ArXivist-generated implementation of:**

> *Asynchronous Deep Double Duelling Q-Learning for Trading-Signal Execution in Limit Order Book Markets*
> Peer Nagy, Jan-Peter Calliess, Stefan Zohren — [arXiv:2301.08688](https://arxiv.org/abs/2301.08688) (2023)

---

## What this paper does

Translates a noisy directional price signal into a limit order trading strategy using deep RL. An APEX (asynchronous prioritised experience replay) agent with a Duelling Double DQN architecture learns to place individual limit orders — choosing between passive (bid/ask) and aggressive (cross-spread) placement — to maximise trading returns on NASDAQ AAPL order book data.

The agent outperforms a heuristic baseline on all three signal noise levels tested, with outperformance of 14.8–32.2 percentage points over 31 out-of-sample test episodes. This is the first application of APEX to LOB environments.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd arxiv_2301_08688
pip install -e .

# 2. (Optional) Get real LOBSTER data — see data/README_data.md
#    Without it, synthetic data is used automatically.

# 3. Debug run (200 steps, validates setup)
python train.py --config configs/config.yaml --debug

# 4. Dry run (build all components, no training)
python train.py --config configs/config.yaml --dry-run

# 5. Full training
python train.py --config configs/config.yaml

# 6. Evaluate
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/final.pt
```

---

## Installation

**Pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

**Conda:**
```bash
conda env create -f environment.yaml
conda activate apex-lob-trader
```

**Docker:**
```bash
cd docker
docker-compose up train
```

---

## Training

```bash
# Default (mid noise, a=1.3)
python train.py --config configs/config.yaml

# High noise (a=1.1)
python train.py --config configs/config.yaml  # edit signal.noise_level_a in config

# Resume from checkpoint
python train.py --config configs/config.yaml --resume checkpoints/checkpoint_step50000.pt

# Debug mode (200 steps)
python train.py --config configs/config.yaml --debug
```

**Full distributed APEX** (42 workers, as in paper): Use RLlib with the parameters in `configs/config.yaml`. Requires a machine with sufficient CPU cores and 1 GPU.

---

## Evaluation

```bash
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/final.pt
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/final.pt --noise-level 1.1
```

---

## Expected Results (from paper, Section 5)

| Noise Level | RL μ | RL Sharpe | Baseline μ | Baseline Sharpe | Outperf (pp) |
|---|---|---|---|---|---|
| a=1.1 (high) | 0.00 | −0.72 | −0.07 | −5.68 | **32.2** |
| a=1.3 (mid) | 0.11 | 7.34 | 0.00 | −0.04 | **14.8** |
| a=1.6 (low) | 0.21 | 14.69 | — | — | **20.7** |

All RL vs. baseline differences are statistically significant (p << 0.1, t-test, 31 episodes).

---

## Repository Structure

```
.
├── src/apex_lob_trader/
│   ├── models/q_network.py          # Duelling DQN (Section 3.2)
│   ├── data/
│   │   ├── signal_generator.py      # Synthetic signal (Section 4.1, Eq. 1–2)
│   │   └── lob_dataset.py           # LOBSTER data loader
│   ├── training/
│   │   ├── environment.py           # OpenAI Gym LOB env (Section 4.2)
│   │   ├── trainer.py               # APEX DQN trainer (Section 3.2, Table 2)
│   │   └── replay_buffer.py         # Uniform replay buffer
│   ├── evaluation/
│   │   ├── baseline.py              # Heuristic baseline (Section 5)
│   │   └── metrics.py               # mu, Sharpe, CI, outperformance
│   └── utils/config.py              # Config loading + seed utilities
├── configs/config.yaml              # All hyperparameters with confidence notes
├── train.py                         # Training entrypoint
├── evaluate.py                      # Evaluation entrypoint
├── inference.py                     # Single-step inference demo
├── notebooks/
│   └── reproduce_arxiv_2301_08688.ipynb  # Reproduction notebook
├── docker/Dockerfile
├── data/README_data.md              # LOBSTER data instructions
└── requirements.txt
```

---

## Reproducibility Notes

### Known deviations and low-confidence assumptions

| Parameter | Paper value | Assumed value | Confidence | Notes |
|---|---|---|---|---|
| `hidden_dim` | Not stated | 256 | 0.55 | Common for finance RL DQN |
| `w_dir_initial` | Not stated | 0.5 | 0.40 | Must start > 0 for curriculum |
| `psi` (w_dir decay) | Not stated | 0.9999 | 0.40 | Must reach ~0 over 300M steps |
| `kappa` | Not stated | 1.0 | 0.40 | Directional reward scale |

### Data

LOBSTER data is **proprietary and commercial**. Synthetic random-walk data is available automatically for testing. See `data/README_data.md`.

### Compute

The paper trains for 300M timesteps with 42 parallel workers and 1 GPU. Full replication requires significant compute (estimated 24–72 hours on a 32-core machine + A100).

---

## Citation

```bibtex
@article{nagy2023asynchronous,
  title={Asynchronous Deep Double Duelling Q-Learning for Trading-Signal Execution in Limit Order Book Markets},
  author={Nagy, Peer and Calliess, Jan-Peter and Zohren, Stefan},
  journal={arXiv preprint arXiv:2301.08688},
  year={2023}
}
```

---

*Generated by ArXivist — arXiv:2301.08688 — SIR confidence: 0.81*
