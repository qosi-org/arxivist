# Deep Learning in Asset Pricing

Reproduction repository for **"Deep Learning in Asset Pricing"** by Luyang Chen, Markus Pelger and Jason Zhu (2019).

> **ArXivist confidence**: 0.85 overall | Architecture: 0.85 | Math: 0.94 | Training: 0.82

---

## Model Overview

The paper combines three neural network structures to estimate the **Stochastic Discount Factor (SDF)**:

| Component | Role | Hyperparameters |
|-----------|------|----------------|
| **StateMacroRNN** (LSTM) | 178 macro series → 4 hidden states h_t | SMV=4 |
| **SDFNetwork** (FFN) | [h_t, firm chars] → SDF weights ω | HL=2, HU=64 |
| **ConditionalMacroRNN** (LSTM) | 178 macro series → 32 hidden states h_t^g | CSMV=32 |
| **ConditionalNetwork** (FFN) | [h_t^g, firm chars] → instruments g | CHL=0, CHU=8 |
| **LoadingNetwork** (FFN) | [h_t, firm chars] → risk loadings β | same as SDF |

**Training**: 3-step GAN procedure (Section III.D) + ensemble of 9 models.

**Key result**: Out-of-sample annual Sharpe Ratio of **2.6** vs 0.22 for Fama-French 5.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e src/

# Smoke test (synthetic data, 2 epochs)
python train.py --config configs/config.yaml --debug

# Full training (requires CRSP data)
python train.py --config configs/config.yaml

# Evaluate
python evaluate.py --config configs/config.yaml --split test
```

---

## Data

This repository **does not include CRSP data** (proprietary subscription required).

- **CRSP returns**: Monthly equity returns 1967–2016 (~10,000 stocks)
- **FRED-MD**: 124 macroeconomic predictors (public — see `data/download.py`)
- **Firm characteristics**: 46 variables from CRSP/Compustat

Configure data paths in `configs/config.yaml` under the `data:` section.

---

## ⚠ Implementation Warnings

| Issue | Confidence | Location |
|-------|-----------|----------|
| LSTM internal hidden size assumed = output state dim | 0.65 | `lstm_encoder.py` |
| Training iterations/epochs not specified in paper | 0.55 | `trainer.py` |
| g() normalization assumed L2 | 0.60 | `conditional_network.py` |

See `sir-registry/.../sir.json` for the full Scientific Intermediate Representation.

---

## Reproducing Table III Results

| Model | SR (test, annual) | EV (test) | XS-R² (test) |
|-------|-------------------|-----------|--------------|
| **GAN** (ours) | **2.6** | **0.08** | **0.23** |
| FFN | 1.5 | 0.04 | 0.15 |
| EN | 1.7 | 0.04 | 0.19 |
| FF-5 | 0.8 | — | — |

*Monthly SRs from paper: GAN=0.75, FFN=0.44, EN=0.50, annualized ×√12.*

---

## Citation

```bibtex
@article{chen2019deep,
  title={Deep Learning in Asset Pricing},
  author={Chen, Luyang and Pelger, Markus and Zhu, Jason},
  year={2019},
  note={Working Paper, Stanford University}
}
```
