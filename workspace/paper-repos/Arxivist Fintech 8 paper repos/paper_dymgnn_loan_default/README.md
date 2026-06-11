# DYMGNN — Dynamic Multilayer Graph Neural Network for Loan Default Prediction

**ArXivist-generated implementation of:**

> *Attention-based dynamic multilayer graph neural networks for loan default prediction*
> Sahab Zandi, Kamesh Korangi, María Óskarsdóttir, Christophe Mues, Cristián Bravo
> European Journal of Operational Research 321 (2025) 586–599
> DOI: [10.1016/j.ejor.2024.09.025](https://doi.org/10.1016/j.ejor.2024.09.025)

---

## What this paper does

Predicts 1-year-ahead mortgage loan default by combining **Graph Neural Networks** (GCN or GAT) for spatial borrower relationships with **Recurrent Neural Networks** (LSTM or GRU) for temporal dynamics, applied to a **dynamic multilayer network** built from Freddie Mac US mortgage data.

Two layers capture different borrower connections:
- **Area layer**: borrowers in the same geographic region (first 2 zip-code digits)
- **Company layer**: borrowers using the same mortgage provider

A **custom soft attention mechanism** over time snapshots lets the model weight recent months more heavily. The best configuration (**GAT-LSTM-ATT** on the double-layer network) achieves **AUC=0.812, F1=0.851** — outperforming XGBoost, LR, DNN, and static GNN baselines.

---

## Quick Start

```bash
# 1. Install
git clone <repo>
cd paper_dymgnn_loan_default
pip install -e .

# 2. Debug run (5 epochs, synthetic data, validates pipeline)
python train.py --config configs/config.yaml --debug

# 3. Dry run (build all components, no training)
python train.py --config configs/config.yaml --dry-run

# 4. Full training (best config: GAT-LSTM-ATT double layer)
python train.py --config configs/config.yaml

# 5. Evaluate
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
```

---

## Installation

```bash
pip install -r requirements.txt && pip install -e .
# or
conda env create -f environment.yaml && conda activate dymgnn
# or
cd docker && docker-compose up train
```

---

## Model Variants

All 8 configurations from Tables 5–7:

| Config | GNN | RNN | Attention | Best AUC (double layer) |
|--------|-----|-----|-----------|------------------------|
| **GAT-LSTM-ATT** ★ | GAT | LSTM | ✓ | **0.812 ± 0.008** |
| GCN-LSTM-ATT | GCN | LSTM | ✓ | 0.810 ± 0.009 |
| GAT-GRU-ATT | GAT | GRU | ✓ | 0.804 ± 0.006 |
| GCN-GRU-ATT | GCN | GRU | ✓ | 0.793 ± 0.011 |
| GAT-LSTM | GAT | LSTM | ✗ | 0.807 ± 0.008 |
| GCN-LSTM | GCN | LSTM | ✗ | 0.806 ± 0.010 |
| GAT-GRU | GAT | GRU | ✗ | 0.800 ± 0.004 |
| GCN-GRU | GCN | GRU | ✗ | 0.789 ± 0.010 |

★ Best configuration — default in `configs/config.yaml`

---

## Repository Structure

```
.
├── src/dymgnn/
│   ├── models/
│   │   ├── gcn_layer.py         # GCN (Section 3.2, Eq. 1)
│   │   ├── gat_layer.py         # GAT (Section 3.2, Eq. 2–4)
│   │   ├── temporal_attention.py # Soft attention (Section 3.5, Eq. 16–18)
│   │   ├── decoder.py           # FF decoder (Section 3.6, Fig. 5)
│   │   └── dymgnn.py            # Full DYMGNN (Sections 3.4–3.5)
│   ├── data/
│   │   └── dataset.py           # Freddie Mac loader + graph builder (Section 4)
│   ├── training/
│   │   ├── losses.py            # BCE loss (Section 3.6, Eq. 19)
│   │   └── trainer.py           # Adam, early stopping (Table C.1)
│   ├── evaluation/
│   │   ├── metrics.py           # AUC, F1, 95% bootstrap CI (Section 4.3)
│   │   └── baselines.py         # LR, XGB, DNN, Static GCN/GAT (Section 5.1)
│   └── utils/config.py          # Config loading + seed utilities
├── configs/config.yaml          # All hyperparameters with confidence annotations
├── train.py                     # Training entrypoint
├── evaluate.py                  # Evaluation entrypoint
├── notebooks/reproduce_dymgnn.ipynb
├── docker/
├── data/README_data.md          # Freddie Mac data access instructions
└── requirements.txt
```

---

## Expected Results (Tables 3–7)

### Best performing model: GAT-LSTM-ATT (double layer network)
| Metric | This model | XGBoost (best baseline) | Gain |
|--------|:---------:|:-----------------------:|:----:|
| AUC | 0.812 ± 0.008 | 0.805 ± 0.018 | +0.87% |
| F1  | 0.851 ± 0.007 | 0.837 ± 0.012 | +1.67% |

### Key findings
- Dynamic models > Static GNN baselines (most pronounced difference)
- Double-layer > Single-layer (richer connectivity, tighter CIs)
- With attention > Without attention (consistently across all configs)
- GAT > GCN; LSTM > GRU for this task
- Attention weights peak at timestamp 6 (most recent snapshot, Fig. 9)

---

## Reproducibility Notes

| Parameter | Paper | Assumed | Confidence |
|-----------|-------|---------|-----------|
| Embedding dim D | Not stated | 64 | 0.45 |
| GAT heads | Not stated | 4 | 0.50 |
| LSTM layers | Not stated | 1 | 0.68 |
| Decoder sizes | Fig. 5: 20→10 | 20→10 | 0.92 |
| Decoder dropout | Fig. 5: 0.5 | 0.5 | 0.97 |

**Data**: Freddie Mac SFLL requires free registration. Synthetic fallback auto-activates for testing. See `data/README_data.md`.

**Compute**: Paper used AMD Milan 7413 + NVidia A100 40GB (Table C.2). Longest config (GAT-LSTM-ATT double layer): ~12,120s training time (Table 9).

---

## Citation

```bibtex
@article{zandi2025attention,
  title={Attention-based dynamic multilayer graph neural networks for loan default prediction},
  author={Zandi, Sahab and Korangi, Kamesh and {\'O}skarsd{\'o}ttir, Mar{\'\i}a and Mues, Christophe and Bravo, Crist{\'i}an},
  journal={European Journal of Operational Research},
  volume={321},
  pages={586--599},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.ejor.2024.09.025}
}
```

---

*Generated by ArXivist — SIR confidence: 0.87*
