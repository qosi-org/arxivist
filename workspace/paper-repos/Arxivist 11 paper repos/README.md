# Asset Pricing in Pre-trained Transformers

**ArXivist-generated reproduction repository**

- **Paper**: *Asset Pricing in Pre-trained Transformers*
- **Author**: Shanyan Lai (Department of Economics and Related Studies, University of York)
- **arXiv**: [2505.01575v3](https://arxiv.org/abs/2505.01575) (q-fin.CP)
- **Generated**: 2026-07-12 by ArXivist

## What this paper does

The paper proposes two Transformer-family model innovations for U.S. large-cap stock
return prediction and factor investing: **SERT** (Single-directional Encoder
Representations from Transformer — an encoder-only, causally-masked, MLP-autoencoder-
pretrained BERT variant) and a **pre-trained Transformer** (a full encoder-decoder
Transformer with the same MLP-autoencoder pre-training module). Both are benchmarked
against standard (non-pretrained) Transformer and encoder-only Transformer models across
three periods spanning the COVID-19 pandemic. The proposed models achieve the highest
out-of-sample R² (up to ~11.9%) and Sortino ratios roughly 47% (equal-weighted) / 28%
(value-weighted) higher than a buy-and-hold benchmark during the pandemic period.

## Quick start

```bash
git clone <this-repo> && cd arxiv_2505_01575
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Smoke-test the whole pipeline on synthetic data (no real dataset needed):
python train.py --config configs/config.yaml --debug

# Once you have real data (see data/README_data.md):
python data/download.py                 # checks / instructs on obtaining data
python train.py --config configs/config.yaml
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --period 2212
python backtest.py --config configs/config.yaml --checkpoint checkpoints/best.pt --weighting equal --tc-mode static
```

## Installation

### pip
```bash
pip install -r requirements.txt
pip install -e .
```

### conda
```bash
conda env create -f environment.yaml
conda activate sert-asset-pricing
```

### Docker
```bash
docker compose -f docker/docker-compose.yml up train
docker compose -f docker/docker-compose.yml up notebook   # Jupyter on :8888
```

## Training

```bash
python train.py --config configs/config.yaml [--resume checkpoints/best.pt] [--seed 42] [--debug] [--dry-run]
```
`configs/config.yaml` defaults to the paper's best-performing variant, `P_Trans_H3`
(pre-trained Transformer, 3 heads). Switch `model.family` to `sert`, `sert_lnf`,
`standard_transformer`, `encoder_only_transformer`, or `pretrained_transformer_lnf` to
reproduce the other five model families in Table 2 of the paper.

## Evaluation

```bash
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --period {1911,2112,2212}
```
Computes the paper's *corrected* (v3 revision) out-of-sample R² using the
Campbell-Thompson-style denominator over the historical train+validation mean
(Appendix E).

## Backtesting

```bash
python backtest.py --config configs/config.yaml --checkpoint checkpoints/best.pt \
    --weighting {equal,value} --tc-mode {static,dynamic} [--softmax-filter]
```
Reproduces Section 5.3's sign-signal (trend-following) and softmax-filtered
backtests, with static (50bps) or dynamic (turnover × 20bps) transaction costs,
reporting annualized return, Sharpe, Sortino, max drawdown, and turnover.

## Expected results (from the paper, Table 3/5/16/17)

| Model | Period | OOS R² | Notes |
|---|---|---|---|
| P_Trans_H3 | 2112 (COVID-inclusive) | 0.1113 | Best pre-trained Transformer, all-period winner |
| SERT_H7 | 2112 | 0.1194 | Best overall OOS R² in the paper |
| SERT_H4 | 2212 (COVID + 1yr) | 0.1147 | Best SERT for this period |
| C_Trans_H1..H4 (benchmark) | 2112 | 0.0123-0.0273 | Standard Transformer, no pretraining |

Best proposed models achieve an annualized Sortino ratio ~47% (equal-weighted) / 28%
(value-weighted) higher than buy-and-hold during the pandemic period (Section 5.3.3,
Fig. 13/15).

## Reproducibility notes / known deviations

This implementation is faithful to every explicitly-stated architectural and
mathematical detail in the paper (Eq. 1-13, Appendix A-E), but several hyperparameters
were **not numerically specified** in the paper text and are therefore `ASSUMED` in
`configs/config.yaml` (see inline comments and
`sir-registry/arxiv_2505_01575/sir.json` → `implementation_assumptions[]` /
`ambiguities[]` for full provenance):

1. **L1 regularization coefficient** (`training.l1_lambda`, default `1e-5`, confidence 0.35).
2. **Pre-training MLP-autoencoder hidden-layer count** (`model.pretrain_hidden_layers`,
   default `1`, confidence 0.4).
3. **Batch size** (assumed full cross-section per step, confidence 0.5).
4. **LR schedule** (assumed constant Adam LR=0.001, no warmup/decay, confidence 0.6).
5. **Early stopping patience** (default `10`, confidence 0.4).
6. **Exact 420-stock universe / 182-factor construction is not enumerated** — see
   `data/README_data.md` for how to source a compatible dataset (HIGH risk item; see
   `architecture_plan.json` → `risk_assessment`).
7. **Diebold-Mariano HAC lag length** (default `12`, confidence 0.5).

Values marked `# ASSUMED` in `configs/config.yaml` should be treated as
starting points, not paper-verified constants.

## Citation

```bibtex
@article{lai2025assetpricing,
  title   = {Asset Pricing in Pre-trained Transformers},
  author  = {Lai, Shanyan},
  journal = {arXiv preprint arXiv:2505.01575},
  year    = {2025},
  note    = {v3 revised June 2026}
}
```

## Repository structure

```
.
├── configs/config.yaml           # all hyperparameters, ASSUMED values annotated
├── src/sert_asset_pricing/
│   ├── models/                   # positional_encoding, mlp_autoencoder, attention, blocks, transformer_variants
│   ├── data/                     # dataset (rolling-window), transforms (missingness filter)
│   ├── training/                 # losses (MSE+L1), trainer (Adam + early stopping)
│   ├── evaluation/                # metrics (OOS R2, DM-HAC test), backtest (Sharpe/Sortino/MDD)
│   └── utils/                    # config loading, seeding
├── train.py / evaluate.py / backtest.py / inference.py
├── data/README_data.md           # dataset sourcing instructions (data is NOT bundled)
├── docker/                       # Dockerfile + docker-compose.yml
└── notebooks/                    # reproduce_arxiv_2505_01575.ipynb walkthrough
```
