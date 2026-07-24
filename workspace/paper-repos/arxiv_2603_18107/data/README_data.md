# Data — arxiv_2603_18107 (ARTEMIS)

This paper uses 4 datasets (Section 3). Availability differs sharply across them.

| Dataset | Publicly available? | Source | Notes |
|---|---|---|---|
| Jane Street | ✅ Yes | [Kaggle: Jane Street Real-Time Market Data Forecasting](https://kaggle.com/competitions/jane-street-real-time-market-data-forecasting) | ~7.37M train / ~4.61M val / 200k test rows after windowing; streamed loader required (dataset exceeds memory) |
| Optiver | ✅ Yes | [Kaggle: Optiver Realized Volatility Prediction](https://kaggle.com/competitions/optiver-realized-volatility-prediction) | Requires fusing order-book + trade streams into a 1Hz 600-step sequence (Section 3.2) |
| Time-IMM (EPA-Air) | ✅ Yes | Time-IMM collection, EPA-Air domain (Chang et al. 2025, arXiv:2506.10412) | Hourly, 8 US counties, only temperature is densely sampled |
| DSLOB | ❌ **No** | **Not available.** Synthetic dataset introduced by this paper, generated from an unnamed "real high-frequency limit order book dataset" (Section 3.4) | See below |

## ⚠ DSLOB cannot be reproduced

Section 3.4 states DSLOB's synthetic generation is seeded from "a real high-frequency limit
order book dataset" but **never names, cites, or provides this source dataset anywhere in the
paper**. Since DSLOB is the dataset behind:
- the paper's headline directional-accuracy claim (64.96%),
- the entire 7-variant ablation study (Table 3),
- the regime-degradation analysis (Figure 4),

this is a **structural blocker to full reproduction**, not an implementation gap. See
`comparison/hallucination_report.md` for the full analysis.

### What this repo provides instead

`data/make_synthetic_seed_lob.py` generates a **placeholder** seed mid-price + feature series
(normal-regime random walk + an embedded synthetic "crash") so that the actual DSLOB
*generation procedure* described in Section 3.4 — CUSUM/change-point crash-window detection,
Vasicek mid-price MLE fitting + amplification, GARCH(1,1) volatility fitting + amplification,
VAR(1)-correlated noise for the remaining 84 features, Gaussian-process time warping — can be
exercised end-to-end for **code validation only**.

Running this pipeline validates that `DSLOBGenerator` is implemented correctly. It does
**not** produce the paper's actual DSLOB dataset, and any metrics computed on it are not
comparable to Table 2/3.

```bash
python data/make_synthetic_seed_lob.py --out data/raw/seed_lob.npz --n-obs 5000
python scripts/prepare_data.py --dataset dslob --raw-dir data/raw --out-dir data/processed
```

## Jane Street / Optiver / Time-IMM

These three are genuinely public. To use them:

1. Download from the Kaggle competitions / Time-IMM repository linked above.
2. Place raw files under `data/raw/<dataset>/`.
3. Use the loader classes in `src/arxivist_artemis/data/preprocessing.py`
   (`JaneStreetPreprocessor`, `OptiverPreprocessor`, `TimeIMMPreprocessor`) to build the
   `.npz` cache expected by `scripts/train_model.py` (see each class's docstring for the exact
   expected raw-file schema, derived directly from Sections 3.1–3.3).
4. `scripts/prepare_data.py` currently only wires up the `dslob` case end-to-end; the other
   three preprocessors are implemented and unit-testable but require you to supply the actual
   competition/collection files (which ArXivist cannot bundle) to drive the CLI. See each
   preprocessor class for direct Python usage.

## Directory layout expected

```
data/
├── README_data.md              (this file)
├── make_synthetic_seed_lob.py  (DSLOB placeholder seed generator)
└── raw/                        (git-ignored)
    ├── seed_lob.npz            (from make_synthetic_seed_lob.py)
    ├── jane_street/             (user-supplied Kaggle files)
    ├── optiver/                 (user-supplied Kaggle files)
    └── time_imm/                (user-supplied Time-IMM files)
```
