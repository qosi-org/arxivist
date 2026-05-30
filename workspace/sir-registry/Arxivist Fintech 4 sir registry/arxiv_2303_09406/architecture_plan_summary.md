# Architecture Plan: FS-GCLSTM
## Paper: Liu (2023/2025) — arXiv 2303.09406

---

## 1. Framework Selection

**Framework**: PyTorch 2.1+ with PyTorch Geometric (PyG)  
**Reasoning**: GCN layers are standard in PyG (`GCNConv`). LSTM cells require custom implementation to apply GCN to all inputs. No GPU architecture specified in paper; CUDA recommended for graph operations.  
**Python**: 3.10+  
**CUDA**: Required (12.x recommended), CPU fallback supported  
**Config**: YAML + dataclass

---

## 2. Module Hierarchy

```
paper-repos/arxiv_2303_09406/
├── src/fsgclstm/
│   ├── models/
│   │   ├── gcn_layer.py          # GraphConvLayer (Eq. 1-2)
│   │   ├── fsgclstm_cell.py      # FSGCLSTMCell (Eqs. 2-6)
│   │   ├── fsgclstm_model.py     # Full FS-GCLSTM: stacked cells + MLP
│   │   └── baselines.py          # FCL, LSTM baselines
│   ├── data/
│   │   ├── dataset.py            # RollingWindowGraphDataset
│   │   └── graph_builder.py      # Adjacency matrix construction from LSEG data
│   ├── training/
│   │   ├── trainer.py            # Rolling-window train/val/test loop
│   │   └── losses.py             # MSE loss (assumed)
│   ├── evaluation/
│   │   └── metrics.py            # MSE, MAE, Directional Acc, Sharpe, Sortino
│   └── utils/
│       └── config.py             # Config dataclass + YAML loading + seed
├── configs/config.yaml
├── docker/Dockerfile
├── docker/docker-compose.yml
├── data/README_data.md
├── notebooks/reproduce_arxiv_2303_09406.ipynb
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## 3. Tensor Flow

```
FORWARD PASS: FSGCLSTMModel
  Input sequence:  [seq_len, N, d]  (seq_len=d rolling window days, N nodes, d features)
  A_t:             [N, N]            adjacency matrix

  For each time step t in [1..seq_len]:
    H_x   = GCN_2layer(X_t, A_t)        → [N, hidden_dim]   # GCN on input features
    H_h   = GCN_2layer(h_{t-1}, A_t)    → [N, hidden_dim]   # GCN on previous hidden state
    H_c   = GCN_2layer(c_{t-1}, A_t)    → [N, hidden_dim]   # GCN on previous cell state  ← KEY INNOVATION
    f_t   = σ(W_f [H_h, H_x] + b_f)    → [N, hidden_dim]
    i_t   = σ(W_i [H_h, H_x] + b_i)    → [N, hidden_dim]
    g_t   = tanh(W_c [H_h, H_x] + b_c) → [N, hidden_dim]
    c_t   = f_t ⊙ H_c + i_t ⊙ g_t     → [N, hidden_dim]
    o_t   = σ(W_o [H_h, H_x] + b_o)    → [N, hidden_dim]
    h_t   = o_t ⊙ tanh(c_t)            → [N, hidden_dim]

  Stacked 3 cells → final h_T from each cell
  Concatenate: [h_T_layer1, h_T_layer2, h_T_layer3] → [N, 3*hidden_dim]
  Select N_pred target nodes             → [N_pred, 3*hidden_dim]
  Flatten                                → [N_pred * 3 * hidden_dim]
  MLP                                    → [N_pred]  (predicted next-day returns)
```

---

## 4. Config Schema (config.yaml)

```yaml
model:
  hidden_dim: 64           # ASSUMED: not stated in paper (conf: 0.45)
  n_gcn_layers: 2          # Stated: "two GCN layers" (conf: 0.99)
  n_lstm_layers: 3         # Stated: "three stacked FS-GCLSTM cells" (conf: 0.99)
  input_seq_len: 60        # Default from paper (conf: 0.99); also tested: 30, 90, 120
  mlp_hidden: 128          # ASSUMED: not stated (conf: 0.50)
  dropout: 0.0             # ASSUMED: not mentioned (conf: 0.55)

training:
  optimizer: adam
  lr: 0.001                # Stated (conf: 0.99)
  weight_decay: 1.0e-5     # Stated (conf: 0.99)
  lr_schedule: onecycle
  max_epochs: 30           # Stated (conf: 0.99)
  early_stop_patience: 10  # Stated (conf: 0.99)
  loss: mse                # ASSUMED (conf: 0.70)
  initial_window_days: 3000
  train_frac: 0.70
  val_frac: 0.20
  test_frac: 0.10
  advance_days: 300

data:
  price_start: "2000-01-01"
  price_end: "2024-12-31"
  min_trading_days: 5000
  lseg_confidence_threshold: 0.20  # Stated (conf: 0.99)
  bidirectional_edges: true
  transaction_cost_bps: 1          # Stated (conf: 0.99)

evaluation:
  risk_free_eurostoxx: "EONIA"
  risk_free_sp500: "USD_LIBOR_ON"
  portfolio_type: "equal_weight_long_only"
  rebalance: "daily"

hardware:
  device: cuda
  precision: float32
  seed: 42
```

---

## 5. Dependencies

```
torch>=2.1.0
torch-geometric>=2.4.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.10.0
pyyaml>=6.0
tqdm>=4.65.0
pmdarima>=2.0.3          # For ARIMA baseline
statsmodels>=0.14.0
```

---

## 6. Risk Assessment

| Risk | Severity | Description | Mitigation |
|------|----------|-------------|------------|
| Hidden dim unknown | High | Results depend on hidden_dim; paper doesn't state it | Expose as config; default 64 |
| MLP architecture unknown | High | Layer sizes not stated | Stub with TODO; expose as config |
| LSEG data proprietary | High | Value-chain data requires LSEG subscription | Synthetic graph generator for testing |
| Batch size unclear | Medium | Full-batch may OOM on large graphs | Configurable; default full-batch |
| Loss function not stated | Medium | Paper evaluates MSE but may train with different loss | Default MSE; flag as assumed |
| Cell state GCN shape mismatch | Medium | h and c may have different dims than X; concat before gates | Careful dimension tracking |
