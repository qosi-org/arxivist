# Architecture Plan Summary — arxiv_2607_12990
**A Noise-Aware Quantum Algorithm for Credit Valuation Adjustments on Real Quantum Hardware**

## Framework
- **Primary**: PyTorch (for classical gradient-based training of variational circuit parameters) + **Qiskit** (mandatory quantum-circuit / hardware backend, cited explicitly by the paper).
- Python 3.10+, no CUDA required (circuits are ≤9 qubits; simulation and hardware calls are the bottleneck, not GPU compute).
- Config: plain YAML (hyperparameter space is small and well-bounded; no need for Hydra/OmegaConf).

## Module Hierarchy (13 files)
| Module | Role |
|---|---|
| `circuits/state_preparation.py` | QCBM ansatz `G_theta` (joint time-market distribution) |
| `circuits/controlled_rotations.py` | CRCA blocks: `R_v` (snake topology, exposure), `R_p`/`R_q` (native-tree, discount/default) |
| `circuits/cva_oracle.py` | Assembles `A_Theta`, builds Grover iterate `Q`, marked-subspace readout `Pi_111` |
| `estimation/cabiqae.py` | **Core contribution**: CABIQAE, plus BIQAE/BAE/DCS baselines |
| `estimation/contrast_calibration.py` | Hardware contrast-decay fit (`c0`, `tau_c`, `b`) + readout mitigation + hardware-replay model |
| `finance/classical_cva.py` | GBM path simulator, Black-Scholes pricer, CDS bootstrap, MC/finite-grid CVA |
| `finance/grid_encoding.py` | Builds `P_{i,j}`, `V+_{i,j}`, rescaling constants `C_v,C_p,C_q` |
| `hardware/backend_manager.py` | IBM Quantum backend + Q-CTRL Performance Management wrapper |
| `training/trainer.py`, `training/losses.py` | Classical variational training loop |
| `evaluation/error_decomposition.py` | Reproduces Section 4.5's formal error budget |
| `evaluation/metrics.py` | Query-cost accounting, log-log scaling-exponent fits |
| `utils/config.py` | YAML config + global seeding |

## Key Tensor Flows
1. **CVA oracle construction**: `|0>^9 → G_theta → R_v → R_p → R_q → Π₁₁₁ projection → a_CVA → rescale → CVA_Δ`
2. **CABIQAE loop**: adaptive Bayesian credible-interval shrinkage over Grover powers `k`, using the calibrated contrast-decay likelihood instead of the ideal `sin²(Kθ)` model
3. **Classical benchmark pipeline**: market calibration → correlated GBM paths → Black-Scholes pricing → CVA_MC (continuous) and CVA_tab (finite-grid) references

## Dependencies
`qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `numpy`, `scipy`, `torch`, `pandas`, `matplotlib`, `pyyaml` (+ standard dev tooling: pytest, black, ruff, mypy).

## Entrypoints
- `train.py` — trains QCBM/CRCA blocks
- `evaluate.py` — runs CABIQAE vs BIQAE vs BAE vs DCS comparison, noiseless or hardware-replay
- `inference.py` — single CVA estimation run
- `run_hardware_calibration.py` — paper-specific: fits the contrast-decay model from a Grover-power sweep

## Top Risks
1. **[High]** Q-CTRL's Performance Management internals are a documented black box → wrapped with a clearly-labelled fallback that replays the paper's published calibration data instead of claiming exact reproduction.
2. **[High]** Optimizer/LR/iteration counts for variational training are unstated → exposed as swappable config fields, all marked `# ASSUMED`.
3. **[Medium]** Market data (LSEG Workspace) is proprietary → `data/download.sh` + a synthetic fallback generator matching Table 4's published values.
4. **[Medium]** `R_v`'s snake-topology layer-repetition rule (L=2) is only diagrammed for L=1 → repetition logic will be unit-tested against Table 5's published gate counts (98 single-qubit, 86 two-qubit gates) to validate correctness.
5. **[Low]** Seeds and Fisher-information numerical edge cases (division near baseline `b`) → centralized seeding + the paper's own clipping/fallback stabilization (Appendix A.3) will be implemented directly.

**Next**: Stage 4 (Code Generator) will build the full repository at `paper-repos/arxiv_2607_12990/` from this plan.
