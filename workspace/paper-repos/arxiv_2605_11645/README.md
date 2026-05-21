# GeomHerd

**GeomHerd: A Forward-looking Herding Quantification via Ricci Flow Geometry on Agent Interactive Simulations**

Lake Yang, Junwei Su, Jingfeng Zeng, Wenhao Lu, Xingzhi Qian, Weitong Zhang, Chuan Wu, Dunhong Jin

📄 [arXiv:2605.11645](https://arxiv.org/abs/2605.11645) | May 2026

---

## What This Paper Does

GeomHerd detects market herding — the phenomenon where traders align their behaviour and act collectively — *before* it manifests in prices. Classical methods (LSV buy/sell imbalance, CSAD return dispersion) are inherently post-hoc: they can only fire after coordinated actions have already moved realized returns. GeomHerd sidesteps this by measuring coordination directly on the **agent interaction graph**: a dynamic network where each node is a trader and each edge encodes recent behavioural agreement.

The core insight is geometric: herding is a **topological collapse** of the agent graph. As traders imitate each other, their graph neighbourhoods become more similar and tightly connected — this is captured by the **Ollivier-Ricci curvature** of the graph edges. Positive curvature (rising `κ̄⁺_OR`) signals within-clique coordination; strongly negative curvature (`β⁻`) identifies bridge edges along which contagion propagates. A Ricci-flow singularity time `τ_sing` provides a forward-looking time-to-collapse estimate, and an effective vocabulary `V_eff` measures behavioral homogenisation.

On the Cividino-Sornette financial ABM, GeomHerd fires a **median 272 steps before order-parameter onset** (recall-oriented) and outpaces price-correlation graph baselines by 40 steps on co-firing trajectories. The curvature signature transfers out-of-domain to the Vicsek particle model (AUROC 0.99) and conditions a forecasting head that reduces cascade-window log-return MAE.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd arxiv_2605_11645
pip install -e .

# 2. Run GeomHerd on a single CWS trajectory (rule-based agents, no LLM needed)
python run_detection.py --substrate cws --kappa 1.8 --seeds 5 --operating_point recall

# 3. Run full CWS sweep (reproduces paper Table 3, ~2-4h on CPU)
for kappa in 0.5 0.8 1.2 1.8 2.5; do
    python run_detection.py --substrate cws --kappa $kappa --seeds 80
done

# 4. Evaluate and print Table 3
python run_evaluation.py --results_dir results/detection/

# 5. Run Vicsek OOD transfer
python run_detection.py --substrate vicsek --eta 1.6 --seeds 20
```

---

## Installation

### pip
```bash
pip install -e .
# For LLM agents:
pip install -e ".[llm]"
# For Kronos forecasting head:
pip install -e ".[forecasting]"
```

### conda
```bash
conda env create -f environment.yaml
conda activate geomherd
pip install -e .
```

---

## Usage

### Detection pipeline
```bash
# Precision-oriented (low false alarms, high precision)
python run_detection.py --substrate cws --kappa 1.8 --seeds 80 --operating_point precision

# Recall-oriented (abstract headline: 272-step lead)
python run_detection.py --substrate cws --kappa 1.8 --seeds 80 --operating_point recall

# LLM-driven agents (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="sk-ant-..."
python run_detection.py --substrate cws --kappa 1.8 --seeds 10 --llm_mode true

# Quick debug run
python run_detection.py --substrate cws --kappa 1.8 --debug
```

### Augmented CCK regression (Table 2 text: γ₃)
```bash
python run_cck_regression.py --data_dir results/detection/ --output results/cck.json
# Expected: gamma3 median = -0.0072, CI = [-0.00769, -0.00602]
```

### Kronos forecasting head (Section 3.3.3)
```bash
# Dry run (validates architecture)
python train_kronos.py --dry_run

# Training (requires CWS trajectory data)
python train_kronos.py --data_dir results/detection/ --epochs 50
```

---

## Expected Results

### Table 3: Multi-axis detection profile (CWS binary-edge)

| Detector | Precision | Recall_super | FAR_sub | AUROC | Median lead |
|----------|-----------|-------------|---------|-------|-------------|
| **GeomHerd κ̄⁺_OR (precision)** | 0.45 | 0.04 | 0.07 | 0.48 | **178 steps** |
| **GeomHerd κ̄⁺_OR (recall)** | — | 0.52 | 0.76 | 0.48 | **272 steps** |
| **GeomHerd β⁻ (contagion)** | 0.55 | 0.65 | 0.81 | 0.80 | **318 steps** |
| Srinivasan 2026 (PCG) | 0.71 | 0.79 | 0.49 | 0.66 | 20 steps |
| Sandhu 2016 (PCG) | 0.72 | 0.95 | 0.55 | 0.72 | 80 steps |
| CSAD (CCK) | 0.69 | 1.00 | 0.68 | 0.75 | 180 steps |
| LSV 1992 | 0.60 | 1.00 | **1.00** | 0.48 | 355 steps |

*Note: LSV/CSAD fire on all trajectories (FAR_sub=1.00/0.68) and are not regime classifiers.*

### Table 2: Paired-bootstrap lead difference (precision-oriented point)

| Comparator | Lead diff (steps) | 95% CI | p-value |
|-----------|-------------------|--------|---------|
| Srinivasan 2026 | +191.7 | [-35.0, 393.3] | 0.11 |
| Sandhu 2016 | +74.4 | [-36.7, 195.6] | 0.20 |
| Lap-CSAD (Huang 2023) | +153.8 | [28.7, 297.5] | **0.03** |

### Augmented CCK regression (Eq. 8)
- γ₃ (kappa_bar_OR coefficient): median **−0.0072**, CI [−0.00769, −0.00602]
- Sign-consistent with Proposition 1 (CSADₜ monotonically decreasing in κ̄_OR)

### Vicsek OOD transfer (Section 3.3.3, Figure 5b)
- κ̄_OR(τ★) AUROC: **0.99** (95% CI [0.98, 1.00])
- Per-η medians monotone: +0.08 (η=0.5) → −0.26 (η=2.5)

---

## Repository Structure

```
src/geomherd/
├── graph/agent_graph.py          # Windowed action-agreement graph (Eq. 1)
├── geometry/
│   ├── ricci_curvature.py        # ORC via LP-W1 (Eqs. 2-3)
│   ├── ricci_flow.py             # Discrete Ricci flow → tau_sing [ASSUMED update rule]
│   └── vocabulary.py             # V_eff = exp(H(p_t)), FSQ codebook
├── detection/cusum.py            # CUSUM + Kendall-tau detectors (Eqs. 4-6)
├── pipeline/geomherd_pipeline.py # Top-level pipeline orchestrator
├── simulation/
│   ├── cws_substrate.py          # Cividino-Sornette CWS model
│   ├── llm_agent.py              # PersonaAgent (LLM) + RuleBasedAgentFallback
│   └── vicsek_substrate.py       # Vicsek particle model
├── forecasting/kronos_head.py    # Kronos forecasting head [STUB — arch unknown]
├── evaluation/
│   ├── metrics.py                # All detection and forecasting metrics
│   └── baselines.py              # LSV, CSAD/CCK regression
└── utils/config.py               # Config dataclass + seed utility
```

---

## Reproducibility Notes

### Known Implementation Assumptions

1. **Ricci flow update rule** *(Risk R1 — High severity)*
   - Paper specifies the neckpinch stopping criterion but **not** the per-step flow update.
   - Implementation assumes multiplicative: `w(e) ← w(e) * (1 − η * κ(e))`
   - Configurable via `ricci_flow.flow_variant` in `config.yaml`
   - `τ_sing` values may differ from paper without the exact rule

2. **Kronos head architecture** *(Risk R2 — High severity)*
   - Layer count, hidden dim, and head count absent from paper.
   - Assumed: 2 layers, 4 heads, d_model=64 (small configurable transformer)
   - Forecasting MAE numbers will likely not reproduce exactly

3. **LLM persona prompts** *(Risk R3 — Medium severity)*
   - Authors intentionally withheld prompts
   - This repo provides `RuleBasedAgentFallback` as default (no LLM required)
   - The geometric pipeline is LLM-agnostic; geometry results hold either way

4. **Kendall-tau parameters** *(Risk R4 — Medium)*
   - `tau_thresh = -0.4` inferred from Table 3 row label; `W_tau = 20` assumed
   - Configurable in `config.yaml`

5. **CWS substrate mechanics** *(Risk R5 — Medium)*
   - Full CWS parameterization from Cividino et al. (2023) required for exact reproduction
   - This implementation is a faithful approximation from the paper description

All assumed hyperparameters are marked with `# ASSUMED` comments in `configs/config.yaml`.

---

## Citation

```bibtex
@article{yang2026geomherd,
  title={GeomHerd: A Forward-looking Herding Quantification via Ricci Flow Geometry
         on Agent Interactive Simulations},
  author={Yang, Lake and Su, Junwei and Zeng, Jingfeng and Lu, Wenhao and
          Qian, Xingzhi and Zhang, Weitong and Wu, Chuan and Jin, Dunhong},
  journal={arXiv preprint arXiv:2605.11645},
  year={2026}
}
```

---

*This repository was automatically generated by [ArXivist](https://github.com/anthropics/arxivist) from the paper PDF.*
*Overall SIR confidence: 0.80. See `sir-registry/` for full scientific intermediate representation.*
