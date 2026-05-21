# GeomHerd — Architecture Plan Summary
**paper_id:** `arxiv_2605_11645`  
**Generated:** 2026-05-13

---

## Framework
- **Primary:** Python 3.10+ with PyTorch (forecasting head only) + NumPy/SciPy/POT (geometric pipeline)
- **CUDA:** Optional (only needed for Kronos head training; core geometry is CPU-only)
- **Config:** YAML + Python dataclasses

---

## Module Map

```
src/geomherd/
├── graph/
│   └── agent_graph.py          ← AgentGraph: windowed agreement edges (Eq. 1)
├── geometry/
│   ├── ricci_curvature.py      ← OllivierRicciComputer: LP-W1 via POT (Eq. 2-3)
│   ├── ricci_flow.py           ← DiscreteRicciFlow: neckpinch tau_sing [R1: ASSUMED update rule]
│   └── vocabulary.py           ← FSQVocabularyTracker: V_eff = exp(H(p_t))
├── detection/
│   └── cusum.py                ← CUSUMDetector, KendallTauDetector, ContagionDetector
├── pipeline/
│   └── geomherd_pipeline.py    ← GeomHerdPipeline: top-level orchestrator
├── simulation/
│   ├── cws_substrate.py        ← CWSSubstrate: N=66 agents, na=4 assets, kappa sweep
│   ├── llm_agent.py            ← PersonaAgent (LLM) + RuleBasedAgentFallback
│   └── vicsek_substrate.py     ← VicsekSubstrate: N=600 particles, eta sweep
├── forecasting/
│   └── kronos_head.py          ← PriceTokeniser (frozen VQ-VAE) + AdaLNZero + KronosHead [R2: ASSUMED arch]
├── evaluation/
│   ├── metrics.py              ← DetectionMetrics, ForecastMetrics (rliable IQM)
│   └── baselines.py            ← LSVBaseline, CSADBaseline (+ CCK regression Eq. 8)
└── utils/
    └── config.py               ← GeomHerdConfig dataclass
```

---

## Core Data Flow

```
Agent actions [N, Tw=100]
        │
        ▼
AgentGraph ──── w[N,N] (windowed agreement, sparsified at 0.5)
        │
        ▼
OllivierRicciComputer ──── kappa_dict {(i,j): float}
        │                         │
        │                         ├── kappa_bar_plus  (mean over E+, κ>0.1)
        │                         └── beta_minus      (|E-|/|E|, κ<-0.1)
        │
        ├──► DiscreteRicciFlow ──── tau_sing (neckpinch time)
        │
        └──► FSQVocabularyTracker ── V_eff = exp(H(codebook utilization))

kappa_bar_plus ──► CUSUMDetector ──────────────────► alarm_plus (bool)
beta_minus     ──► CUSUMDetector + KendallTauDetector ► alarm_minus (bool)

(kappa_bar_OR, tau_sing, V_eff) ──► KronosHead ──► log_return_pred
```

---

## config.yaml Template

```yaml
# GeomHerd configuration
# [ASSUMED] = not explicitly in paper; see risk assessment

graph:
  Tw: 100              # windowed agreement window (explicitly stated)
  w0: 0.5              # sparsification threshold (explicitly stated)
  delta_t: 10          # snapshot stride (explicitly stated)

curvature:
  alpha: 0.5           # lazy-walk laziness (explicitly stated, matches Sandhu 2016)
  kappa_plus_thresh: 0.1   # sign decomp threshold (explicitly stated)
  kappa_minus_thresh: -0.1

ricci_flow:
  step_size: 0.01      # [ASSUMED: not in paper] multiplicative update rate
  max_iter: 1000       # [ASSUMED] flow iteration budget
  flow_variant: "multiplicative"  # swappable: multiplicative | additive

detection:
  baseline_window: 35  # W CUSUM baseline (Appendix D: W=35 samples)
  operating_point: "precision"  # recall | precision
  recall_oriented:
    k_sigma: 0.5
    h_sigma: 4.0
  precision_oriented:
    k_sigma: 2.0
    h_sigma: 4.0
  kendall_tau_thresh: -0.4   # [ASSUMED: inferred from Table 3 label]
  kendall_window: 20         # [ASSUMED]

vocabulary:
  codebook_dims: 3      # explicitly stated
  levels_per_dim: 4     # explicitly stated
  K: 64                 # = 4^3

simulation:
  llm_mode: false        # set true to use real LLM agents (requires ANTHROPIC_API_KEY)
  llm_model: "claude-sonnet-4-20250514"   # [ASSUMED] smaller model sufficient
  N_agents: 66           # explicitly stated
  N_assets: 4            # explicitly stated
  kappa_values: [0.5, 0.8, 1.2, 1.8, 2.5]
  seeds_per_kappa: 80
  # Vicsek
  N_particles: 600
  eta_values: [0.5, 1.0, 1.6, 2.0, 2.5]
  seeds_vicsek: 20
  knn_k: 10

kronos:
  d_model: 64            # [ASSUMED: not in paper]
  n_layers: 2            # [ASSUMED]
  n_heads: 4             # [ASSUMED]
  tokeniser_codebook_size: 512   # [ASSUMED]
  train_epochs: 50       # [ASSUMED]
  lr: 1.0e-4             # [ASSUMED]
  cascade_window_steps: 100   # [ASSUMED]

evaluation:
  n_boot: 5000           # explicitly stated
  herding_event_threshold: 0.5   # theta_event, explicitly stated
  geom_threshold: 0.30   # theta_geom, explicitly stated
```

---

## Entrypoints

| Script | Purpose |
|--------|---------|
| `run_detection.py` | Full GeomHerd detection pipeline on CWS or Vicsek |
| `run_baselines.py` | All 7 baseline detectors |
| `run_evaluation.py` | Compute Tables 2 & 3 from saved outputs |
| `train_kronos.py` | Train Kronos forecasting head |
| `run_cck_regression.py` | Augmented CCK regression (Eq. 8) |

---

## Key Risks

| ID | Severity | Issue |
|----|----------|-------|
| R1 | **High** | Ricci flow update rule unspecified — assuming multiplicative; make swappable |
| R2 | **High** | Kronos head architecture absent — small configurable transformer |
| R3 | Medium | LLM prompts withheld — rule-based fallback provided |
| R4 | Medium | Kendall-tau params inferred — expose as config |
| R5 | Medium | CWS full mechanics need Cividino 2023 paper for validation |
| R6 | Low | LP-W1 is O(N³ log N) per edge — batch with POT |
