# Noisy QNN Universal Approximation — Black-Scholes Pricing

ArXivist-generated reproduction of:

> **Quantitative Universal Approximation for Noisy Quantum Neural Networks**
> Lukas Gonon, Antoine Jacquier, Marcel Mordarski — arXiv:2604.02064v3 [quant-ph]

This repository implements the paper's parameterized quantum-circuit universal
approximator, its depolarising-noise error bounds, and the Black-Scholes European
Put option pricing experiments, both in simulation and against real IBM Quantum
hardware.

> ⚠️ **Reproducibility status**: several implementation details are **not fully
> specified in the paper** (optimizer hyperparameters, the exact loss
> normalisation constant, the readout error probability used in the reported
> hardware bound, and the exact UCR circuit decomposition). These are called out
> explicitly below, in `configs/config.yaml` (as `*_comment: "ASSUMED: ..."`
> fields), and in `sir-registry/arxiv_2604_002064/sir.json` →
> `implementation_assumptions` / `ambiguities`. Do not treat this repo's output
> numbers as a bit-exact match to the paper without first tightening these.

## What this paper does

Proves non-asymptotic universal-approximation error bounds for **noisy quantum
neural networks (QNNs)** — parameterized quantum circuits approximating a target
function — as a function of circuit width (accuracy blocks `n`), qubit count, and
hardware noise parameters. Motivated by option pricing, it specialises the bounds
to depolarising noise calibrated to a real IBM device (`ibm_fez`), shows a
two-parameter affine post-processing layer can exactly cancel the depolarising
bias, and validates everything against hardware execution on Black-Scholes Put
option prices.

## Repository layout

```
.
├── src/noisy_qnn_uat/
│   ├── models/
│   │   ├── qnn_circuit.py       # V (state prep) + U(theta,x) via UCRZ/UCRY (Sec. 3.2, 4.1)
│   │   ├── measurement.py       # outcome grouping -> P0..P3 -> scalar output (Eq. 2.2)
│   │   ├── noise_channels.py    # depolarising channel, hardware lambda_V/lambda_U (Sec. 3.5-3.7)
│   │   └── postprocessing.py    # Theorem 3.15 affine bias-cancellation layer
│   ├── data/
│   │   ├── dataset.py           # Gaussian density + Black-Scholes Put generators
│   │   └── transforms.py        # input normalisation (Eq. 4.1), classical BS Put formula
│   ├── training/
│   │   ├── losses.py            # MSE loss (Section 4.1)
│   │   └── trainer.py           # Methods A (L-BFGS-B) / B (two-stage) / C (Adam)
│   ├── evaluation/
│   │   ├── metrics.py           # RMSE, MAE, max error, bound ratio
│   │   └── hardware_bounds.py   # Statement 2.1 / Thm 3.6 / Thm 3.17 / Prop 3.20 bounds
│   └── utils/config.py          # config loading + reproducibility seeding
├── scripts/
│   ├── run_hardware.py          # ibm_fez execution + epsilon_total envelope (Section 4.5)
│   └── simulate_noise.py        # depolarising-noise sweep (Section 4.4)
├── configs/
│   ├── config.yaml              # full paper-scale configuration
│   └── config_debug.yaml        # reduced-scale config for fast local smoke tests
├── docker/Dockerfile
├── data/download.py             # no-op: all datasets are generated analytically
├── notebooks/                   # populated by ArXivist Stage 5 (Notebook Generator)
├── comparison/                  # populated by ArXivist Stage 6 (Results Comparator)
├── train.py
├── evaluate.py
├── inference.py
├── setup.py
├── requirements.txt
├── requirements-dev.txt
└── environment.yaml
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yaml
conda activate noisy_qnn_uat
```

Or with Docker:

```bash
docker build -t noisy-qnn-uat -f docker/Dockerfile .
docker run --rm noisy-qnn-uat python train.py --config configs/config_debug.yaml --debug
```

## Quickstart

```bash
# Fast smoke test (small circuit, few iterations, no hardware access needed)
python train.py --config configs/config_debug.yaml --debug

# Full-scale training (Method A: L-BFGS-B, Section 4.1)
python train.py --config configs/config.yaml --method A

# Evaluate a trained checkpoint on the 40x40 Black-Scholes grid (Section 4.3)
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/theta_methodA_seed42.json

# Add depolarising noise at evaluation time (Section 4.4)
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/theta_methodA_seed42.json --noise depolarising

# Single-sample inference: price one option (S,K,T,r,sigma)
python inference.py --config configs/config.yaml --checkpoint checkpoints/theta_methodA_seed42.json --input 100,100,1.0,0.03,0.2

# Depolarising-noise sweep (Section 4.4)
python scripts/simulate_noise.py --config configs/config.yaml

# Hardware execution on IBM ibm_fez (falls back to AerSimulator if no credentials; Section 4.5)
export IBM_QUANTUM_TOKEN=...   # see .env.example
python scripts/run_hardware.py --config configs/config.yaml --checkpoint checkpoints/theta_methodA_seed42.json
```

## Known gaps vs. the paper (read before trusting exact numbers)

| Gap | Where it's handled | Confidence |
|---|---|---|
| Optimizer hyperparameters (lr, iterations) for Methods A/B/C unspecified | `configs/config.yaml` → `training.*` (ASSUMED defaults) | 0.3 |
| Loss normalisation constant ambiguous (PDF extraction corruption) | `training/losses.py` (`normalisation` param, default `1/(2n)`) | 0.5 |
| Readout error probability `p` for ibm_fez missing from paper's own hardware table, but used in the reported bound | `configs/config.yaml` → `hardware.readout_p` (ASSUMED 0.01) | 0.35 |
| `n0` padding value not confirmed for main experiments | `configs/config.yaml` → `model.n0` (ASSUMED 0, per Remark 3.14) | 0.4 |
| Exact UCR gate decomposition of U(theta,x) delegated to external reference [23] | `models/qnn_circuit.py` (Qiskit `UCRZGate`/`UCRYGate` reconstruction) | 0.6 — validate against the closed-form reference (`closed_form_reference_output`) per Section 4.1.1 before trusting derived noise parameters |

Full detail: `sir-registry/arxiv_2604_002064/sir.json` → `implementation_assumptions`,
`ambiguities`; `sir-registry/arxiv_2604_002064/architecture_plan.json` → `risk_assessment`.

### Circuit-construction validation (Section 4.1.1 check)

`src/noisy_qnn_uat/models/qnn_circuit.py` derives, in a docstring, exactly why its
UCRZ/UCRY-based reconstruction of U(theta,x) reproduces the paper's closed-form
reference formula `closed_form_reference_output` (Section 4.1.1). This was verified
during repository generation by running the paper's own validation check: sample
the circuit on `AerSimulator`, compare against the closed-form value, and confirm
the residual stays within the paper's stated `R / sqrt(N_shots)` shot-noise bound.
Re-run it yourself after any change to `qnn_circuit.py` or `measurement.py`:

```python
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from noisy_qnn_uat.models.qnn_circuit import QNNCircuitBuilder, closed_form_reference_output
from noisy_qnn_uat.models.measurement import MeasurementProcessor

n_blocks, n_qubits, d, R, shots = 4, 4, 2, 10.0, 8192
builder, processor, sim = QNNCircuitBuilder(n_blocks, n_qubits), MeasurementProcessor(), AerSimulator()
theta = [(np.random.uniform(-1, 1, d), np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(n_blocks)]
x = np.random.uniform(0, 1, d)

circuit = transpile(builder.assemble_circuit(theta, x), sim, optimization_level=1)
counts = sim.run(circuit, shots=shots).result().get_counts()
probs = processor.group_counts(counts, n_blocks, n_qubits)
sampled = processor.qnn_output(probs, R)
reference = closed_form_reference_output(theta, x, R, n_blocks)
print(abs(sampled - reference), "should be <~", R / np.sqrt(shots))
```

Note: `AerSimulator` does not natively execute `UCRZGate`/`UCRYGate` instructions —
always `transpile()` the circuit to the simulator's basis gates first (as shown
above and in `scripts/run_hardware.py`), or you will hit `AerError: unknown
instruction: ucrz`.

## Citation

If you use this reproduction, please cite the original paper:

```
Gonon, L., Jacquier, A., and Mordarski, M. (2026).
Quantitative Universal Approximation for Noisy Quantum Neural Networks.
arXiv:2604.02064.
```

---
*This repository was generated by ArXivist from the paper's Scientific Intermediate
Representation (SIR); see `sir-registry/arxiv_2604_002064/` for the full parsed spec,
architecture plan, and provenance trail.*
