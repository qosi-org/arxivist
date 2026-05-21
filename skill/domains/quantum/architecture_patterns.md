# Domain: Quantum — Architecture Patterns (Stage 3 Enrichment)

Load alongside `agents/03_architecture_planner.md` when domain is **Quantum**.

---

## Framework selection

**PennyLane** — default for quantum ML and variational algorithms. Best Python integration.
**Qiskit** — use when paper explicitly mentions IBM hardware or Qiskit primitives.
**Cirq** — use for Google hardware or Cirq-specific circuit patterns.
**Never implement a quantum simulator from scratch** — always use a framework backend.

## Standard circuit patterns

**Hardware-efficient ansatz:**
```python
import pennylane as qml

def hardware_efficient_ansatz(params, n_qubits, n_layers):
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[l, i, 0], wires=i)
            qml.RZ(params[l, i, 1], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
```

**QAOA ansatz:**
```python
def qaoa_layer(gamma, beta, cost_h, mixer_h):
    qml.ApproxTimeEvolution(cost_h, gamma, 1)
    qml.ApproxTimeEvolution(mixer_h, beta, 1)
```

## Module structure for quantum papers

```
src/{project}/
├── circuits/
│   ├── ansatz.py          ← Parameterised circuit definitions
│   ├── hamiltonians.py    ← Problem Hamiltonians
│   └── noise_models.py    ← Hardware noise model definitions
├── optimisation/
│   ├── classical.py       ← Classical optimiser wrappers
│   └── gradients.py       ← Parameter shift rule, SPSA
├── simulation/
│   ├── statevector.py     ← Exact simulation backend
│   └── shot_based.py      ← Sampling simulation backend
├── evaluation/
│   └── metrics.py         ← Fidelity, energy error, expressibility
└── utils/
    └── config.py
```

## Config schema for quantum papers

```yaml
circuit:
  n_qubits: 4
  n_layers: 2
  ansatz: hardware_efficient
  backend: default.qubit      # pennylane device
  shots: 1024                 # null for statevector (exact)
  noise_model: null           # path to noise model JSON if used

optimisation:
  optimizer: adam
  learning_rate: 0.01
  max_steps: 500
  grad_method: parameter_shift  # or finite_diff, spsa

hardware:
  use_real_device: false
  device_name: null           # e.g. ibm_nairobi
  ibm_token: null             # loaded from env var IBM_TOKEN
```
