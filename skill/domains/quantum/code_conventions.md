# Domain: Quantum — Code Conventions (Stage 4 Enrichment)

Load alongside `agents/04_code_generator.md` when domain is **Quantum**.

---

## Circuit implementation standards

**Always implement a `simulate_exact` and `simulate_shots` mode:**
```python
def get_device(config):
    if config.circuit.shots is None:
        return qml.device("default.qubit", wires=config.circuit.n_qubits)
    return qml.device("default.qubit", wires=config.circuit.n_qubits,
                       shots=config.circuit.shots)
```

**Parameter initialisation must be reproducible:**
```python
rng = np.random.default_rng(config.hardware.seed)
params = rng.uniform(0, 2 * np.pi, shape=(n_layers, n_qubits, 2))
```

**Always implement the parameter shift rule explicitly** even if the framework provides it
automatically — it makes the gradient computation transparent and verifiable:
```python
def parameter_shift_gradient(circuit, params, idx, shift=np.pi/2):
    params_plus = params.copy(); params_plus[idx] += shift
    params_minus = params.copy(); params_minus[idx] -= shift
    return (circuit(params_plus) - circuit(params_minus)) / (2 * np.sin(shift))
```

**Shot noise:** results with shots < 1024 have high variance. Always run multiple seeds
and report mean ± std. Make `n_seeds` a config parameter.

## Requirements

```
pennylane>=0.36.0
pennylane-lightning>=0.36.0   # faster CPU simulation
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
# Optional for hardware runs:
# qiskit>=1.0.0
# qiskit-ibm-runtime>=0.20.0
```

## What Must NOT be done

- Do NOT hardcode qubit count — always use `config.circuit.n_qubits`
- Do NOT run on real hardware without the user explicitly enabling it in config
- Do NOT ignore shot noise — always report variance for shot-based results
- Do NOT implement Shor's or Grover's from scratch — use framework primitives
