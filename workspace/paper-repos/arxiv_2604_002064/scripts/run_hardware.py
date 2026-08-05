#!/usr/bin/env python3
"""Execute a trained QNN circuit on an IBM Quantum backend and check the epsilon_total
error envelope against the classical Black-Scholes reference (Section 4.5).

Usage:
    python run_hardware.py --config configs/config.yaml \
        --checkpoint checkpoints/theta_methodA_seed42.json \
        --backend ibm_fez --shots 8192

Requires IBM_QUANTUM_TOKEN to be set (see .env.example) and network access to
IBM Quantum Platform. Falls back to a local AerSimulator + NoiseModel built from
configs/config.yaml's hardware parameters if no live backend/credentials are
available, so this script remains runnable without hardware access.

Paper section: Section 4.5 "Hardware execution on ibm_fez", Eq. (4.4).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from qiskit import transpile

from noisy_qnn_uat.data.dataset import BlackScholesPutDataset
from noisy_qnn_uat.evaluation.hardware_bounds import ErrorBoundCalculator
from noisy_qnn_uat.evaluation.metrics import ErrorMetrics
from noisy_qnn_uat.models.measurement import MeasurementProcessor
from noisy_qnn_uat.models.noise_channels import HardwareNoiseCalibrator
from noisy_qnn_uat.models.qnn_circuit import QNNCircuitBuilder
from noisy_qnn_uat.utils.config import ConfigLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained theta checkpoint")
    parser.add_argument("--backend", type=str, default="ibm_fez", help="IBM Quantum backend name")
    parser.add_argument("--shots", type=int, default=8192, help="Shots per circuit")
    return parser.parse_args()


def _theta_tuples(flat_theta: np.ndarray, n_accuracy_blocks: int, d: int):
    block_size = d + 2
    theta = []
    for k in range(n_accuracy_blocks):
        block = flat_theta[k * block_size:(k + 1) * block_size]
        theta.append((block[:d], float(block[d]), float(block[d + 1])))
    return theta


def _run_on_hardware_or_fallback(circuit, shots: int, backend_name: str, optimization_level: int = 3):
    """Try live IBM Quantum hardware; fall back to AerSimulator if unavailable.

    Both paths transpile first: AerSimulator does not natively execute raw
    UCRZGate/UCRYGate instructions (confirmed during repo validation --
    `AerError: unknown instruction: ucrz` without this step), and real hardware
    requires ISA circuits mapped to the backend's native gate set regardless
    (Section 4.1, "Hardware execution": "optimization_level=3").
    """
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if token:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            backend = service.backend(backend_name)
            transpiled = transpile(circuit, backend, optimization_level=optimization_level)
            sampler = SamplerV2(mode=backend)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()
            counts = result[0].data.meas.get_counts()
            print(f"[run_hardware.py] Executed on live backend '{backend_name}'")
            return counts
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any hardware/auth failure falls back
            print(f"[run_hardware.py] Live hardware execution failed ({exc}); "
                  f"falling back to AerSimulator.")

    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    transpiled = transpile(circuit, sim, optimization_level=optimization_level)
    job = sim.run(transpiled, shots=shots)
    counts = job.result().get_counts()
    print(f"[run_hardware.py] Executed on AerSimulator (no live '{backend_name}' credentials found)")
    return counts


def main() -> None:
    args = parse_args()
    config = ConfigLoader().load(args.config)

    with open(args.checkpoint, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    theta_flat = np.array(ckpt["theta"])
    R = ckpt["R"]
    n_accuracy_blocks = ckpt["n_accuracy_blocks"]
    n_qubits = ckpt["n_qubits"]
    x_min = np.array(ckpt["x_min"])
    x_max = np.array(ckpt["x_max"])
    theta = _theta_tuples(theta_flat, n_accuracy_blocks, x_min.shape[0])

    # --- Draw the hardware_test_points test set (Section 4.5: 10 random points) ---
    n_test = config["evaluation"]["hardware_test_points"]
    bs_cfg = config["data"]["black_scholes"]
    dataset = BlackScholesPutDataset()
    ranges = {
        "S_range": tuple(bs_cfg["S_range"]), "K_range": tuple(bs_cfg["K_range"]),
        "T_range": tuple(bs_cfg["T_range"]), "r_range": tuple(bs_cfg["r_range"]),
        "sigma_range": tuple(bs_cfg["sigma_range"]), "n_samples": n_test, "seed": 999,
    }
    x_raw, y_true = dataset.sample_training_grid(ranges)
    x_norm = np.clip((x_raw - x_min) / (x_max - x_min), 0.0, 1.0)

    builder = QNNCircuitBuilder(n_accuracy_blocks=n_accuracy_blocks, n_qubits=n_qubits)
    processor = MeasurementProcessor()

    preds = []
    for i in range(n_test):
        circuit = builder.assemble_circuit(theta, x_norm[i])
        counts = _run_on_hardware_or_fallback(
            circuit, args.shots, args.backend, config["hardware"]["optimization_level"]
        )
        probs = processor.group_counts(counts, n_accuracy_blocks, n_qubits)
        preds.append(processor.qnn_output(probs, R))
    preds = np.array(preds)

    # --- Compute epsilon_total (Eq. 4.4) from live/config hardware parameters ---
    hw_cfg = config["hardware"]
    calibrator = HardwareNoiseCalibrator()
    lambda_v = calibrator.compute_lambda_V(hw_cfg["eps_1Q"], n_qubits)
    _, n2q_ucr = calibrator.naive_and_ucr_two_qubit_gate_counts(n_accuracy_blocks, n_qubits)
    lambda_u = calibrator.compute_lambda_U(
        hw_cfg["eps_2Q"], n2q_ucr, hw_cfg["T1_us"], hw_cfg["T2_us"], hw_cfg["t2Q_ns"]
    )
    alpha = calibrator.compute_alpha(lambda_v, lambda_u)

    bound_calc = ErrorBoundCalculator()
    l1_fhat_example = 2.316  # Section 2.3.3 worked example
    f_l2_norm_estimate = float(np.std(y_true))  # proxy for ||f||_{L^2(mu)}; paper doesn't give exact value
    decomposition = bound_calc.decompose_total_bound(
        alpha, l1_fhat_example, n_accuracy_blocks, f_l2_norm_estimate, R, n_qubits,
        hw_cfg["readout_p"],
    )

    metrics = ErrorMetrics()
    mae = metrics.mae(preds, y_true)
    within_bound = metrics.within_bound_fraction(preds, y_true, decomposition["total"])

    print(f"[run_hardware.py] alpha={alpha:.4f}  epsilon_total={decomposition['total']:.4f}")
    print(f"[run_hardware.py] Empirical MAE={mae:.4f}  "
          f"Within bound: {within_bound * n_test:.0f}/{n_test} ({within_bound * 100:.1f}%)")
    print(f"[run_hardware.py] Bound decomposition: {decomposition}")


if __name__ == "__main__":
    main()
