#!/usr/bin/env python3
"""
inference.py -- Single-run: build the trained CVA oracle, run one CABIQAE
estimation trajectory, and print the recovered monetary CVA value
(arXiv:2607.12990, Eq. 37).

Example:
    python inference.py --config configs/config.yaml --checkpoint-dir checkpoints/ --epsilon 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quantum_cva.circuits import CVAOracle, NativeTreeCRCA, QCBMStatePreparation, SnakeCRCA
from quantum_cva.estimation import CABIQAE
from quantum_cva.utils import load_config, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--backend", type=str, default="statevector",
        choices=["statevector", "aer_noisy", "ibm_basquecountry"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.training["seed"])

    checkpoint_dir = Path(args.checkpoint_dir)
    m = cfg.model["num_time_qubits_m"]
    n_asset_total = cfg.model["num_asset_qubits_per_underlying"] * cfg.model["num_underlyings_d"]

    print("Assembling trained CVA oracle A_Theta...")
    qcbm = QCBMStatePreparation(num_time_qubits=m, num_asset_qubits=n_asset_total)
    g_circuit = qcbm.build_circuit(cfg.model["qcbm_layers_L"]).assign_parameters(
        np.load(checkpoint_dir / "qcbm_params.npy")
    )
    r_v = SnakeCRCA(num_time_qubits=m, num_asset_qubits=n_asset_total)
    rv_circuit = r_v.build_circuit(cfg.model["crca_rv_layers_L"]).assign_parameters(
        np.load(checkpoint_dir / "crca_rv_params.npy")
    )
    r_p = NativeTreeCRCA(num_time_qubits=m)
    rp_circuit = r_p.build_circuit(cfg.model["crca_rp_layers_L"]).assign_parameters(
        np.load(checkpoint_dir / "crca_rp_params.npy")
    )
    r_q = NativeTreeCRCA(num_time_qubits=m)
    rq_circuit = r_q.build_circuit(cfg.model["crca_rq_layers_L"]).assign_parameters(
        np.load(checkpoint_dir / "crca_rq_params.npy")
    )

    oracle = CVAOracle(num_register_qubits=m + n_asset_total, num_ancillas=cfg.model["num_ancillas"])
    a_theta = oracle.assemble(g_circuit, rv_circuit, rp_circuit, rq_circuit)

    if args.backend != "statevector":
        raise NotImplementedError(
            "Noisy/hardware inference requires qiskit-aer noise models or "
            "IBM Quantum credentials; see hardware/backend_manager.py and "
            "run_hardware_calibration.py for the hardware-replay path used "
            "in the paper's reported results."
        )

    print("Running one CABIQAE trajectory (statevector-exact ideal regime)...")
    a_true = oracle.marked_amplitude_statevector(a_theta)
    theta_true = np.arcsin(np.sqrt(a_true))
    rng = np.random.default_rng(cfg.training["seed"])

    def executor(k: int, n_shots: int):
        K = 2 * k + 1
        p_true = np.sin(K * theta_true) ** 2
        return int(rng.binomial(n_shots, p_true)), n_shots

    cabiqae = CABIQAE(c0=1.0, tau_c=1e12, b=0.5, rho_min=cfg.evaluation["rho_min"])
    result = cabiqae.estimate(
        executor,
        epsilon=args.epsilon,
        alpha=cfg.evaluation["failure_probability_alpha"],
        n_batch=cfg.evaluation["shots_per_batch_cva"],
    )

    # NOTE: C_v, C_p, C_q rescaling constants must come from the same
    # finite-grid construction used in train.py; recompute or persist them
    # alongside checkpoints for a fully monetary CVA readout (Eq. 37).
    print(f"\n{result}")
    print(f"Marked amplitude a_CVA (point estimate): {result.a_hat:.6f}")
    print(
        "To convert to monetary units: CVA_AE = M*(1-R_CVA)*C_v*C_p*C_q*a_CVA "
        "(Eq. 37) -- persist C_v, C_p, C_q from train.py's grid-builder output "
        "to complete this conversion."
    )


if __name__ == "__main__":
    main()
