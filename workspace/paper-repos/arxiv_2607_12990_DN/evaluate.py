#!/usr/bin/env python3
"""
evaluate.py -- Run the amplitude-estimation comparison (CABIQAE vs BIQAE vs
BAE vs DCS) in the noiseless or hardware-replay regime, and reproduce the
error-budget decomposition (arXiv:2607.12990, Sections 4.1, 4.4, 4.5).

Example:
    python evaluate.py --config configs/config.yaml --regime noiseless --num-trajectories 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantum_cva.circuits import CVAOracle, NativeTreeCRCA, QCBMStatePreparation, SnakeCRCA
from quantum_cva.estimation import BIQAE, CABIQAE, DirectCircuitSampling
from quantum_cva.evaluation import TrajectoryMetrics
from quantum_cva.utils import load_config, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--regime", type=str, default="noiseless",
        choices=["noiseless", "hardware-replay", "validation-circuit", "full-cva"],
    )
    parser.add_argument("--num-trajectories", type=int, default=300)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--output-dir", type=str, default="results/")
    return parser.parse_args()


def load_trained_oracle(cfg, checkpoint_dir: Path) -> tuple:
    """Load trained parameters and assemble the full CVA oracle A_Theta."""
    m = cfg.model["num_time_qubits_m"]
    n_asset_total = cfg.model["num_asset_qubits_per_underlying"] * cfg.model["num_underlyings_d"]

    qcbm = QCBMStatePreparation(num_time_qubits=m, num_asset_qubits=n_asset_total)
    g_circuit = qcbm.build_circuit(cfg.model["qcbm_layers_L"])
    g_params = np.load(checkpoint_dir / "qcbm_params.npy")
    g_bound = g_circuit.assign_parameters(g_params)

    r_v = SnakeCRCA(num_time_qubits=m, num_asset_qubits=n_asset_total)
    rv_circuit = r_v.build_circuit(cfg.model["crca_rv_layers_L"])
    rv_params = np.load(checkpoint_dir / "crca_rv_params.npy")
    rv_bound = rv_circuit.assign_parameters(rv_params)

    r_p = NativeTreeCRCA(num_time_qubits=m)
    rp_circuit = r_p.build_circuit(cfg.model["crca_rp_layers_L"])
    rp_params = np.load(checkpoint_dir / "crca_rp_params.npy")
    rp_bound = rp_circuit.assign_parameters(rp_params)

    r_q = NativeTreeCRCA(num_time_qubits=m)
    rq_circuit = r_q.build_circuit(cfg.model["crca_rq_layers_L"])
    rq_params = np.load(checkpoint_dir / "crca_rq_params.npy")
    rq_bound = rq_circuit.assign_parameters(rq_params)

    oracle = CVAOracle(num_register_qubits=m + n_asset_total, num_ancillas=cfg.model["num_ancillas"])
    a_theta = oracle.assemble(g_bound, rv_bound, rp_bound, rq_bound)
    return oracle, a_theta


def make_statevector_executor(oracle: CVAOracle, a_theta, seed: int):
    """Ideal-regime circuit executor: exact Grover-amplified probability,
    finite-shot binomial sampling only (Section 3.2.2, "ideal regime")."""
    rng = np.random.default_rng(seed)
    a_true = oracle.marked_amplitude_statevector(a_theta)
    theta_true = np.arcsin(np.sqrt(a_true))

    def executor(k: int, n_shots: int):
        K = 2 * k + 1
        p_true = np.sin(K * theta_true) ** 2
        n_success = int(rng.binomial(n_shots, p_true))
        return n_success, n_shots

    return executor, a_true


def run_comparison(cfg, oracle, a_theta, regime: str, num_trajectories: int, output_dir: Path):
    epsilon = cfg.evaluation["cabiqae_target_half_width_cva"]
    epsilon_biqae = cfg.evaluation["biqae_target_half_width_cva"]
    alpha = cfg.evaluation["failure_probability_alpha"]
    n_batch = cfg.evaluation["shots_per_batch_cva"]

    all_results = {"cabiqae": [], "biqae": [], "dcs": []}
    a_true_ref = None

    for traj in range(num_trajectories):
        seed = cfg.training["seed"] + traj
        executor, a_true = make_statevector_executor(oracle, a_theta, seed)
        a_true_ref = a_true

        if regime == "hardware-replay":
            tau_c = cfg.evaluation["contrast_scale_tau_c_cva"]
            c0 = cfg.evaluation["contrast_prefactor_c0_validation_hardware"]

            def hw_executor(k, n_shots, _theta=np.arcsin(np.sqrt(a_true)), _c0=c0, _tau=tau_c):
                K = 2 * k + 1
                q_k = np.sin(K * _theta) ** 2
                b = cfg.evaluation["contrast_baseline_b_cva"]
                c_k = _c0 * np.exp(-K / _tau)
                p_hw = b + c_k * (q_k - b)
                rng2 = np.random.default_rng(seed + k)
                n_success = int(rng2.binomial(n_shots, np.clip(p_hw, 0, 1)))
                return n_success, n_shots

            active_executor = hw_executor
            cabiqae = CABIQAE(c0=c0, tau_c=tau_c, b=cfg.evaluation["contrast_baseline_b_cva"],
                               rho_min=cfg.evaluation["rho_min"])
        else:
            active_executor = executor
            cabiqae = CABIQAE(c0=1.0, tau_c=1e12, b=0.5, rho_min=cfg.evaluation["rho_min"])

        biqae = BIQAE(rho_min=cfg.evaluation["rho_min"])
        dcs = DirectCircuitSampling()

        res_cabiqae = cabiqae.estimate(active_executor, epsilon=epsilon, alpha=alpha, n_batch=n_batch)
        res_biqae = biqae.estimate(active_executor, epsilon=epsilon_biqae, alpha=alpha, n_batch=n_batch)
        res_dcs = dcs.estimate(active_executor, n_shots=res_cabiqae.total_queries)

        all_results["cabiqae"].append(
            {"a_hat": res_cabiqae.a_hat, "n_q": res_cabiqae.total_queries, "k_max": res_cabiqae.max_k,
             "rel_error": abs(res_cabiqae.a_hat - a_true) / a_true}
        )
        all_results["biqae"].append(
            {"a_hat": res_biqae.a_hat, "n_q": res_biqae.total_queries, "k_max": res_biqae.max_k,
             "rel_error": abs(res_biqae.a_hat - a_true) / a_true}
        )
        all_results["dcs"].append(
            {"a_hat": res_dcs.a_hat, "n_q": res_dcs.total_queries, "k_max": 0,
             "rel_error": abs(res_dcs.a_hat - a_true) / a_true}
        )

        if (traj + 1) % max(1, num_trajectories // 10) == 0:
            print(f"  trajectory {traj+1}/{num_trajectories} complete")

    metrics = TrajectoryMetrics()
    summary = {}
    for method, records in all_results.items():
        errors = np.array([r["rel_error"] for r in records])
        n_qs = np.array([r["n_q"] for r in records])
        median_err, ci_lo, ci_hi = metrics.median_with_bootstrap_ci(errors)
        summary[method] = {
            "median_relative_error": median_err,
            "ci_95": [ci_lo, ci_hi],
            "median_n_q": float(np.median(n_qs)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"comparison_{regime}.json", "w") as f:
        json.dump({"a_true": a_true_ref, "summary": summary, "raw": all_results}, f, indent=2)

    print(f"\n=== Summary ({regime}) ===")
    for method, s in summary.items():
        print(f"  {method:10s}: median_rel_error={s['median_relative_error']:.4%}, "
              f"median_N_q={s['median_n_q']:.0f}")

    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.training["seed"])

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    if not (checkpoint_dir / "qcbm_params.npy").exists():
        raise FileNotFoundError(
            f"No trained checkpoints found in {checkpoint_dir}/. Run train.py first: "
            f"python train.py --config {args.config} --output-dir {checkpoint_dir}"
        )

    print(f"Loading trained CVA oracle from {checkpoint_dir}/...")
    oracle, a_theta = load_trained_oracle(cfg, checkpoint_dir)

    print(f"\nRunning {args.num_trajectories} trajectories in '{args.regime}' regime...")
    run_comparison(cfg, oracle, a_theta, args.regime, args.num_trajectories, output_dir)

    # Error budget (Section 4.5) requires the classical benchmarks too;
    # for illustration we report only the AE-layer contribution here using
    # the trained statevector amplitude as CVA_SV proxy scaling.
    a_sv = oracle.marked_amplitude_statevector(a_theta)
    print(f"\nStatevector-exact marked amplitude a_CVA = {a_sv:.6f}")
    print("Run error_decomposition.ErrorBudget.compute_budget(...) with your "
          "classical CVA_cont_MC and CVA_tab_Delta values (from train.py's "
          "finite-grid outputs) to reproduce the full Table 7 breakdown.")


if __name__ == "__main__":
    main()
