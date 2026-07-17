#!/usr/bin/env python3
"""
train.py -- Train QCBM (G_theta) and CRCA (R_v, R_p, R_q) variational
parameters against the finite-grid classical CVA benchmark
(arXiv:2607.12990, Section 3.2.3).

Example:
    python train.py --config configs/config.yaml --block all --output-dir checkpoints/
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np

from quantum_cva.circuits import NativeTreeCRCA, QCBMStatePreparation, SnakeCRCA
from quantum_cva.finance import (
    BlackScholesPricer,
    FiniteGridBuilder,
    Instrument,
    MultiAssetGBMSimulator,
)
from quantum_cva.training import VariationalTrainer
from quantum_cva.utils import load_config, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--block",
        type=str,
        default="all",
        choices=["all", "qcbm", "crca_rp", "crca_rq", "crca_rv"],
        help="Which block to train",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints/", help="Where to save trained parameters")
    parser.add_argument("--seed", type=int, default=100000, help="Random seed")
    parser.add_argument(
        "--gradient-method", type=str, default="parameter_shift", choices=["parameter_shift", "spsa"],
        help="parameter_shift is exact but expensive for many-parameter blocks (e.g. R_v); "
             "spsa is a cheap (2 evals/step) stochastic alternative -- see training/trainer.py docstring",
    )
    return parser.parse_args()


def build_finite_grid(cfg):
    """Reproduce Section 3.2.1/4.2: simulate paths, build finite grid inputs."""
    data_cfg = cfg.data
    m = cfg.model["num_time_qubits_m"]
    expected_M = 2**m
    if data_cfg["monitoring_dates_M"] != expected_M:
        raise ValueError(
            f"data.monitoring_dates_M ({data_cfg['monitoring_dates_M']}) must "
            f"equal 2**model.num_time_qubits_m ({expected_M}) so the "
            f"monitoring dates encode exactly in the time register with no "
            f"unused basis states (paper Section 2.2.1, footnote 3)."
        )
    instruments = [Instrument(**inst) for inst in data_cfg["netting_set"]]
    underlyings = list(data_cfg["spots"].keys())
    monitoring_dates = np.linspace(
        data_cfg["cva_maturity_years"] / data_cfg["monitoring_dates_M"],
        data_cfg["cva_maturity_years"],
        data_cfg["monitoring_dates_M"],
    )

    # Synthetic/placeholder market curves (replace with real LSEG data via
    # data/download.sh once available -- see data/README_data.md)
    r_flat = 0.025
    discount_vec = np.exp(-r_flat * monitoring_dates)
    default_incr_vec = np.full(len(monitoring_dates), 2.0e-4)

    vol_level = {"EURO STOXX 50": 0.20, "SMI": 0.16}
    volatilities = {u: np.full(len(monitoring_dates), vol_level[u]) for u in underlyings}

    d = len(underlyings)
    if d == 2:
        rho = data_cfg["correlation_SX5E_SMI"]
        correlation_matrix = np.array([[1.0, rho], [rho, 1.0]])
    else:
        correlation_matrix = np.eye(d)

    simulator = MultiAssetGBMSimulator(
        spots=data_cfg["spots"],
        dividend_yields=data_cfg["dividend_yields"],
        risk_free_rate=r_flat,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
    )
    paths = simulator.simulate_paths(
        n_paths=data_cfg["n_mc_paths"], monitoring_dates=monitoring_dates, seed=cfg.training["seed"]
    )

    pricer = BlackScholesPricer()
    grid_builder = FiniteGridBuilder(pricer)

    n_qubits_per_asset = cfg.model["num_asset_qubits_per_underlying"]
    n_bins = 2**n_qubits_per_asset
    bin_edges_list = []
    for k, u in enumerate(underlyings):
        mu = float(np.mean(paths[:, -1, k]))
        sigma = float(np.std(paths[:, -1, k]))
        lower, upper = grid_builder.truncate_domain(mu, sigma)
        bin_edges_list.append(grid_builder.bin_edges(lower, upper, n_bins))

    prob_tensor = grid_builder.build_probability_tensor(paths, bin_edges_list, monitoring_dates)
    exposure_tensor = grid_builder.build_exposure_tensor(
        instruments, bin_edges_list, underlyings, monitoring_dates, r_flat,
        data_cfg["dividend_yields"], volatilities,
    )
    c_v, c_p, c_q = grid_builder.rescale_constants(exposure_tensor, discount_vec, default_incr_vec)

    return {
        "prob_tensor": prob_tensor,
        "exposure_tensor": exposure_tensor,
        "discount_vec": discount_vec,
        "default_incr_vec": default_incr_vec,
        "C_v": c_v,
        "C_p": c_p,
        "C_q": c_q,
        "monitoring_dates": monitoring_dates,
        "n_asset_qubits": n_qubits_per_asset,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(args.seed, deterministic=cfg.hardware.get("deterministic", True))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building finite-grid classical benchmark inputs (Section 3.2.1)...")
    grid = build_finite_grid(cfg)

    m = cfg.model["num_time_qubits_m"]
    n_asset_total = cfg.model["num_asset_qubits_per_underlying"] * cfg.model["num_underlyings_d"]

    trainer = VariationalTrainer(
        optimizer=cfg.training["optimizer"],
        learning_rate=cfg.training["learning_rate"],
        seed=args.seed,
        gradient_method=args.gradient_method,
    )

    results = {}

    if args.block in ("all", "qcbm"):
        print("\n=== Training QCBM (G_theta) ===")
        qcbm = QCBMStatePreparation(num_time_qubits=m, num_asset_qubits=n_asset_total)
        circuit = qcbm.build_circuit(num_layers=cfg.model["qcbm_layers_L"])
        target_dist = grid["prob_tensor"].flatten()
        target_dist = target_dist / target_dist.sum()

        loss_fn = functools.partial(
            qcbm.cross_entropy_loss,
            circuit,
            target_dist=target_dist,
            eps_num=cfg.model["numerical_floor_eps_num"],
        )
        params, history, _ = trainer.train_qcbm(
            loss_fn, circuit.num_parameters, cfg.training["qcbm_iterations"],
            log_every_n_steps=cfg.training["log_every_n_steps"],
        )
        np.save(output_dir / "qcbm_params.npy", params)
        results["qcbm_final_loss"] = history[-1]
        results["qcbm_final_kl"] = qcbm.kl_divergence(circuit, params, target_dist)
        print(f"QCBM training complete. Final loss={history[-1]:.6e}, KL={results['qcbm_final_kl']:.6e}")

    if args.block in ("all", "crca_rp"):
        print("\n=== Training CRCA R_p (discount factor) ===")
        r_p = NativeTreeCRCA(num_time_qubits=m)
        circuit = r_p.build_circuit(num_layers=cfg.model["crca_rp_layers_L"])
        target = grid["discount_vec"] / grid["C_p"]
        loss_fn = functools.partial(r_p.mse_loss, circuit, target_fn=target)
        params, history, _ = trainer.train_crca(
            loss_fn, circuit.num_parameters, cfg.training["crca_rp_rq_iterations"],
            log_every_n_steps=cfg.training["log_every_n_steps"],
        )
        np.save(output_dir / "crca_rp_params.npy", params)
        results["rp_final_loss"] = history[-1]
        print(f"R_p training complete. Final loss={history[-1]:.6e}")

    if args.block in ("all", "crca_rq"):
        print("\n=== Training CRCA R_q (default probability) ===")
        r_q = NativeTreeCRCA(num_time_qubits=m)
        circuit = r_q.build_circuit(num_layers=cfg.model["crca_rq_layers_L"])
        target = grid["default_incr_vec"] / grid["C_q"]
        loss_fn = functools.partial(r_q.mse_loss, circuit, target_fn=target)
        params, history, _ = trainer.train_crca(
            loss_fn, circuit.num_parameters, cfg.training["crca_rp_rq_iterations"],
            log_every_n_steps=cfg.training["log_every_n_steps"],
        )
        np.save(output_dir / "crca_rq_params.npy", params)
        results["rq_final_loss"] = history[-1]
        print(f"R_q training complete. Final loss={history[-1]:.6e}")

    if args.block in ("all", "crca_rv"):
        print("\n=== Training CRCA R_v (positive exposure) -- most demanding block ===")
        r_v = SnakeCRCA(num_time_qubits=m, num_asset_qubits=n_asset_total)
        circuit = r_v.build_circuit(num_layers=cfg.model["crca_rv_layers_L"])
        target = grid["exposure_tensor"] / grid["C_v"]
        loss_fn = functools.partial(
            r_v.mse_loss, circuit, target_tensor=target,
            asset_qubits_per_underlying=grid["n_asset_qubits"],
        )
        params, history, _ = trainer.train_crca(
            loss_fn, circuit.num_parameters, cfg.training["crca_rv_iterations"],
            log_every_n_steps=cfg.training["log_every_n_steps"],
        )
        np.save(output_dir / "crca_rv_params.npy", params)
        results["rv_final_loss"] = history[-1]
        print(f"R_v training complete. Final loss={history[-1]:.6e}")

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll checkpoints written to {output_dir}/")


if __name__ == "__main__":
    main()
