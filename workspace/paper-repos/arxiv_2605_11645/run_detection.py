#!/usr/bin/env python3
"""
run_detection.py
Run the GeomHerd detection pipeline on CWS or Vicsek substrate.
Paper: arXiv:2605.11645

Usage:
    python run_detection.py --substrate cws --kappa 1.8 --seeds 80
    python run_detection.py --substrate vicsek --eta 1.6 --seeds 20
    python run_detection.py --substrate cws --operating_point recall --llm_mode false
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from geomherd.pipeline.geomherd_pipeline import GeomHerdOutput, GeomHerdPipeline
from geomherd.simulation.cws_substrate import CWSSubstrate
from geomherd.simulation.llm_agent import build_agent_population
from geomherd.simulation.vicsek_substrate import VicsekSubstrate
from geomherd.utils.config import GeomHerdConfig, set_global_seed


def run_cws_trajectory(
    kappa: float,
    seed: int,
    cfg: GeomHerdConfig,
    llm_mode: bool = False,
) -> Dict:
    """Run one CWS trajectory and return detection results."""
    set_global_seed(seed)
    cws_cfg = cfg.simulation.cws
    substrate = CWSSubstrate(
        N=cws_cfg.N_agents,
        na=cws_cfg.N_assets,
        kappa=kappa,
        seed=seed,
        sbase=cws_cfg.sbase,
        spost=cws_cfg.spost,
    )

    # Override pipeline N_agents from substrate
    cfg.simulation.cws.N_agents = substrate.N
    pipeline = GeomHerdPipeline(cfg)
    pipeline.reset()

    # Optionally build LLM agents (default: rule-based fallback)
    if llm_mode:
        agents = build_agent_population(
            N=substrate.N, llm_mode=True,
            model=cfg.simulation.llm_model, seed=seed
        )
    else:
        agents = None  # use CWS dynamics directly

    actions = substrate.reset(seed=seed)
    order_params = []
    geomherd_outputs: List[Optional[GeomHerdOutput]] = []
    event_threshold = cfg.detection.herding_event_threshold

    herding_event_t: Optional[int] = None
    consecutive_above = 0

    for t in range(cws_cfg.T_steps):
        if agents is not None:
            # LLM-driven agents decide based on market state
            prices = substrate._prices.tolist()
            returns_now = [0.0] * substrate.na  # simplified
            majority = int(np.bincount(actions, minlength=3).argmax())
            market_state = {
                "prices": prices,
                "returns": returns_now,
                "majority_action": majority,
                "fundamental": 100.0,
            }
            actions = np.array([a.decide(market_state) for a in agents], dtype=np.int32)
        else:
            actions, _, info = substrate.step()

        # Push to pipeline
        output = pipeline.step(actions, t)
        geomherd_outputs.append(output)
        order_param = substrate.get_order_parameter()
        order_params.append(order_param)

        # Detect herding event: Va(t) > theta_event
        if order_param > event_threshold:
            consecutive_above += 1
            if consecutive_above >= 1 and herding_event_t is None:
                herding_event_t = t
        else:
            consecutive_above = 0

    alarm_plus_t = pipeline.first_alarm_time("plus")
    alarm_minus_t = pipeline.first_alarm_time("minus")

    is_supercritical = kappa > 1.0
    result = {
        "kappa": kappa,
        "seed": seed,
        "is_supercritical": is_supercritical,
        "herding_event_t": herding_event_t,
        "alarm_plus_t": alarm_plus_t,
        "alarm_minus_t": alarm_minus_t,
        "lead_plus": (herding_event_t - alarm_plus_t)
            if (herding_event_t is not None and alarm_plus_t is not None) else None,
        "lead_minus": (herding_event_t - alarm_minus_t)
            if (herding_event_t is not None and alarm_minus_t is not None) else None,
        "order_params": order_params,
        "n_snapshots": sum(1 for o in geomherd_outputs if o is not None),
    }
    # Collect time series
    result["kappa_bar_plus_series"] = [
        o.kappa_bar_plus for o in geomherd_outputs if o is not None
    ]
    result["beta_minus_series"] = [
        o.beta_minus for o in geomherd_outputs if o is not None
    ]
    result["tau_sing_series"] = [
        o.tau_sing for o in geomherd_outputs if o is not None
    ]
    result["v_eff_series"] = [
        o.V_eff for o in geomherd_outputs if o is not None
    ]
    return result


def run_vicsek_trajectory(eta: float, seed: int, cfg: GeomHerdConfig) -> Dict:
    """Run one Vicsek trajectory and return detection results."""
    from geomherd.graph.agent_graph import AgentGraph
    from geomherd.geometry.ricci_curvature import OllivierRicciComputer

    set_global_seed(seed)
    vc = cfg.simulation.vicsek
    substrate = VicsekSubstrate(
        N=vc.N_particles, eta=eta, speed=vc.speed,
        radius=vc.radius, seed=seed
    )
    substrate.reset(seed=seed)

    # Vicsek uses a separate k-NN agent graph (not rolling window)
    # Appendix G: k-NN (k=10) on heading sequence
    orc = OllivierRicciComputer(alpha=cfg.curvature.alpha)

    polarisations = []
    kappa_at_event: Optional[float] = None
    herding_event_t: Optional[int] = None
    consecutive_above = 0
    headings_history = []

    for t in range(vc.T_steps):
        headings, info = substrate.step()
        pol = info["polarisation"]
        polarisations.append(pol)
        headings_history.append(headings.copy())

        # Downsample: evaluate graph every snapshot_stride steps
        if t % vc.snapshot_stride == 0 and t > vc.snapshot_stride:
            # Build k-NN agent graph from heading sequence
            W = _build_knn_heading_graph(
                headings_history[-vc.snapshot_stride:], k=vc.knn_k, N=vc.N_particles
            )
            kappa_dict = orc.compute(W)
            kappa_all = orc.mean_curvature_all(kappa_dict)

            # Record kappa at herding event
            if herding_event_t is not None and kappa_at_event is None:
                kappa_at_event = kappa_all

        # Herding event: polarisation > 0.5 for 3 consecutive steps (Appendix G)
        if pol > vc.polarisation_threshold:
            consecutive_above += 1
            if consecutive_above >= 3 and herding_event_t is None:
                herding_event_t = t
        else:
            consecutive_above = 0

    return {
        "eta": eta,
        "seed": seed,
        "is_ordered": eta < vc.eta_critical,
        "herding_event_t": herding_event_t,
        "kappa_at_event": kappa_at_event,
        "polarisations": polarisations,
    }


def _build_knn_heading_graph(
    headings_window: List[np.ndarray], k: int, N: int
) -> np.ndarray:
    """Build binary k-NN agent graph from heading time series (Appendix G)."""
    # Stack headings: [T_window, N]
    H = np.stack(headings_window, axis=0)  # [T, N]
    W = np.zeros((N, N), dtype=np.float32)
    # Pairwise angular similarity: cos(theta_i - theta_j) averaged over window
    for t_idx in range(H.shape[0]):
        angles = H[t_idx]  # [N]
        # Agreement: same k-NN neighborhood in heading space
        diff = np.abs(angles[:, None] - angles[None, :])
        diff = np.minimum(diff, 2 * np.pi - diff)  # circular distance
        W += (diff < (np.pi / k)).astype(np.float32)
    W /= H.shape[0]
    np.fill_diagonal(W, 0.0)
    # Binary threshold at 0.5
    W = (W > 0.5).astype(np.float32)
    return W


def main():
    parser = argparse.ArgumentParser(description="GeomHerd Detection Pipeline")
    parser.add_argument("--substrate", choices=["cws", "vicsek"], default="cws")
    parser.add_argument("--kappa", type=float, default=1.8,
                        help="CWS coupling parameter (supercritical if > 1.0)")
    parser.add_argument("--eta", type=float, default=1.6,
                        help="Vicsek noise parameter (ordered if < 1.6)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds to run")
    parser.add_argument("--operating_point", choices=["recall", "precision"],
                        default="precision")
    parser.add_argument("--llm_mode", type=lambda x: x.lower() == "true",
                        default=False, help="Use LLM agents (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--output_dir", type=str, default="results/detection/")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: run 2 seeds, 200 steps")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build all components but don't run simulation")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        cfg = GeomHerdConfig.from_yaml(args.config)
    else:
        print(f"Config not found at {args.config}, using defaults.")
        cfg = GeomHerdConfig()

    cfg.detection.operating_point = args.operating_point

    if args.debug:
        args.seeds = 2
        cfg.simulation.cws.T_steps = 200
        cfg.simulation.vicsek.T_steps = 200
        print("[DEBUG] Running 2 seeds, 200 steps.")

    if args.dry_run:
        print("[DRY RUN] Building components...")
        pipeline = GeomHerdPipeline(cfg)
        print(f"  {pipeline}")
        print("[DRY RUN] All components initialized successfully. Exiting.")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    print(f"\nGeomHerd Detection | substrate={args.substrate} | "
          f"op={args.operating_point} | llm={args.llm_mode}")
    print(f"Running {args.seeds} seeds...")
    t0 = time.time()

    for s in range(args.seeds):
        seed = args.seed_offset + s
        if args.substrate == "cws":
            result = run_cws_trajectory(
                kappa=args.kappa, seed=seed, cfg=cfg, llm_mode=args.llm_mode
            )
            label = f"kappa={args.kappa}"
        else:
            result = run_vicsek_trajectory(eta=args.eta, seed=seed, cfg=cfg)
            label = f"eta={args.eta}"

        all_results.append(result)
        lead_str = (f"lead_plus={result.get('lead_plus', 'n/a')}" if args.substrate == "cws"
                    else f"kappa@event={result.get('kappa_at_event', 'n/a'):.3f}" if result.get('kappa_at_event') else "no event")
        print(f"  seed={seed:3d} | {label} | {lead_str}")

    elapsed = time.time() - t0
    out_path = Path(args.output_dir) / f"{args.substrate}_{label.replace('=','')}_op{args.operating_point}_s{args.seeds}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDone in {elapsed:.1f}s. Results saved to {out_path}")

    # Quick summary
    if args.substrate == "cws":
        leads = [r["lead_plus"] for r in all_results if r["lead_plus"] is not None]
        if leads:
            print(f"Median lead (alarm+ vs event): {np.median(leads):.0f} steps "
                  f"(n={len(leads)}/{args.seeds} with both alarm and event)")
        else:
            print("No co-firing trajectories at this operating point.")


if __name__ == "__main__":
    main()
