"""
Q-Ising Training Script — Full Pipeline Entrypoint.
Runs all three stages: Ising inference → State construction → CQL training.

Usage:
    python train.py --config configs/sbm_default.yaml
    python train.py --config configs/village_default.yaml --ensemble
    python train.py --config configs/sbm_default.yaml --debug

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Q-Ising: Dynamic Treatment Policy Learning on Networks"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use MCMC ensemble policy (Stage 3). Default: EMVS point estimate.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed from config")
    parser.add_argument("--village-id", type=int, default=None,
                        help="Single village index to run (village experiment only)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (not yet implemented)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: reduce data size and training steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build all components but skip training")
    return parser.parse_args()


def run_sbm_experiment(cfg, ensemble: bool, debug: bool, dry_run: bool):
    """Run Q-Ising on the Stochastic Block Model (Section 5.1)."""
    from q_ising.data.network import NetworkData
    from q_ising.data.sis_simulator import SISSimulator
    from q_ising.models.ising import DynamicIsingModel
    from q_ising.models.state_constructor import StateConstructor
    from q_ising.training.cql_trainer import CQLTrainer
    from q_ising.training.ensemble_trainer import EnsembleTrainer
    from q_ising.evaluation.baselines import (
        RandomPolicy, DegreePolicy, LIRPolicy, DegreeBinPolicy
    )
    from q_ising.evaluation.metrics import PolicyEvaluator
    from q_ising.utils.sbm_generator import generate_sbm, get_sbm_block_labels

    print("\n=== Stage 0: Build SBM Network ===")
    T_train = 10 if debug else cfg.training.T_train
    max_steps = 1000 if debug else cfg.cql.max_steps

    M = generate_sbm(
        n_per_block=cfg.sis.n_per_block,
        p_in=cfg.sis.p_in,
        p_out=cfg.sis.p_out,
        seed=cfg.seed,
    )
    bin_labels = get_sbm_block_labels(cfg.sis.n_per_block)
    network = NetworkData(M=M, bin_labels=bin_labels)
    print(f"  {network}")

    print("\n=== Stage 0: Generate Observational Panel via Random Policy ===")
    simulator = SISSimulator(
        network=network,
        spread_rates=cfg.sis.spread_rates,
        churn_rates=cfg.sis.churn_rates,
        spread_by_bin=True,
    )
    random_policy = RandomPolicy(network)
    panel = simulator.generate_panel(T=T_train, policy=random_policy, seed=cfg.seed)
    print(f"  {panel}")

    print("\n=== Stage 1: Ising Inference ===")
    t0 = time.time()
    ising = DynamicIsingModel(network=network, config=cfg.ising)
    if ensemble or cfg.ising.estimation_method == "mcmc":
        print("  Running MCMC (this may take several minutes)...")
        if not dry_run:
            ising.fit_mcmc(panel, seed=cfg.seed)
    else:
        print("  Running EMVS...")
        if not dry_run:
            ising.fit_emvs(panel)
    print(f"  Ising inference done in {time.time()-t0:.1f}s")

    print("\n=== Stage 1b: State Construction ===")
    state_ctor = StateConstructor(ising_model=ising, network=network)
    if not dry_run:
        states = state_ctor.build_all_states(panel)
        print(f"  States shape: {states.shape}")  # [T+1, 2K]

    print("\n=== Stage 2: Offline Q-Learning (CQL) ===")
    cfg.cql.max_steps = max_steps
    state_dim = 2 * network.K

    if not dry_run:
        # Build transitions
        transitions = panel.to_rl_transitions(states=states, bin_labels=network.bin_labels)

        if ensemble and ising._mcmc_draws:
            print(f"  Training ensemble of {cfg.cql.n_ensemble_agents} CQL agents...")
            draws_to_use = ising.get_mcmc_draws()[:cfg.cql.n_ensemble_agents]
            state_seqs = state_ctor.build_states_ensemble(panel, draws_to_use)
            ensemble_transitions = [
                panel.to_rl_transitions(states=s, bin_labels=network.bin_labels)
                for s in state_seqs
            ]
            ens_trainer = EnsembleTrainer(K=network.K, state_dim=state_dim, config=cfg.cql)
            ens_trainer.train_from_transitions(ensemble_transitions)
            q_ising_policy_raw = ens_trainer.majority_vote_policy
        else:
            trainer = CQLTrainer(K=network.K, state_dim=state_dim, config=cfg.cql)
            trainer.train(transitions)
            q_ising_policy_raw = trainer.get_policy()

    print("\n=== Stage 3: Evaluation ===")
    if not dry_run:
        H = cfg.training.H_test
        n_runs = 5 if debug else cfg.training.n_test_runs

        # Wrap policies to use (y, t) interface
        def q_ising_policy(y, t=0):
            s = state_ctor.build_state(y, t)
            bin_act = q_ising_policy_raw(s)
            members = network.get_bin_members(int(bin_act))
            return int(np.random.choice(members))

        evaluator = PolicyEvaluator(simulator=simulator, network=network)
        policies = {
            "Random": RandomPolicy(network),
            "Degree": DegreePolicy(network),
            "LIR": LIRPolicy(network),
            "DegreeBin": DegreeBinPolicy(network),
            "Q-Ising": q_ising_policy,
        }
        results = evaluator.compare_policies(policies, H=H, n_runs=n_runs)

        print("\n=== Results (Mean ± Std adoption rate) ===")
        for name, r in results.items():
            print(f"  {name:15s}: {r.mean_reward:.4f} ± {r.std_reward:.4f}")

        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "sbm_results.npy",
                {name: r.rewards for name, r in results.items()})
        print(f"\n  Results saved to {output_dir}/sbm_results.npy")


def run_village_experiment(cfg, village_id: int, ensemble: bool, debug: bool, dry_run: bool):
    """Run Q-Ising on Indian microfinance village networks (Section 5.2)."""
    print(f"\n=== Village {village_id} Experiment ===")
    print("  NOTE: Requires village adjacency data in data/villages/")
    print("  Run: python data/download_villages.py to obtain data")

    data_path = Path(cfg.paths.data_dir) / f"village_{village_id}_adjacency.npy"
    if not data_path.exists():
        print(f"  ERROR: {data_path} not found. Run download_villages.py first.")
        return

    from q_ising.data.network import NetworkData
    M = np.load(str(data_path))
    network = NetworkData(M=M)
    network.assign_bins(
        method=cfg.network.bin_method,
        min_size=cfg.network.min_community_size,
    )

    if network.K < 2:
        print(f"  Skipping village {village_id}: fewer than 2 communities detected.")
        return

    print(f"  {network}")
    # ... rest of pipeline identical to SBM; omitted for brevity
    # Full implementation follows the same 3-stage pattern as run_sbm_experiment()
    print("  [Village pipeline not fully implemented in this stub — see run_sbm_experiment]")


def main():
    args = parse_args()

    # Load config
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from q_ising.utils.config import load_config, set_global_seed

    cfg = load_config(args.config)

    # Overrides
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir

    set_global_seed(cfg.seed)

    print(f"Q-Ising | experiment={cfg.experiment} | seed={cfg.seed} | ensemble={args.ensemble}")
    print(f"Config: {args.config}")

    if args.dry_run:
        print("[DRY RUN] Building components only — skipping actual training.")

    if cfg.experiment == "sbm":
        run_sbm_experiment(cfg, ensemble=args.ensemble, debug=args.debug, dry_run=args.dry_run)
    elif cfg.experiment == "village":
        village_ids = [args.village_id] if args.village_id is not None else list(range(42))
        for vid in village_ids:
            run_village_experiment(cfg, village_id=vid, ensemble=args.ensemble,
                                   debug=args.debug, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unknown experiment: {cfg.experiment}")


if __name__ == "__main__":
    main()
