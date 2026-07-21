"""
160-trial TPE hyperparameter search for the RL experiment.

Implements the hyperparameter optimisation described in Appendix I.4:
  - 160 trials with Tree-structured Parzen Estimator (TPE) sampler
  - Top 10 configurations re-evaluated with 10 seeds each

Usage:
    python scripts/run_rl_tpe.py --env BreakoutNoFrameskip-v4 --parameterization natural
    python scripts/run_rl_tpe.py --env SeaquestNoFrameskip-v4 --parameterization softmax

Requirements:
    pip install optuna
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="RL TPE hyperparameter search (Appendix I.4)")
    p.add_argument("--env",            type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--parameterization",
                   choices=["natural", "softmax"], default="natural")
    p.add_argument("--config",         type=str, default="configs/rl.yaml")
    p.add_argument("--n_trials",       type=int, default=160,
                   help="Number of TPE trials (paper: 160)")
    p.add_argument("--n_final_seeds",  type=int, default=10,
                   help="Seeds for top-k re-evaluation (paper: 10)")
    p.add_argument("--top_k",          type=int, default=10,
                   help="Number of best configs to re-evaluate (paper: 10)")
    p.add_argument("--storage",        type=str, default=None,
                   help="Optuna storage URL (default: in-memory)")
    p.add_argument("--output_dir",     type=str, default="results/rl/")
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--quick_timesteps", type=int, default=200_000,
                   help="Timesteps per trial (keep short for search; paper uses 8M for final)")
    return p.parse_args()


def objective(trial, config, device, env_name, timesteps):
    """Optuna objective: train one PPO trial and return mean episodic return."""
    import torch
    from src.catnat.models.ppo_agent import PPOAgent
    from src.catnat.training.trainers import PPOTrainer
    from src.catnat.data.atari_wrappers import make_atari_env
    from src.catnat.utils.reproducibility import set_seed
    import math

    # Sample hyperparameters from the ranges in Table 7 (Appendix I.4)
    config.training.lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
    config.training.num_steps = trial.suggest_int("num_steps", 32, 512, step=32)
    config.training.update_epochs = trial.suggest_int("update_epochs", 1, 16, step=2)
    config.training.clip_coef = trial.suggest_float("clip_coef", 0.01, 0.9)
    config.training.ent_coef  = trial.suggest_float("ent_coef", 0.0, 1.0)
    config.training.num_envs  = trial.suggest_int("num_envs", 8, 16, step=2)
    config.training.num_minibatches = trial.suggest_int("num_minibatches", 2, 16, step=4)
    config.training.max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 10.0, step=0.1)

    config.training.total_timesteps = timesteps
    config.env.name = env_name

    set_seed(trial.number)
    envs = [make_atari_env(env_name, seed=trial.number + i)
            for i in range(config.training.num_envs)]
    n_actions = envs[0].action_space.n
    K_required = 2 ** math.ceil(math.log2(n_actions))
    config.catnat.K = K_required

    agent = PPOAgent(config, n_actions=n_actions)
    trainer = PPOTrainer(agent, config, device)
    trainer.train(envs)

    # Evaluate
    from src.catnat.evaluation.metrics import RLMetrics
    eval_env = make_atari_env(env_name, seed=999)
    result = RLMetrics.episodic_return(eval_env, agent, device, n_episodes=5)
    eval_env.close()
    for env in envs:
        env.close()

    return result["mean"]


def main():
    args = parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    from src.catnat.utils.config import load_config
    from src.catnat.utils.reproducibility import get_device

    config = load_config(args.config)
    config.parameterization = args.parameterization
    device = get_device(args.device)

    # Silence optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n{'='*60}")
    print(f"RL TPE Hyperparameter Search")
    print(f"  Env:             {args.env}")
    print(f"  Parameterization: {args.parameterization}")
    print(f"  Trials:          {args.n_trials}")
    print(f"  Timesteps/trial: {args.quick_timesteps:,}")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=args.storage,
        study_name=f"rl_{args.env}_{args.parameterization}",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, config, device, args.env, args.quick_timesteps),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Top-k configurations
    all_trials = sorted(study.trials, key=lambda t: t.value or -1e9, reverse=True)
    top_k = all_trials[:args.top_k]

    print(f"\nTop {args.top_k} configurations:")
    for i, trial in enumerate(top_k):
        print(f"  [{i+1}] Return={trial.value:.1f} | {trial.params}")

    # Re-evaluate top-k with n_final_seeds seeds
    print(f"\nRe-evaluating top {args.top_k} configs with {args.n_final_seeds} seeds each...")

    import torch
    from src.catnat.models.ppo_agent import PPOAgent
    from src.catnat.training.trainers import PPOTrainer
    from src.catnat.data.atari_wrappers import make_atari_env
    from src.catnat.evaluation.metrics import RLMetrics
    from src.catnat.utils.reproducibility import set_seed
    import math, statistics

    final_results = []
    for i, trial in enumerate(top_k):
        # Apply hyperparams from this trial
        for k, v in trial.params.items():
            setattr(config.training, k, v)
        config.training.total_timesteps = 8_000_000   # Full training for final eval

        seed_returns = []
        for seed in range(args.n_final_seeds):
            set_seed(seed)
            envs = [make_atari_env(args.env, seed=seed + j)
                    for j in range(config.training.num_envs)]
            n_actions = envs[0].action_space.n
            K_required = 2 ** math.ceil(math.log2(n_actions))
            config.catnat.K = K_required

            agent = PPOAgent(config, n_actions=n_actions)
            trainer = PPOTrainer(agent, config, device)
            trainer.train(envs)

            eval_env = make_atari_env(args.env, seed=9999 + seed)
            result = RLMetrics.episodic_return(eval_env, agent, device, n_episodes=10)
            seed_returns.append(result["mean"])
            eval_env.close()
            for env in envs:
                env.close()

        mean_ret = statistics.mean(seed_returns)
        std_ret  = statistics.stdev(seed_returns)
        print(f"  Config {i+1}: {mean_ret:.1f} ± {std_ret:.1f}")
        final_results.append({
            "config_rank": i + 1,
            "params": trial.params,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "seed_returns": seed_returns,
        })

    # Save
    out_dir = Path(args.output_dir) / args.env / args.parameterization
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tpe_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    best = max(final_results, key=lambda r: r["mean_return"])
    print(f"\n{'='*60}")
    print(f"Best config: {best['mean_return']:.1f} ± {best['std_return']:.1f}")
    print(f"  Params: {best['params']}")
    print(f"Results saved to {out_dir / 'tpe_results.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
