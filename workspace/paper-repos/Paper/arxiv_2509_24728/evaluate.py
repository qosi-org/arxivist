"""
Main evaluation entrypoint. Reproduces Tables 2, 3, and 4.

Usage:
    python evaluate.py --experiment vae --checkpoint checkpoints/vae/best.pt
    python evaluate.py --experiment gsl --checkpoint checkpoints/gsl/best.pt --theta_star 0.5
    python evaluate.py --experiment rl  --checkpoint checkpoints/rl/final.pt
    python evaluate.py --experiment vae --checkpoint checkpoints/vae/best.pt --compute_fim

Flags:
    --experiment    {gsl, vae, rl}   [REQUIRED]
    --checkpoint    Path to .pt checkpoint file [REQUIRED]
    --config        Path to YAML config
    --n_eval_samples [VAE] Number of IS samples for IWAE NLL (default: 512)
    --theta_star    [GSL] True theta* for MAE on θ metric
    --output_dir    Save results JSON to this directory
    --compute_fim   Run FIM diagonality verification (slow)
    --device        {cuda, cpu}
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate catnat models (arXiv:2509.24728)")
    p.add_argument("--experiment",  choices=["gsl", "vae", "rl"], required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--config",      type=str, default=None)
    p.add_argument("--n_eval_samples", type=int, default=512)
    p.add_argument("--theta_star",  type=float, default=0.5)
    p.add_argument("--output_dir",  type=str, default=None)
    p.add_argument("--compute_fim", action="store_true")
    p.add_argument("--device",      type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.catnat.utils.config import load_config
    from src.catnat.utils.reproducibility import set_seed, get_device

    default_configs = {"gsl": "configs/gsl.yaml",
                       "vae": "configs/vae.yaml",
                       "rl":  "configs/rl.yaml"}
    config = load_config(args.config or default_configs[args.experiment])
    if args.device:
        config.hardware.device = args.device

    set_seed(0)
    device = get_device(config.hardware.device)

    results = {}
    if args.experiment == "gsl":
        results = _eval_gsl(config, args, device)
    elif args.experiment == "vae":
        results = _eval_vae(config, args, device)
    elif args.experiment == "rl":
        results = _eval_rl(config, args, device)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if args.output_dir:
        out_path = Path(args.output_dir) / f"eval_{args.experiment}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


def _eval_gsl(config, args, device):
    from src.catnat.models.gsl import GSLModel
    from src.catnat.data.gsl_dataset import get_gsl_dataloaders, build_theta_star
    from src.catnat.evaluation.metrics import GSLMetrics

    model = GSLModel(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, _, test_loader = get_gsl_dataloaders(config, theta_star=args.theta_star)
    theta_star = build_theta_star(
        config.model.n_nodes, config.model.n_communities, args.theta_star
    )

    metrics = GSLMetrics.evaluate_all(
        model, test_loader, theta_star, device, M=config.training.M_samples
    )
    return metrics


def _eval_vae(config, args, device):
    from src.catnat.models.vae import CatVAE
    from src.catnat.data.mnist_dataset import get_vae_dataloaders
    from src.catnat.evaluation.metrics import VAEMetrics
    from src.catnat.evaluation.fim_analysis import FIMAnalyzer
    from src.catnat.catnat import CatNat

    model = CatVAE(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, _, test_loader = get_vae_dataloaders(config)

    results = {}
    print(f"Computing IWAE NLL with {args.n_eval_samples} importance samples...")
    results["nll_iwae"] = VAEMetrics.iwae_nll(
        model, test_loader, device, n_samples=args.n_eval_samples
    )
    results["elbo"] = VAEMetrics.elbo(model, test_loader, device)

    if args.compute_fim and isinstance(model.pi, CatNat):
        print("Verifying FIM diagonality (this may take a moment)...")
        fim_result = FIMAnalyzer.verify_diagonal(model.pi, config.catnat.K)
        results["fim_is_diagonal"]    = fim_result["is_diagonal"]
        results["fim_max_offdiag"]    = fim_result["max_offdiag_abs"]
        results["fim_diag_mean"]      = fim_result["diag_mean"]

    return results


def _eval_rl(config, args, device):
    from src.catnat.models.ppo_agent import PPOAgent
    from src.catnat.data.atari_wrappers import make_atari_env
    from src.catnat.evaluation.metrics import RLMetrics
    import math

    env = make_atari_env(config.env.name, seed=42)
    n_actions = env.action_space.n
    K_required = 2 ** math.ceil(math.log2(n_actions))
    if config.catnat.K < n_actions:
        config.catnat.K = K_required

    agent = PPOAgent(config, n_actions=n_actions)
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent"])
    agent.to(device).eval()

    result = RLMetrics.episodic_return(env, agent, device, n_episodes=10)
    env.close()
    return result


if __name__ == "__main__":
    main()
