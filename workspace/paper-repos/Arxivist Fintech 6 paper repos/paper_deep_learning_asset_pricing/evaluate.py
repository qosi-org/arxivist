"""
evaluate.py — Evaluation entry point.

Loads trained ensemble, computes SR / EV / XS-R2 on test set, and
reports variable importance.

Usage:
    python evaluate.py --config configs/config.yaml --split test
    python evaluate.py --config configs/config.yaml --split valid --ensemble-idx 0
"""

import argparse
import json
from pathlib import Path

import torch

from dlap.utils.config import load_config, set_seed, get_device
from dlap.models.gan_model import GANAssetPricingModel
from dlap.evaluation.metrics import compute_all_metrics, compute_variable_importance
from dlap.data.dataset import make_synthetic_dataset

CHAR_NAMES = [
    "ST_REV", "SUV", "r12_2", "NOA", "SGA2S", "LME", "RNA", "LTurnover",
    "Lev", "Resid_Var", "ROA", "E2P", "D2P", "Spread", "CF2P", "BEME",
    "Variance", "D2A", "PCM", "A2ME", "AT", "Rel2High", "CF", "Q",
    "Investment", "PM", "DPI2A", "ROE", "S2P", "FC2Y", "AC", "CTO",
    "LT_Rev", "OP", "PROF", "IdioVol", "r12_7", "Beta", "OA", "ATO",
    "MktBeta", "OL", "C", "r36_13", "NI", "r2_1",
]


def load_ensemble(cfg: dict, device: torch.device, num_models: int):
    """Load all ensemble model checkpoints."""
    models = []
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    for idx in range(num_models):
        ckpt_path = ckpt_dir / f"best_model_ensemble_{idx}.pt"
        if not ckpt_path.exists():
            print(f"[WARNING] Checkpoint not found: {ckpt_path}. Skipping.")
            continue
        model = GANAssetPricingModel(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        models.append(model)
    if not models:
        print("[WARNING] No checkpoints found. Using untrained model.")
        models = [GANAssetPricingModel(cfg).to(device)]
    return models


def ensemble_predict(models, macro_b, chars, returns):
    """Average SDF weights and loadings across ensemble members."""
    omegas, F_ts, betas = [], [], []
    with torch.no_grad():
        for model in models:
            omega, F_t, M_t, h_t = model.forward_sdf(macro_b, chars, returns)
            beta = model.forward_loadings(h_t, chars)
            omegas.append(omega)
            F_ts.append(F_t)
            betas.append(beta)

    omega_avg = torch.stack(omegas).mean(0)
    F_t_avg = torch.stack(F_ts).mean(0)
    beta_avg = torch.stack(betas).mean(0)
    return omega_avg, F_t_avg, beta_avg


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAN Asset Pricing Model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--ensemble-idx", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["hardware"]["device"])

    print(f"\nDeep Learning in Asset Pricing — Evaluation ({args.split})")

    # Load data
    print("[WARNING] Using synthetic data — replace with real CRSP data.")
    dataset = make_synthetic_dataset(T=600, N=500, device=device)
    macro, chars, returns, panel_weights = dataset.get_all()

    # Split selection
    splits = {"train": (0, 250), "valid": (250, 350), "test": (350, 600)}
    s, e = splits[args.split]
    macro_s = macro[s:e].unsqueeze(0)
    chars_s = chars[s:e]
    returns_s = returns[s:e]
    pw_s = panel_weights

    # Load ensemble
    num_models = 1 if args.ensemble_idx is not None else cfg["training"]["num_ensemble_models"]
    if args.ensemble_idx is not None:
        models = []
        ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        ckpt_path = ckpt_dir / f"best_model_ensemble_{args.ensemble_idx}.pt"
        model = GANAssetPricingModel(cfg).to(device)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
        model.eval()
        models = [model]
    else:
        models = load_ensemble(cfg, device, cfg["training"]["num_ensemble_models"])

    # Predict
    omega_avg, F_t_avg, beta_avg = ensemble_predict(models, macro_s, chars_s, returns_s)

    # Metrics
    metrics = compute_all_metrics(returns_s, beta_avg, F_t_avg, pw_s, annualize=True)

    print(f"\n{'='*50}")
    print(f"Results on {args.split} split:")
    print(f"  Sharpe Ratio (annual):  {metrics['sharpe_ratio']:.4f}")
    print(f"  Explained Variation:    {metrics['explained_variation']:.4f}")
    print(f"  Cross-Sectional R2:     {metrics['xs_r2']:.4f}")
    print(f"{'='*50}")

    # Variable importance
    if len(models) > 0:
        print("\nComputing variable importance...")
        vi = compute_variable_importance(
            models[0], macro_s.squeeze(0), chars_s, returns_s, CHAR_NAMES
        )
        top10 = sorted(vi.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most important characteristics:")
        for i, (name, score) in enumerate(top10, 1):
            print(f"  {i:2d}. {name:<20s} {score:.5f}")

    # Save results
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"metrics_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
