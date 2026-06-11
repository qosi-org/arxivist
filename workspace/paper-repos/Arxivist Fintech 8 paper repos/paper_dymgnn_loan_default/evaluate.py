"""
evaluate.py — Evaluation Entrypoint for DYMGNN.

Evaluates a trained model on the test set, reporting AUC and F1 with
95% bootstrapped confidence intervals (Section 4.3).

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --all-configs

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

from dymgnn.utils.config import load_config, set_seeds, get_device
from dymgnn.data.dataset import FreddieDataset
from dymgnn.models.dymgnn import DYMGNN
from dymgnn.evaluation.metrics import evaluate_full, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DYMGNN model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def collect_predictions(model, dataset, device, cfg):
    """Run model on all test windows, return concatenated predictions and labels."""
    model.eval()
    model_cfg = cfg["model"]
    all_preds, all_labels = [], []

    with torch.no_grad():
        for window in dataset:
            feats = [f.to(device) for f in window.snapshot_feats]
            adjs  = [a.to(device) for a in window.snapshot_adjs]
            y_hat, _ = model(feats, adjs, node_mask=None)
            all_preds.append(y_hat.squeeze(-1).cpu().numpy())
            all_labels.append(window.labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed:
        cfg["seed"] = args.seed
    set_seeds(cfg["seed"])
    device = get_device(cfg)

    # Load test data
    test_dataset = FreddieDataset(
        data_dir=cfg["data"]["data_dir"],
        network_type=cfg["model"]["network_type"],
        split="test", cfg=cfg,
    )
    test_dataset.load()

    # Build and load model
    model_cfg = cfg["model"]
    sample = test_dataset[0]
    num_nodes = len(sample) * (2 if model_cfg["network_type"] == "double" else 1)

    model = DYMGNN(
        num_features=model_cfg["num_features"],
        embedding_dim=model_cfg["embedding_dim"],
        num_snapshots=model_cfg["num_snapshots"],
        num_nodes=num_nodes,
        gnn_type=model_cfg["gnn_type"],
        rnn_type=model_cfg["rnn_type"],
        use_attention=model_cfg["use_attention"],
        num_gat_heads=model_cfg["num_gat_heads"],
        decoder_hidden1=model_cfg["decoder_hidden_1"],
        decoder_hidden2=model_cfg["decoder_hidden_2"],
        decoder_dropout=model_cfg["decoder_dropout"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    print(f"\nEvaluating {model} on {len(test_dataset)} test window(s)...")
    y_score, y_true = collect_predictions(model, test_dataset, device, cfg)
    results = evaluate_full(y_true, y_score, do_bootstrap=True)

    model_name = f"{model_cfg['gnn_type']}-{model_cfg['rnn_type']}"
    if model_cfg["use_attention"]:
        model_name += "-ATT"
    model_name += f" ({model_cfg['network_type']})"

    print_results_table({model_name: results})

    # Compare to paper's best result
    print("\nPaper best (GAT-LSTM-ATT double layer):")
    print("  AUC = 0.812 ± 0.008  |  F1 = 0.851 ± 0.007  (Table 7)")

    auc_diff = results["auc"] - 0.812
    print(f"\nYour model vs paper best: ΔAUC = {auc_diff:+.3f}")


if __name__ == "__main__":
    main()
