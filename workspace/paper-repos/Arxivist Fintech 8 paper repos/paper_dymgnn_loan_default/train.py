"""
train.py — Main Training Entrypoint for DYMGNN.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug
    python train.py --config configs/config.yaml --resume checkpoints/best.pt
    python train.py --config configs/config.yaml --dry-run

Paper: "Attention-based dynamic multilayer graph neural networks for loan default prediction"
Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dymgnn.utils.config import load_config, set_seeds, get_device
from dymgnn.data.dataset import FreddieDataset
from dymgnn.models.dymgnn import DYMGNN
from dymgnn.training.trainer import DYMGNNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DYMGNN for loan default prediction (Zandi et al. EJOR 2025)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed from config.")
    parser.add_argument("--debug", action="store_true",
                        help="Run 5-epoch debug loop. Validates full pipeline quickly.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build all components and print summary without training.")
    parser.add_argument("--gnn-type", type=str, default=None,
                        help="Override model.gnn_type (GCN or GAT).")
    parser.add_argument("--rnn-type", type=str, default=None,
                        help="Override model.rnn_type (LSTM or GRU).")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable temporal attention (GNN-RNN instead of GNN-RNN-ATT).")
    parser.add_argument("--network-type", type=str, default=None,
                        help="Override model.network_type (area, company, double).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.gnn_type is not None:
        cfg["model"]["gnn_type"] = args.gnn_type
    if args.rnn_type is not None:
        cfg["model"]["rnn_type"] = args.rnn_type
    if args.no_attention:
        cfg["model"]["use_attention"] = False
    if args.network_type is not None:
        cfg["model"]["network_type"] = args.network_type

    set_seeds(cfg["seed"])
    device = get_device(cfg)

    # Load datasets (falls back to synthetic if Freddie Mac data not available)
    data_cfg = cfg["data"]
    train_dataset = FreddieDataset(
        data_dir=data_cfg["data_dir"],
        network_type=cfg["model"]["network_type"],
        split="train", cfg=cfg,
    )
    train_dataset.load()

    test_dataset = FreddieDataset(
        data_dir=data_cfg["data_dir"],
        network_type=cfg["model"]["network_type"],
        split="test", cfg=cfg,
    )
    test_dataset.load()

    # Infer num_nodes from first window's first snapshot
    sample_window = train_dataset[0]
    num_nodes_base = len(sample_window)
    if cfg["model"]["network_type"] == "double":
        num_nodes = 2 * num_nodes_base
    else:
        num_nodes = num_nodes_base

    # Build model
    model_cfg = cfg["model"]
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
    )

    trainer = DYMGNNTrainer(model=model, cfg=cfg, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.dry_run:
        print("\n[Dry Run] All components initialised successfully.")
        print(f"  Model: {model}")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Train dataset: {train_dataset}")
        print(f"  Test dataset:  {test_dataset}")
        print(f"  Device: {device}")
        print("\n[Dry Run] No training performed.")
        return

    trainer.train(train_dataset=train_dataset, val_dataset=test_dataset, debug=args.debug)


if __name__ == "__main__":
    main()
