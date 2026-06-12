"""
finetune.py
-----------
NLP finetuning entrypoint: SST-2, MNLI, SQuAD v1.1/v2.0.

Usage:
    python finetune.py --pretrained_checkpoint outputs/checkpoint_best.pt --task sst2
    python finetune.py --pretrained_checkpoint outputs/checkpoint_best.pt --task mnli \\
        --num_runs 5 --lr 2e-5 --batch_size 32
    python finetune.py --pretrained_checkpoint outputs/checkpoint_best.pt --task squad_v2

Paper protocol: median of 5 independent runs (Table 6).
Paper ref: Section 4.4, Tables 6, 9
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gmlp.utils.config import gMLPConfig, set_seed
from gmlp.models.gmlp import gMLP
from gmlp.training.finetuner import NLPFinetuner
from gmlp.evaluation.metrics import aggregate_runs


def parse_args():
    parser = argparse.ArgumentParser(description="gMLP NLP finetuning")
    parser.add_argument("--pretrained_checkpoint", required=True,
                        help="Path to MLM pretraining checkpoint (.pt file)")
    parser.add_argument("--task", required=True,
                        choices=["sst2", "mnli", "squad_v1", "squad_v2"])
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config (overrides checkpoint config)")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune/")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of independent runs (paper: 5)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_datasets(task: str, tokenizer, config, data_dir: str):
    from torch.utils.data import DataLoader
    from gmlp.data.glue_dataset import SST2Dataset, MNLIDataset, SQuADDataset

    if task == "sst2":
        train_ds = SST2Dataset(tokenizer, split="train", max_seq_len=128, data_dir=data_dir)
        val_ds = SST2Dataset(tokenizer, split="validation", max_seq_len=128, data_dir=data_dir)
    elif task == "mnli":
        train_ds = MNLIDataset(tokenizer, split="train", max_seq_len=128, data_dir=data_dir)
        val_ds = MNLIDataset(tokenizer, split="validation_matched", max_seq_len=128, data_dir=data_dir)
    elif task in ("squad_v1", "squad_v2"):
        version = "1.1" if task == "squad_v1" else "2.0"
        train_ds = SQuADDataset(tokenizer, version=version, split="train", max_seq_len=512, data_dir=data_dir)
        val_ds = SQuADDataset(tokenizer, version=version, split="validation", max_seq_len=512, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown task: {task}")

    bs = config.training.batch_size
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)
    return train_loader, val_loader


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("finetune")

    # Load pretrained checkpoint
    logger.info(f"Loading checkpoint: {args.pretrained_checkpoint}")
    ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")

    # Build config
    if args.config:
        config = gMLPConfig.from_yaml(args.config)
    else:
        config = ckpt.get("config", gMLPConfig())

    # Override finetuning hyperparameters from paper Table 9
    finetune_defaults = {
        "sst2":     dict(lr=2e-5, batch_size=32, num_epochs=5, lr_schedule="linear", warmup_steps=500),
        "mnli":     dict(lr=2e-5, batch_size=32, num_epochs=5, lr_schedule="linear", warmup_steps=500),
        "squad_v1": dict(lr=5e-5, batch_size=32, num_steps=8000, lr_schedule="linear", warmup_steps=1000),
        "squad_v2": dict(lr=5e-5, batch_size=32, num_steps=8000, lr_schedule="linear", warmup_steps=1000),
    }
    for k, v in finetune_defaults[args.task].items():
        setattr(config.training, k, v)
    if args.lr:
        config.training.lr = args.lr
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-run loop — paper: median of 5 runs
    run_metrics = []
    for run_idx in range(args.num_runs):
        seed = args.seed + run_idx
        set_seed(seed)
        logger.info(f"Run {run_idx + 1}/{args.num_runs}  seed={seed}")

        # Build fresh model and load pretrained weights
        model = gMLP(config.model)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Load datasets
        train_loader, val_loader = load_datasets(
            args.task, tokenizer, config, args.data_dir
        )

        run_output_dir = os.path.join(args.output_dir, args.task, f"run_{run_idx+1}")
        finetuner = NLPFinetuner(
            model=model,
            config=config,
            task=args.task,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=run_output_dir,
            device=device,
        )
        best = finetuner.train()
        run_metrics.append(best)
        logger.info(f"Run {run_idx+1} best: {best:.4f}")

    # Aggregate results
    stats = aggregate_runs(run_metrics)
    logger.info(f"\n{'='*50}")
    logger.info(f"Task: {args.task.upper()}  ({args.num_runs} runs)")
    logger.info(f"  Median: {stats['median']:.4f}")
    logger.info(f"  Mean:   {stats['mean']:.4f} ± {stats['std']:.4f}")
    logger.info(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
    logger.info(f"{'='*50}")

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.task}_results.txt"), "w") as f:
        f.write(f"Task: {args.task}\n")
        f.write(f"Runs: {args.num_runs}\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
