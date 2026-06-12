"""
train.py
--------
Main entrypoint for gMLP / aMLP NLP pretraining (Masked Language Modelling).

Usage:
    python train.py --config configs/gmlp_base_mlm.yaml
    python train.py --config configs/gmlp_base_mlm.yaml --resume outputs/checkpoint_step_50000.pt
    python train.py --preset gmlp-base-ablation --debug
    python train.py --config configs/gmlp_base_mlm.yaml --dry-run

Paper: "Pay Attention to MLPs" (arXiv:2105.08050)
See Appendix A.2 for full pretraining hyperparameters.
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

# Ensure package is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gmlp.utils.config import gMLPConfig, set_seed, get_preset
from gmlp.models.gmlp import gMLP
from gmlp.data.mlm_dataset import MLMDataset
from gmlp.training.trainer_nlp import NLPTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gMLP NLP pretraining (Masked Language Modelling)"
    )
    # Config
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to YAML config file")
    group.add_argument("--preset", type=str,
                       help="Named preset (e.g. gmlp-base-ablation, gmlp-base-mlm)")

    # Overrides
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override number of training steps")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override peak learning rate")
    parser.add_argument("--precision", type=str, default=None,
                        choices=["float32", "bf16", "fp16"],
                        help="Override training precision")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")

    # Control flags
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: tiny dataset, 100 steps, verbose logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build all components and validate setup, but do not train")
    parser.add_argument("--num_workers", type=int, default=None)

    return parser.parse_args()


def build_tokenizer(config: gMLPConfig):
    """
    Load tokenizer. Paper uses 32K cased SentencePiece vocabulary.
    Proxy: google/t5-base tokenizer (closest publicly available match).
    See SIR implementation_assumptions[assume_007] and risk_assessment[R7].
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers")

    # SentencePiece 32K tokenizer proxy (SIR assume_007)
    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)
    # Ensure mask token exists (T5 uses <extra_id_0> as mask)
    if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    return tokenizer


def main() -> None:
    args = parse_args()

    # ── Logging ──────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train")

    # ── Config ───────────────────────────────────────────────────
    if args.config:
        config = gMLPConfig.from_yaml(args.config)
    else:
        config = get_preset(args.preset)

    # Apply CLI overrides
    if args.output_dir:   config.output_dir = args.output_dir
    if args.num_steps:    config.training.num_steps = args.num_steps
    if args.batch_size:   config.training.batch_size = args.batch_size
    if args.lr:           config.training.lr = args.lr
    if args.precision:    config.training.precision = args.precision
    if args.seed:         config.training.seed = args.seed
    if args.num_workers:  config.training.num_workers = args.num_workers

    if args.debug:
        config.training.num_steps = 100
        config.training.log_interval = 10
        config.training.eval_interval = 50
        config.training.save_interval = 100
        config.data.use_streaming = False
        logger.info("[DEBUG] Reduced to 100 steps, streaming disabled.")

    # ── Seed ─────────────────────────────────────────────────────
    set_seed(config.training.seed, deterministic=config.training.deterministic)
    logger.info(f"Seed: {config.training.seed}")

    # ── Tokenizer ────────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = build_tokenizer(config)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ── Datasets ─────────────────────────────────────────────────
    logger.info("Building datasets...")
    train_dataset = MLMDataset(
        tokenizer=tokenizer,
        max_seq_len=config.data.max_seq_len,
        mlm_probability=config.data.mlm_probability,
        dataset_name=config.data.dataset,
        dataset_config="realnewslike" if "ablation" in config.experiment_name else "en",
        split="train",
        use_streaming=config.data.use_streaming,
        data_dir=config.data.data_dir,
        seed=config.training.seed,
    )
    val_dataset = MLMDataset(
        tokenizer=tokenizer,
        max_seq_len=config.data.max_seq_len,
        mlm_probability=config.data.mlm_probability,
        dataset_name=config.data.dataset,
        dataset_config="realnewslike" if "ablation" in config.experiment_name else "en",
        split="validation",
        use_streaming=False,  # val is small; materialise it
        data_dir=config.data.data_dir,
        seed=config.training.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────
    logger.info("Building model...")
    model = gMLP(config.model)
    model.set_task("mlm")
    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {model.get_num_params():,}")

    # ── Dry run ──────────────────────────────────────────────────
    if args.dry_run:
        logger.info("[DRY RUN] All components built successfully. Exiting.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids=input_ids, labels=labels)
        logger.info(f"[DRY RUN] Forward pass OK. Loss: {output.loss.item():.4f}")
        return

    # ── Trainer ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = NLPTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=config.output_dir,
        device=device,
    )

    # Save config alongside checkpoints
    config.to_yaml(os.path.join(config.output_dir, "config.yaml"))
    logger.info(f"Config saved to {config.output_dir}/config.yaml")

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
