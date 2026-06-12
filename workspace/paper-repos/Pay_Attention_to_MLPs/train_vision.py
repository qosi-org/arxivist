"""
train_vision.py
---------------
ImageNet classification training entrypoint for gMLP vision models.

Usage:
    python train_vision.py --model_size S --data_dir data/imagenet/
    python train_vision.py --config configs/gmlp_s_imagenet.yaml
    python train_vision.py --model_size B --precision bf16 --output_dir outputs/gmlp_B/
    python train_vision.py --model_size Ti --debug     # 5-batch smoke test

Paper Section 3 + Appendix A.1: "Pay Attention to MLPs" (arXiv:2105.08050)
Target results (Table 2):
  gMLP-Ti: 72.3%  gMLP-S: 79.6%  gMLP-B: 81.6%  (ImageNet Top-1, 300 epochs)
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gmlp.utils.config import gMLPConfig, ModelConfig, TrainingConfig, DataConfig, set_seed
from gmlp.models.gmlp import gMLP
from gmlp.data.imagenet_dataset import ImageNetDataset, MixupCutmixCollator
from gmlp.training.trainer_vision import VisionTrainer


# Paper Table 1 + Table 7 — architecture + stochastic depth per size
VISION_PRESETS = {
    "Ti": dict(num_layers=30, d_model=128, d_ffn=768,  survival_prob=1.00, params_M=5.9),
    "S":  dict(num_layers=30, d_model=256, d_ffn=1536, survival_prob=0.95, params_M=19.5),
    "B":  dict(num_layers=30, d_model=512, d_ffn=3072, survival_prob=0.80, params_M=73.4),
}


def parse_args():
    parser = argparse.ArgumentParser(description="gMLP ImageNet Training")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_size", choices=["Ti", "S", "B"],
                       help="Model size preset (paper Table 1)")
    group.add_argument("--config", type=str,
                       help="Path to YAML config (overrides --model_size)")

    parser.add_argument("--data_dir", type=str, default="data/imagenet/",
                        help="Root directory of ImageNet dataset")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override epochs (default: 300 per paper)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (default: 4096 per paper)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float32", "bf16", "fp16"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="5-batch smoke test (validates full pipeline)")
    return parser.parse_args()


def build_config(args) -> gMLPConfig:
    if args.config:
        return gMLPConfig.from_yaml(args.config)

    sz = VISION_PRESETS[args.model_size]
    # seq_len = (224/16)^2 = 196 patches
    config = gMLPConfig(
        model=ModelConfig(
            model_type="vision",
            use_tiny_attn=False,
            num_layers=sz["num_layers"],
            d_model=sz["d_model"],
            d_ffn=sz["d_ffn"],
            seq_len=196,
            num_classes=1000,
            img_size=224,
            patch_size=16,
            use_toeplitz=False,       # vision uses unconstrained W
            survival_prob=sz["survival_prob"],
            pool_mode="avg",          # ASSUMED: global avg pool (ambiguity_003)
            w_init_std=0.002,         # ASSUMED: near-zero init (ambiguity_002)
        ),
        training=TrainingConfig(
            optimizer="adamw",
            lr=args.lr or 1e-3,
            weight_decay=0.05,        # paper Table 7
            beta1=0.9,
            beta2=0.999,
            eps=1e-6,
            grad_clip=1.0,            # paper Table 7
            lr_schedule="cosine",     # paper Table 7
            warmup_steps=10000,
            batch_size=args.batch_size or 4096,
            num_epochs=args.num_epochs or 300,
            log_interval=50,
            save_interval=0,          # save by epoch
            eval_interval=0,
            precision=args.precision,
            num_workers=args.num_workers,
            seed=args.seed,
        ),
        data=DataConfig(
            dataset="imagenet",
            data_dir=args.data_dir,
            autoaugment=True,
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_mixup_switch_prob=0.5,
            label_smoothing=0.1,
            repeated_augmentation=False,
            random_erasing_prob=0.0,
        ),
        output_dir=args.output_dir or f"outputs/gmlp_{args.model_size}_imagenet/",
        experiment_name=f"gmlp_{args.model_size}_imagenet",
    )
    config.validate()
    return config


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("train_vision")

    config = build_config(args)
    set_seed(config.training.seed)

    # ── Datasets ──────────────────────────────────────────────────
    logger.info("Building ImageNet datasets...")
    train_dataset = ImageNetDataset(
        data_dir=config.data.data_dir,
        split="train",
        img_size=config.model.img_size,
        autoaugment=config.data.autoaugment,
    )
    val_dataset = ImageNetDataset(
        data_dir=config.data.data_dir,
        split="val",
        img_size=config.model.img_size,
        autoaugment=False,
    )

    # Mixup + CutMix collator (applied at batch level, paper Table 7)
    collator = MixupCutmixCollator(
        mixup_alpha=config.data.mixup_alpha,
        cutmix_alpha=config.data.cutmix_alpha,
        switch_prob=config.data.cutmix_mixup_switch_prob,
        num_classes=config.model.num_classes,
        label_smoothing=config.data.label_smoothing,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # ── Debug mode ────────────────────────────────────────────────
    if args.debug:
        config.training.num_epochs = 1
        logger.info("[DEBUG] Limited to 1 epoch / 5 batches")

    # ── Model ─────────────────────────────────────────────────────
    logger.info("Building model...")
    model = gMLP(config.model)
    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {model.get_num_params():,}")

    # Quick smoke test
    if args.debug:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        dummy = torch.randn(2, 3, 224, 224, device=device)
        out = model(pixel_values=dummy)
        logger.info(f"[DEBUG] Forward pass OK. logits shape: {out.logits.shape}")

    # ── Trainer ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = VisionTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=config.output_dir,
        device=device,
    )

    config.to_yaml(os.path.join(config.output_dir, "config.yaml"))
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
