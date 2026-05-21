"""CIFAR-10 ResNet training entrypoint.

Examples:
  python train.py --config configs/resnet20.yaml
  python train.py --config configs/resnet110.yaml --output-dir runs/resnet110
  python train.py --config configs/config_debug.yaml --debug   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from resnet_cifar.data import CIFAR10DataModule
from resnet_cifar.models import build_model
from resnet_cifar.training import Trainer
from resnet_cifar.utils import load_config, merge_cli_overrides, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 ResNet (He et al. 2015).")
    parser.add_argument("--config", type=str, default="configs/resnet20.yaml",
                        help="Path to YAML config.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override config.model.name (e.g. resnet32).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override config.hardware.device (auto|cuda|cpu).")
    parser.add_argument("--total-iters", type=int, default=None,
                        help="Override training.total_iterations.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override hardware.seed.")
    parser.add_argument("--output-dir", type=str, default="./runs",
                        help="Where to write checkpoints and metrics.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--debug", action="store_true",
                        help="Use a tiny subset of training iterations for smoke testing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build everything but skip training (validates the setup).")
    parser.add_argument("--override", nargs="*", default=None,
                        help="dotted.key=value overrides applied after CLI flags.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.device:
        cfg["hardware"]["device"] = args.device
    if args.total_iters is not None:
        cfg["training"]["total_iterations"] = args.total_iters
    if args.seed is not None:
        cfg["hardware"]["seed"] = args.seed
    if args.debug:
        cfg["training"]["total_iterations"] = min(cfg["training"]["total_iterations"], 200)
        cfg["training"]["lr_drop_iterations"] = [100, 150]
        cfg["evaluation"]["eval_every_n_epochs"] = 100  # only at end

    seed = int(cfg["hardware"]["seed"])
    set_seed(seed, deterministic=bool(cfg["hardware"].get("deterministic", True)))

    device = resolve_device(str(cfg["hardware"].get("device", "auto")))

    data = CIFAR10DataModule(
        data_dir=cfg["data"]["dataset_root"],
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        val_size=int(cfg["data"].get("val_size", 0)),
        mean_subtraction=str(cfg["data"]["mean_subtraction"]),
        download=bool(cfg["data"].get("download", True)),
        seed=seed,
    )

    model = build_model(
        cfg["model"]["name"],
        num_classes=int(cfg["model"]["num_classes"]),
        shortcut_option=str(cfg["model"].get("shortcut_option", "A")),
    )
    print(repr(model))

    output_dir = Path(args.output_dir) / cfg["model"]["name"]
    trainer = Trainer(
        model=model,
        train_loader=data.train_loader(),
        test_loader=data.test_loader(),
        val_loader=data.val_loader(),
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )

    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at iter={ckpt.get('best_iter')}")

    if args.dry_run:
        print("Dry run complete — components built successfully. Skipping training.")
        return

    result = trainer.fit()
    print(f"Best test top-1: {result['best_test_top1']:.2f}%  (iter {result['best_iter']})")
    print(f"Final test error: {100 - result['best_test_top1']:.2f}%")

    summary_path = output_dir / "summary.json"
    summary = {
        "model": cfg["model"]["name"],
        "best_test_top1": result["best_test_top1"],
        "best_test_error": 100 - result["best_test_top1"],
        "best_iter": result["best_iter"],
        "total_iterations": int(cfg["training"]["total_iterations"]),
        "device": str(device),
        "seed": seed,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
