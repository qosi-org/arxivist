"""Evaluate a trained CIFAR-10 ResNet checkpoint on the test set.

Example:
  python evaluate.py --config configs/resnet20.yaml --checkpoint runs/resnet20/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from resnet_cifar.data import CIFAR10DataModule
from resnet_cifar.evaluation.metrics import AccuracyMeter
from resnet_cifar.models import build_model
from resnet_cifar.training.losses import cross_entropy_loss
from resnet_cifar.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 ResNet checkpoint.")
    parser.add_argument("--config", type=str, default="configs/resnet20.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["hardware"]["device"] = args.device

    set_seed(int(cfg["hardware"]["seed"]), deterministic=True)
    device = resolve_device(str(cfg["hardware"].get("device", "auto")))

    data = CIFAR10DataModule(
        data_dir=cfg["data"]["dataset_root"],
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        val_size=0,
        mean_subtraction=str(cfg["data"]["mean_subtraction"]),
        download=False,
        seed=int(cfg["hardware"]["seed"]),
    )

    model = build_model(
        cfg["model"]["name"],
        num_classes=int(cfg["model"]["num_classes"]),
        shortcut_option=str(cfg["model"].get("shortcut_option", "A")),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(repr(model))

    loader = data.test_loader()
    meter = AccuracyMeter()
    loss_sum = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = cross_entropy_loss(logits, labels)
        bs = labels.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        meter.update(logits, labels)

    top1, err = meter.compute()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test loss:   {loss_sum / max(n, 1):.4f}")
    print(f"Top-1 acc:   {top1:.2f}%")
    print(f"Test error:  {err:.2f}%")


if __name__ == "__main__":
    main()
