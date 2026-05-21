"""Single-image inference for a CIFAR-10 ResNet.

Example:
  python inference.py --checkpoint runs/resnet20/best.pt --image dog.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent / "src"))

from resnet_cifar.data import CIFAR10DataModule
from resnet_cifar.models import build_model
from resnet_cifar.utils import load_config, set_seed

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/resnet20.yaml")
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
        batch_size=1,
        num_workers=0,
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

    img = Image.open(args.image).convert("RGB").resize((32, 32))
    tensor = transforms.ToTensor()(img)
    # Apply the same mean subtraction the model was trained with.
    if cfg["data"]["mean_subtraction"] == "per_pixel":
        tensor = tensor - data.mean
    else:
        tensor = tensor - data.mean.view(3, 1, 1)
    tensor = tensor.unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    top_idx = int(probs.argmax().item())
    print(f"Top-1: {CIFAR10_CLASSES[top_idx]} (p={probs[top_idx].item():.3f})")
    print("All classes (descending):")
    for i in torch.argsort(probs, descending=True).tolist():
        print(f"  {CIFAR10_CLASSES[i]:>10}  {probs[i].item():.3f}")


if __name__ == "__main__":
    main()
