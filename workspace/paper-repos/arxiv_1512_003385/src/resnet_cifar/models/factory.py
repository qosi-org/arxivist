"""Name -> ResNetCIFAR factory."""
from __future__ import annotations

from resnet_cifar.models.resnet import ResNetCIFAR

NAME_TO_N: dict[str, int] = {
    "resnet20": 3,
    "resnet32": 5,
    "resnet44": 7,
    "resnet56": 9,
    "resnet110": 18,
}


def build_model(name: str, num_classes: int = 10, shortcut_option: str = "A") -> ResNetCIFAR:
    """Build a CIFAR-10 ResNet by name (resnet20 / resnet32 / resnet44 / resnet56 / resnet110)."""
    name = name.lower()
    if name not in NAME_TO_N:
        raise ValueError(
            f"Unknown model name {name!r}. Valid names: {sorted(NAME_TO_N)}"
        )
    return ResNetCIFAR(n=NAME_TO_N[name], num_classes=num_classes, shortcut_option=shortcut_option)
