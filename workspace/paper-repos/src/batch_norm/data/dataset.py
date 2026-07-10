"""
MNIST data loader for Batch Normalization experiment.
Paper: Section 4.1 — MNIST, batch size 60, 50000 steps.
arXiv:1502.03167
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size: int = 60, data_dir: str = "./data"):
    """
    Returns MNIST train and test loaders.
    Batch size 60 is explicit in Section 4.1.
    Images flattened to 784-dim vectors for MLP input.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
    ])
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
