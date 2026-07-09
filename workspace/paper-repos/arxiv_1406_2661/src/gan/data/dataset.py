import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loader(batch_size=100, train=True, data_dir="./data"):
    """
    MNIST loader — paper's primary dataset (Table 1 results).
    ASSUMED batch_size=100 (confidence 0.4) — not specified in paper.
    Flattens 28x28 images to 784-dim vectors to match MLP input.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten to [784]
    ])
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_cifar10_loader(batch_size=100, train=True, data_dir="./data"):
    """
    CIFAR-10 loader. Paper only reports qualitative samples for CIFAR-10,
    no numeric metric — included for completeness (confidence 0.5).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)