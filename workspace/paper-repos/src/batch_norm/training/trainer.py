"""
Trainer for Batch-Normalized MLP.
Paper: Section 4.1 — SGD with momentum, 50000 steps, batch size 60.
arXiv:1502.03167
"""
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    """
    SGD trainer for the BN-MLP MNIST experiment.
    Learning rate and momentum: ASSUMED standard values (paper states lr=0.0015 for Inception).
    """
    def __init__(self, model, lr: float = 0.01, momentum: float = 0.9, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # ASSUMED: SGD with momentum (confidence: 0.85)
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, loader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
