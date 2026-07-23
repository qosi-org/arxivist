"""Early and Late Dropout for underfitting reduction (Liu et al., 2303.01500)."""
import torch
import torch.nn as nn

class EarlyDropout(nn.Module):
    """Dropout active only during early training epochs."""
    def __init__(self, p: float = 0.1, end_epoch: int = 50, total_epochs: int = 300):
        super().__init__()
        self.p = p
        self.end_epoch = end_epoch
        self.current_epoch = 0
    
    def forward(self, x):
        if self.training and self.current_epoch < self.end_epoch:
            return nn.functional.dropout(x, p=self.p, training=True)
        return x
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

class LateDropout(nn.Module):
    """Dropout active only in late training epochs."""
    def __init__(self, p: float = 0.1, start_epoch: int = 50):
        super().__init__()
        self.p = p
        self.start_epoch = start_epoch
        self.current_epoch = 0
    
    def forward(self, x):
        if self.training and self.current_epoch >= self.start_epoch:
            return nn.functional.dropout(x, p=self.p, training=True)
        return x
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
