import torch

def apply_max_norm(model, max_norm=15.0):
    """Apply per-unit L2 norm constraint."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            norm = torch.norm(param.data, p=2)
            if norm > max_norm:
                param.data = param.data * (max_norm / norm)
