import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator MLP: maps noise z -> data space x.
    ASSUMED: 2 hidden layers, 240 units each (confidence: 0.45)
    Paper (Section 5) only states 'multilayer perceptron' with ReLU + sigmoid activations,
    no layer sizes given. Values here follow the original authors' GitHub repo convention.
    """
    def __init__(self, z_dim=100, hidden_units=240, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
            nn.Sigmoid()  # output in [0,1] for MNIST pixel range
        )

    def forward(self, z):
        return self.net(z)