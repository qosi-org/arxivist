import torch
import torch.nn as nn

class Maxout(nn.Module):
    """Maxout activation: splits linear output into `pieces` groups, takes max."""
    def __init__(self, in_features, out_features, pieces=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * pieces)
        self.out_features = out_features
        self.pieces = pieces

    def forward(self, x):
        x = self.linear(x)
        x = x.view(*x.shape[:-1], self.out_features, self.pieces)
        return x.max(dim=-1)[0]

class Discriminator(nn.Module):
    """
    Discriminator MLP: maps data x -> scalar probability.
    ASSUMED: 2 hidden layers, 1200 units each, maxout activation (confidence: 0.45)
    Paper (Section 5) states maxout units are used but gives no layer sizes.
    Dropout applied during training per paper text (confidence: 0.85).
    """
    def __init__(self, input_dim=784, hidden_units=1200, dropout_p=0.5):
        super().__init__()
        self.layer1 = Maxout(input_dim, hidden_units)
        self.layer2 = Maxout(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.dropout(self.layer1(x))
        x = self.dropout(self.layer2(x))
        return torch.sigmoid(self.out(x))