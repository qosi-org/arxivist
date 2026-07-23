"""Smoke-test training loop for early/late dropout on a tiny ViT."""
import torch, torch.nn as nn
from src.dropout import EarlyDropout

class TinyViT(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, num_layers=12, num_classes=1000):
        super().__init__()
        self.embed = nn.Linear(3*224*224, embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.early_dropout = EarlyDropout(p=0.1, end_epoch=2)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embed(x).unsqueeze(1)  # (B, 1, embed_dim)
        x = self.early_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.head(x)

def main(epochs=5):
    model = TinyViT()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 1000, (4,))
    
    for epoch in range(epochs):
        model.early_dropout.set_epoch(epoch)
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        print(f"Epoch {epoch}: loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
