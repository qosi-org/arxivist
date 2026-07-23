"""Smoke-test training on ImageNet subset."""
import torch, torch.nn as nn
from src.vgg import vgg16

def main(epochs=2):
    model = vgg16()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 1000, (4,))
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        print(f"Epoch {epoch}: loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
