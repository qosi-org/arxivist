"""Smoke-test training loop for U-Net (SGD, momentum 0.99, batch size 1)."""
import torch, torch.nn as nn
from src.unet import UNet


def main(steps=5):
    torch.manual_seed(0)
    model = UNet(in_channels=1, num_classes=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
    lossf = nn.CrossEntropyLoss()
    x = torch.randn(1, 1, 572, 572)
    for i in range(steps):
        opt.zero_grad()
        out = model(x)                      # -> (1,2,388,388)
        target = torch.randint(0, 2, out.shape[2:]).unsqueeze(0)
        loss = lossf(out, target)
        loss.backward()
        opt.step()
        print(f"step {i}  loss {loss.item():.4f}  out {tuple(out.shape)}")


if __name__ == "__main__":
    main()
