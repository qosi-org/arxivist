"""U-Net — faithful to Ronneberger, Fischer & Brox (2015). Unpadded (valid) convs."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3),  # valid (unpadded)
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def center_crop(enc, target):
    _, _, h, w = target.shape
    _, _, H, W = enc.shape
    dh, dw = (H - h) // 2, (W - w) // 2
    return enc[:, :, dh:dh + h, dw:dw + w]


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.d1 = DoubleConv(in_channels, c[0])
        self.d2 = DoubleConv(c[0], c[1])
        self.d3 = DoubleConv(c[1], c[2])
        self.d4 = DoubleConv(c[2], c[3])
        self.bottleneck = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.u4 = DoubleConv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.u3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.u2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u1 = DoubleConv(c[1], c[0])
        self.out = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        s4 = self.d4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        x = self.up4(b); x = self.u4(torch.cat([center_crop(s4, x), x], 1))
        x = self.up3(x); x = self.u3(torch.cat([center_crop(s3, x), x], 1))
        x = self.up2(x); x = self.u2(torch.cat([center_crop(s2, x), x], 1))
        x = self.up1(x); x = self.u1(torch.cat([center_crop(s1, x), x], 1))
        return self.out(x)
