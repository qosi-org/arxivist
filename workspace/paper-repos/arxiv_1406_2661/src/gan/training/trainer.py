import torch
import torch.nn as nn
import torch.optim as optim

class GANTrainer:
    """
    Implements Algorithm 1 from the paper: alternating SGD updates for D and G.
    k=1 discriminator step per generator step (explicit in paper, confidence 0.99).
    Learning rate and batch size are ASSUMED (confidence 0.4) — not specified in paper text.
    """
    def __init__(self, generator, discriminator, z_dim=100, lr=0.001, k=1, device="cpu"):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.z_dim = z_dim
        self.k = k
        self.device = device
        # ASSUMED: SGD with momentum, momentum value not specified in paper (confidence 0.4)
        self.opt_D = optim.SGD(self.D.parameters(), lr=lr, momentum=0.5)
        self.opt_G = optim.SGD(self.G.parameters(), lr=lr, momentum=0.5)
        self.bce = nn.BCELoss()

    def train_step(self, real_batch):
        batch_size = real_batch.size(0)
        real_batch = real_batch.to(self.device)

        # --- k steps of D update ---
        for _ in range(self.k):
            z = torch.randn(batch_size, self.z_dim, device=self.device)
            fake_batch = self.G(z).detach()

            d_real = self.D(real_batch)
            d_fake = self.D(fake_batch)

            loss_d = self.bce(d_real, torch.ones_like(d_real)) + \
                     self.bce(d_fake, torch.zeros_like(d_fake))

            self.opt_D.zero_grad()
            loss_d.backward()
            self.opt_D.step()

        # --- 1 step of G update (non-saturating loss, Section 3) ---
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        fake_batch = self.G(z)
        d_fake = self.D(fake_batch)
        loss_g = self.bce(d_fake, torch.ones_like(d_fake))  # maximize log D(G(z))

        self.opt_G.zero_grad()
        loss_g.backward()
        self.opt_G.step()

        return {"loss_d": loss_d.item(), "loss_g": loss_g.item()}