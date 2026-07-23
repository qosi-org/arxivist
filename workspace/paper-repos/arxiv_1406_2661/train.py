import torch
import yaml
from src.gan.models.generator import Generator
from src.gan.models.discriminator import Discriminator
from src.gan.training.trainer import GANTrainer
from src.gan.data.dataset import get_mnist_loader

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["hardware"]["seed"])
    device = cfg["hardware"]["device"]

    G = Generator(z_dim=cfg["model"]["z_dim"], hidden_units=cfg["model"]["g_hidden_units"],
                  output_dim=cfg["model"]["input_dim"])
    D = Discriminator(input_dim=cfg["model"]["input_dim"], hidden_units=cfg["model"]["d_hidden_units"],
                       dropout_p=cfg["training"]["dropout_p"])

    trainer = GANTrainer(G, D, z_dim=cfg["model"]["z_dim"], lr=cfg["training"]["learning_rate"],
                          k=cfg["training"]["k_discriminator_steps"], device=device)

    loader = get_mnist_loader(batch_size=cfg["training"]["batch_size"])

    for epoch in range(cfg["training"]["epochs"]):
        for real_batch, _ in loader:
            losses = trainer.train_step(real_batch)
        print(f"Epoch {epoch}: loss_d={losses['loss_d']:.4f}, loss_g={losses['loss_g']:.4f}")

    torch.save(G.state_dict(), "checkpoints/generator.pt")
    torch.save(D.state_dict(), "checkpoints/discriminator.pt")

if __name__ == "__main__":
    main()