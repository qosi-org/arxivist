"""
Training entrypoint for Batch Normalization MNIST experiment.
Paper: Section 4.1 — arXiv:1502.03167
"""
import torch
import yaml
from src.batch_norm.models.mlp import BatchNormMLP
from src.batch_norm.training.trainer import Trainer
from src.batch_norm.data.dataset import get_mnist_loaders


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["hardware"]["seed"])
    device = cfg["hardware"]["device"]

    model = BatchNormMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_units=cfg["model"]["hidden_units"],
        num_classes=cfg["model"]["num_classes"],
        epsilon=cfg["model"]["epsilon"],
        momentum=cfg["model"]["bn_momentum"]
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    trainer = Trainer(model, lr=cfg["training"]["learning_rate"],
                      momentum=cfg["training"]["momentum"], device=device)
    train_loader, test_loader = get_mnist_loaders(
        batch_size=cfg["training"]["batch_size"]
    )

    step = 0
    for epoch in range(1000):
        for x, y in train_loader:
            loss = trainer.train_step(x, y)
            step += 1
            if step % 5000 == 0:
                acc = trainer.evaluate(test_loader)
                print(f"Step {step}: loss={loss:.4f}, test_acc={acc:.4f}")
            if step >= cfg["training"]["training_steps"]:
                break
        if step >= cfg["training"]["training_steps"]:
            break

    acc = trainer.evaluate(test_loader)
    print(f"Final test accuracy: {acc:.4f}")
    torch.save(model.state_dict(), "checkpoints/model.pt")


if __name__ == "__main__":
    main()
