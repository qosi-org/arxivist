"""
evaluate.py
-----------
Standalone evaluation entrypoint for gMLP checkpoints.

Usage:
    # MLM perplexity
    python evaluate.py --checkpoint outputs/checkpoint_best.pt --task mlm

    # GLUE classification
    python evaluate.py --checkpoint outputs/finetune_sst2_best.pt --task sst2
    python evaluate.py --checkpoint outputs/finetune_mnli_best.pt --task mnli

    # SQuAD QA (F1)
    python evaluate.py --checkpoint outputs/finetune_squad_v2_best.pt --task squad_v2

    # ImageNet Top-1
    python evaluate.py --checkpoint outputs/gmlp_S_best.pt --task imagenet \\
        --data_dir data/imagenet/

Paper ref: Section 4, Tables 2, 3, 6
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gmlp.utils.config import gMLPConfig, ModelConfig
from gmlp.models.gmlp import gMLP
from gmlp.evaluation.metrics import (
    compute_perplexity, compute_accuracy, aggregate_runs
)


def parse_args():
    parser = argparse.ArgumentParser(description="gMLP Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--task", required=True,
                        choices=["mlm", "sst2", "mnli", "squad_v1", "squad_v2", "imagenet"])
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float32", "bf16", "fp16"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit evaluation to N batches (for quick checks)")
    return parser.parse_args()


def load_model_from_checkpoint(path: str, device: torch.device) -> tuple:
    """Load model and config from checkpoint."""
    ckpt = torch.load(path, map_location=device)
    raw_config = ckpt.get("config")
    if isinstance(raw_config, gMLPConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        # Reconstruct from dict (older checkpoints)
        mc = ModelConfig(**raw_config.get("model", {}))
        config = gMLPConfig(model=mc)
    else:
        raise ValueError("Checkpoint does not contain a gMLPConfig. Pass --config manually.")

    model = gMLP(config.model)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model, config


def eval_mlm(model, config, data_dir, batch_size, num_workers, max_batches, device, precision):
    """Evaluate MLM perplexity on C4 validation."""
    from transformers import AutoTokenizer
    from gmlp.data.mlm_dataset import MLMDataset

    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)
    dataset = MLMDataset(
        tokenizer=tokenizer,
        max_seq_len=config.model.seq_len,
        mlm_probability=0.15,
        dataset_config="realnewslike",
        split="validation",
        use_streaming=False,
        data_dir=data_dir,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    use_amp = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    losses = []
    model.set_task("mlm")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=input_ids, labels=labels)
            losses.append(out.loss.item())

    ppl = compute_perplexity(losses)
    print(f"\nMLM Validation Perplexity: {ppl:.4f}")
    print(f"  (avg CE loss = {sum(losses)/len(losses):.4f} over {len(losses)} batches)")
    return ppl


def eval_glue(model, task, data_dir, batch_size, num_workers, max_batches, device):
    """Evaluate GLUE classification accuracy."""
    from transformers import AutoTokenizer
    from gmlp.data.glue_dataset import SST2Dataset, MNLIDataset

    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)
    if task == "sst2":
        dataset = SST2Dataset(tokenizer, split="validation", data_dir=data_dir)
    else:
        dataset = MNLIDataset(tokenizer, split="validation_matched", data_dir=data_dir)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.set_task("classification")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids)
            all_preds.append(out.logits.argmax(dim=-1))
            all_labels.append(labels)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = compute_accuracy(preds, labels) * 100
    print(f"\n{task.upper()} Accuracy: {acc:.2f}%")
    return acc


def eval_imagenet(model, data_dir, batch_size, num_workers, max_batches, device):
    """Evaluate ImageNet Top-1 / Top-5 accuracy."""
    from gmlp.data.imagenet_dataset import ImageNetDataset
    dataset = ImageNetDataset(data_dir=data_dir, split="val", autoaugment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    correct1 = correct5 = total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            out = model(pixel_values=images)
            _, pred5 = out.logits.topk(5, dim=-1)
            correct1 += (pred5[:, 0] == labels).sum().item()
            correct5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

    top1 = 100.0 * correct1 / max(total, 1)
    top5 = 100.0 * correct5 / max(total, 1)
    print(f"\nImageNet Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")
    return top1, top5


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")

    model, config = load_model_from_checkpoint(args.checkpoint, device)
    print(f"Model: {model}")

    if args.task == "mlm":
        eval_mlm(model, config, args.data_dir, args.batch_size,
                 args.num_workers, args.max_batches, device, args.precision)
    elif args.task in ("sst2", "mnli"):
        eval_glue(model, args.task, args.data_dir, args.batch_size,
                  args.num_workers, args.max_batches, device)
    elif args.task == "imagenet":
        eval_imagenet(model, args.data_dir, args.batch_size,
                      args.num_workers, args.max_batches, device)
    else:
        print(f"[INFO] {args.task} evaluation not yet implemented. Use finetune.py.")


if __name__ == "__main__":
    main()
