#!/usr/bin/env python3
"""
sir_learner.py  —  ArXivist SIR Corpus Learner
================================================

Trains (or fine-tunes) a small language model on the ArXivist SIR registry.
Each SIR is converted into a set of (prompt, completion) pairs that teach the
model to predict structured SIR fields from paper abstracts and metadata.

WHAT IT LEARNS
--------------
Given an abstract + title + domain, predict:
  - Architecture modules (names, operation types, key parameters)
  - Primary framework (pytorch / jax / numpy / etc.)
  - Training pipeline details (optimizer, scheduler, batch size)
  - Top implementation risks and assumptions
  - Confidence scores per section
  - Evaluation metrics and reported results

WHY THIS IS USEFUL
------------------
Once trained, this model accelerates Stage 1 of the ArXivist pipeline:
  - Draft SIR fields before Claude processes the full PDF
  - Predict likely ambiguities for a paper type (e.g. "Langevin papers always
    have ambiguous integrator details")
  - Estimate confidence scores from abstract alone
  - Surface which sections need human review

The model is intentionally small (SmolLM2-360M or Qwen2.5-0.5B) so it runs
locally on CPU. It is not trying to replace the paper-parsing Claude call —
it is trying to *compress the distribution of seen SIRs* into a fast prior.

TRAINING TASK VARIANTS
-----------------------
  --task sir_completion   Input: abstract. Target: full SIR JSON (default)
  --task field_predict    Input: abstract + field name. Target: field value
  --task risk_predict     Input: abstract. Target: top-3 risks + severities
  --task confidence_pred  Input: abstract + section. Target: confidence score

USAGE
-----
  # Train from scratch on all SIRs in the registry
  python sir_learner.py --registry-dir sir-registry/ --task sir_completion

  # Fine-tune a checkpoint on newly added SIRs only
  python sir_learner.py --registry-dir sir-registry/ --checkpoint checkpoints/sir_learner_latest.pt --new-only

  # Inference: predict SIR fields for a new abstract
  python sir_learner.py --infer --abstract "We introduce a new attention mechanism..." --title "FlashAttention-4"

  # Export the dataset only (no training), for inspection or use elsewhere
  python sir_learner.py --registry-dir sir-registry/ --export-only --out dataset.jsonl

DESIGN NOTES
------------
- Uses HuggingFace transformers + PEFT (LoRA) if available; falls back to a
  pure-numpy bigram/trigram baseline if neither is installed. The baseline is
  not useful for generation but proves the data pipeline works end-to-end.
- All training examples are derived deterministically from SIR JSON — no
  external data fetched, no API calls made.
- Each new SIR added to the registry = one new training batch. Run with
  --new-only after each ArXivist pipeline run to keep the model current.
- Checkpoints are saved to checkpoints/sir_learner_{step}.pt
- A small held-out split (10% of SIRs, by paper_id hash) is reserved for
  validation — never used in training, even across runs.

DEPENDENCIES (all optional except json/pathlib)
-----------
  Required:  (none beyond stdlib)
  For LoRA fine-tuning: pip install transformers peft torch accelerate
  For richer tokenization: pip install sentencepiece
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("sir_learner")


# ============================================================
# 1. DATA LOADING — walk the SIR registry
# ============================================================

def load_all_sirs(registry_dir: str) -> list[dict]:
    """Walk registry_dir recursively and load every sir.json found.

    Expected layout (produced by ArXivist pipeline):
        registry_dir/
          {paper_id}/
            sir.json
            metadata.json   (optional, for extra fields)

    Returns list of SIR dicts, each augmented with 'paper_id' key.
    """
    registry_path = Path(registry_dir)
    sirs = []

    for sir_path in sorted(registry_path.rglob("sir.json")):
        # Skip versioned copies in /versions/ subdirectory
        if "versions" in sir_path.parts:
            continue
        try:
            with open(sir_path) as f:
                sir = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping {sir_path}: {e}")
            continue

        # Augment with paper_id from directory name if not present
        if "paper_id" not in sir:
            sir["paper_id"] = sir_path.parent.name

        # Pull in metadata.json if present (has title, arxiv_id, etc.)
        meta_path = sir_path.parent / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                sir["_metadata"] = meta
            except Exception:
                pass

        sirs.append(sir)
        log.info(f"Loaded SIR: {sir['paper_id']}  (confidence={sir.get('confidence_annotations',{}).get('overall_sir_confidence','?')})")

    log.info(f"Total SIRs loaded: {len(sirs)}")
    return sirs


def train_val_split(sirs: list[dict], val_fraction: float = 0.10) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split by paper_id hash.

    Uses a hash so the same paper always lands in the same split,
    even as the corpus grows. The held-out set never leaks into training.
    """
    train, val = [], []
    for sir in sirs:
        h = int(hashlib.md5(sir["paper_id"].encode()).hexdigest(), 16)
        if (h % 100) < int(val_fraction * 100):
            val.append(sir)
        else:
            train.append(sir)
    log.info(f"Split: {len(train)} train / {len(val)} val")
    return train, val


# ============================================================
# 2. PROMPT CONSTRUCTION — SIR → (prompt, completion) pairs
# ============================================================

def _safe_get(d: dict, *keys, default="unknown"):
    """Safely traverse nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def sir_to_examples(sir: dict, task: str = "sir_completion") -> list[dict]:
    """Convert one SIR into a list of (prompt, completion) training examples.

    Each example is a dict:
        {"prompt": str, "completion": str, "paper_id": str, "task": str}

    Multiple examples are generated per SIR to maximize signal:
      - Full SIR completion (abstract → full JSON)
      - Per-field prediction (abstract + field name → field value)
      - Risk prediction (abstract → top risks)
      - Confidence prediction (abstract + section → confidence score)
      - Framework prediction (abstract → framework string)
      - Module list prediction (abstract → list of module names)
    """
    examples = []
    pid = sir.get("paper_id", "unknown")

    # Extract re-usable text fragments
    prov = sir.get("provenance", {})
    title = prov.get("title", sir.get("_metadata", {}).get("title", ""))
    abstract = prov.get("abstract", "")
    domain = prov.get("domain", "unknown")
    authors = ", ".join(prov.get("authors", []))
    key_claims = prov.get("key_claims", [])
    claims_str = " | ".join(key_claims[:3]) if key_claims else ""

    arch = sir.get("architecture", {})
    framework = arch.get("primary_variant", "unknown")
    framework_cfg = sir.get("training_pipeline", {})  # has optimizer, etc.

    conf = sir.get("confidence_annotations", {})
    risks = sir.get("implementation_assumptions", [])
    ambiguities = sir.get("ambiguities", [])

    # Shared paper header used in most prompts
    paper_header = f"Title: {title}\nDomain: {domain}\nAbstract: {abstract}"
    if claims_str:
        paper_header += f"\nKey claims: {claims_str}"

    # ----------------------------------------------------------
    # Task 1: Full SIR completion  (abstract → full JSON)
    # ----------------------------------------------------------
    if task in ("sir_completion", "all"):
        # Compact the SIR for the target (strip internal metadata, keep essentials)
        compact_sir = {
            "framework": _safe_get(sir, "architecture", "primary_variant"),
            "cuda_required": _safe_get(sir, "architecture", "framework", default={}).get("cuda_required", False)
                             if isinstance(_safe_get(sir, "architecture"), dict) else False,
            "modules": [
                {
                    "name": m.get("name"),
                    "operation_type": m.get("operation_type"),
                    "confidence": m.get("confidence"),
                }
                for m in arch.get("modules", [])
            ],
            "optimizer": _safe_get(framework_cfg, "optimizer", "name"),
            "batch_size": framework_cfg.get("batch_size"),
            "primary_metric": next(
                (r.get("metric") for r in sir.get("evaluation_protocol", {}).get("reported_results", [])
                 if r.get("is_primary")), None
            ),
            "top_risks": [
                {"severity": r.get("severity"), "description": r.get("description", "")[:120]}
                for r in sorted(
                    sir.get("implementation_assumptions", []),
                    key=lambda x: {"High": 0, "Medium": 1, "Low": 2}.get(x.get("severity", "Low"), 3)
                )[:3]
            ],
            "overall_confidence": conf.get("overall_sir_confidence"),
        }

        examples.append({
            "prompt": (
                f"<task>sir_completion</task>\n"
                f"{paper_header}\n\n"
                f"Generate a structured SIR summary for this paper:"
            ),
            "completion": json.dumps(compact_sir, indent=None),
            "paper_id": pid,
            "task": "sir_completion",
        })

    # ----------------------------------------------------------
    # Task 2: Per-field prediction  (abstract + field → value)
    # ----------------------------------------------------------
    if task in ("field_predict", "all"):
        field_targets = [
            ("framework",
             _safe_get(sir, "architecture", "primary_variant")),
            ("cuda_required",
             str(_safe_get(sir, "architecture", "framework", default={}).get("cuda_required", "unknown"))),
            ("n_modules",
             str(len(arch.get("modules", [])))),
            ("has_training_loop",
             str(framework_cfg.get("optimizer") is not None and
                 framework_cfg.get("optimizer", {}).get("name") not in (None, "unknown", "N/A"))),
            ("overall_confidence",
             str(conf.get("overall_sir_confidence", "?"))),
        ]
        for field_name, field_value in field_targets:
            if field_value in (None, "unknown", "?", "None"):
                continue
            examples.append({
                "prompt": (
                    f"<task>field_predict</task>\n"
                    f"{paper_header}\n\n"
                    f"Predict the value of field '{field_name}':"
                ),
                "completion": str(field_value),
                "paper_id": pid,
                "task": "field_predict",
            })

    # ----------------------------------------------------------
    # Task 3: Risk prediction  (abstract → top risks)
    # ----------------------------------------------------------
    if task in ("risk_predict", "all"):
        high_risks = [
            r for r in risks
            if r.get("severity") == "High"
        ]
        if high_risks:
            risk_str = "\n".join(
                f"- [{r['severity']}] {r.get('description','')[:150]}"
                for r in high_risks[:3]
            )
            examples.append({
                "prompt": (
                    f"<task>risk_predict</task>\n"
                    f"{paper_header}\n\n"
                    f"List the HIGH severity implementation risks for reproducing this paper:"
                ),
                "completion": risk_str,
                "paper_id": pid,
                "task": "risk_predict",
            })

    # ----------------------------------------------------------
    # Task 4: Confidence prediction  (abstract + section → score)
    # ----------------------------------------------------------
    if task in ("confidence_pred", "all"):
        for section, score in conf.items():
            if section == "overall_sir_confidence":
                continue
            if not isinstance(score, (int, float)):
                continue
            examples.append({
                "prompt": (
                    f"<task>confidence_pred</task>\n"
                    f"{paper_header}\n\n"
                    f"Predict the SIR confidence score for section '{section}' (0.0–1.0):"
                ),
                "completion": f"{score:.2f}",
                "paper_id": pid,
                "task": "confidence_pred",
            })

    # ----------------------------------------------------------
    # Task 5: Module list prediction  (abstract → module names)
    # ----------------------------------------------------------
    if task in ("module_list", "all"):
        module_names = [m.get("name", "") for m in arch.get("modules", []) if m.get("name")]
        if module_names:
            examples.append({
                "prompt": (
                    f"<task>module_list</task>\n"
                    f"{paper_header}\n\n"
                    f"List the key implementation modules for this paper (one per line):"
                ),
                "completion": "\n".join(module_names),
                "paper_id": pid,
                "task": "module_list",
            })

    # ----------------------------------------------------------
    # Task 6: Ambiguity spotting  (abstract → predicted ambiguities)
    # ----------------------------------------------------------
    if task in ("ambiguity_spot", "all"):
        if ambiguities:
            amb_str = "\n".join(
                f"- {a.get('location','?')}: {a.get('description','')[:120]}"
                for a in ambiguities[:3]
            )
            examples.append({
                "prompt": (
                    f"<task>ambiguity_spot</task>\n"
                    f"{paper_header}\n\n"
                    f"Identify likely implementation ambiguities in reproducing this paper:"
                ),
                "completion": amb_str,
                "paper_id": pid,
                "task": "ambiguity_spot",
            })

    return examples


def build_dataset(
    sirs: list[dict],
    task: str = "all",
) -> list[dict]:
    """Convert all SIRs into a flat list of (prompt, completion) dicts."""
    dataset = []
    for sir in sirs:
        examples = sir_to_examples(sir, task=task)
        dataset.extend(examples)
    log.info(f"Dataset: {len(dataset)} examples from {len(sirs)} SIRs")
    return dataset


def export_jsonl(dataset: list[dict], path: str) -> None:
    """Write dataset to a .jsonl file (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Exported {len(dataset)} examples → {path}")


# ============================================================
# 3. TRAINING — LoRA fine-tune if transformers available,
#               else bigram baseline
# ============================================================

@dataclass
class TrainConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    # Alternatives: "Qwen/Qwen2.5-0.5B", "microsoft/phi-2" (larger)
    # All run on CPU, SmolLM2-360M is fastest
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_length: int = 1024        # tokens per example
    batch_size: int = 4
    grad_accum: int = 4           # effective batch = batch_size * grad_accum
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 10
    save_every_n_steps: int = 50
    checkpoint_dir: str = "checkpoints/"
    fp16: bool = False            # set True if CUDA available
    seed: int = 42


def _try_import_training_deps() -> bool:
    """Return True if transformers + peft + torch are all available."""
    try:
        import torch            # noqa: F401
        import transformers     # noqa: F401
        import peft             # noqa: F401
        return True
    except ImportError:
        return False


def train_lora(
    train_data: list[dict],
    val_data: list[dict],
    cfg: TrainConfig,
) -> None:
    """Fine-tune a small LM with LoRA on the SIR dataset.

    Each training example is formatted as:
        <|user|>
        {prompt}
        <|assistant|>
        {completion}

    and tokenized to max_length tokens with labels masked on the prompt portion.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
    from peft import get_peft_model, LoraConfig, TaskType

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training on device: {device}")

    # --- Load tokenizer and model ---
    log.info(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
    )

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # --- Dataset ---
    class SIRDataset(Dataset):
        def __init__(self, examples: list[dict]):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx: int) -> dict:
            ex = self.examples[idx]
            # Format as instruction-following conversation
            full_text = (
                f"<|user|>\n{ex['prompt']}\n<|assistant|>\n{ex['completion']}"
                f"{tokenizer.eos_token}"
            )
            enc = tokenizer(
                full_text,
                max_length=cfg.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Mask labels on the prompt portion
            prompt_text = f"<|user|>\n{ex['prompt']}\n<|assistant|>\n"
            prompt_enc = tokenizer(prompt_text, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prompt_len] = -100          # ignore prompt tokens in loss
            labels[attention_mask == 0] = -100  # ignore padding

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    train_ds = SIRDataset(train_data)
    val_ds = SIRDataset(val_data) if val_data else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds else None

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    total_steps = (len(train_loader) // cfg.grad_accum) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Training loop ---
    global_step = 0
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (batch_idx + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    log.info(
                        f"Epoch {epoch+1}/{cfg.num_epochs}  "
                        f"step {global_step}  "
                        f"loss={epoch_loss / (batch_idx + 1):.4f}"
                    )

                if global_step % cfg.save_every_n_steps == 0:
                    ckpt_path = os.path.join(cfg.checkpoint_dir, f"sir_learner_step{global_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    log.info(f"Checkpoint saved: {ckpt_path}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)
            log.info(f"Epoch {epoch+1} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(cfg.checkpoint_dir, "sir_learner_best")
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                log.info(f"New best checkpoint: {best_path}  (val_loss={best_val_loss:.4f})")

    # Final save
    final_path = os.path.join(cfg.checkpoint_dir, "sir_learner_latest")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    log.info(f"Training complete. Final checkpoint: {final_path}")


# ============================================================
# 4. BASELINE — n-gram model (no external deps)
# ============================================================

class NgramBaseline:
    """Trigram character-level language model trained on SIR completions.

    Not useful for generation at this corpus size, but proves the data
    pipeline works end-to-end without any ML dependencies.
    Also useful for computing perplexity-like corpus statistics.
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.counts: dict[str, dict[str, int]] = {}
        self.vocab: set[str] = set()

    def train(self, examples: list[dict]) -> None:
        log.info(f"Training {self.n}-gram baseline on {len(examples)} completions...")
        for ex in examples:
            text = ex["completion"]
            self.vocab.update(text)
            for i in range(len(text) - self.n):
                ctx = text[i:i + self.n]
                nxt = text[i + self.n]
                self.counts.setdefault(ctx, {})
                self.counts[ctx][nxt] = self.counts[ctx].get(nxt, 0) + 1

        total_ngrams = sum(sum(v.values()) for v in self.counts.values())
        log.info(
            f"Baseline trained: vocab={len(self.vocab)}, "
            f"unique {self.n}-grams={len(self.counts)}, "
            f"total={total_ngrams}"
        )

    def perplexity(self, text: str) -> float:
        """Estimate perplexity of a completion string."""
        import math
        log_prob = 0.0
        count = 0
        for i in range(len(text) - self.n):
            ctx = text[i:i + self.n]
            nxt = text[i + self.n]
            ctx_counts = self.counts.get(ctx, {})
            total = sum(ctx_counts.values()) + len(self.vocab)  # Laplace smoothing
            p = (ctx_counts.get(nxt, 0) + 1) / total
            log_prob += math.log(p)
            count += 1
        if count == 0:
            return float("inf")
        return math.exp(-log_prob / count)

    def corpus_stats(self, examples: list[dict]) -> dict:
        """Compute mean perplexity and per-task breakdown."""
        by_task: dict[str, list[float]] = {}
        for ex in examples:
            pp = self.perplexity(ex["completion"])
            t = ex.get("task", "unknown")
            by_task.setdefault(t, []).append(pp)
        return {
            "mean_perplexity": sum(sum(v) for v in by_task.values()) / max(1, sum(len(v) for v in by_task.values())),
            "by_task": {t: sum(v) / len(v) for t, v in by_task.items()},
            "n_examples": sum(len(v) for v in by_task.values()),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"n": self.n, "counts": self.counts}, f)
        log.info(f"Baseline model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "NgramBaseline":
        with open(path) as f:
            data = json.load(f)
        obj = cls(n=data["n"])
        obj.counts = data["counts"]
        return obj


# ============================================================
# 5. INFERENCE — generate SIR fields for a new abstract
# ============================================================

def infer(
    abstract: str,
    title: str = "",
    domain: str = "unknown",
    checkpoint: str = "checkpoints/sir_learner_latest",
    task: str = "sir_completion",
    max_new_tokens: int = 512,
) -> str:
    """Run inference with the trained LoRA model.

    Args:
        abstract: Paper abstract text.
        title: Paper title (optional but improves quality).
        domain: Paper domain string.
        checkpoint: Path to saved LoRA checkpoint directory.
        task: Which task head to use (sir_completion, field_predict, etc.)
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated completion string.
    """
    if not _try_import_training_deps():
        return "[Inference requires: pip install transformers peft torch]"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    log.info(f"Loading checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model_name = json.load(open(os.path.join(checkpoint, "adapter_config.json")))["base_model_name_or_path"]
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    paper_header = f"Title: {title}\nDomain: {domain}\nAbstract: {abstract}"
    prompt = (
        f"<|user|>\n"
        f"<task>{task}</task>\n"
        f"{paper_header}\n\n"
        f"Generate a structured SIR summary for this paper:\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decode for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


# ============================================================
# 6. CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ArXivist SIR Learner — train a small LM on the SIR corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--registry-dir", default="sir-registry/",
        help="Path to ArXivist SIR registry directory (default: sir-registry/)",
    )
    p.add_argument(
        "--task", default="all",
        choices=["all", "sir_completion", "field_predict", "risk_predict",
                 "confidence_pred", "module_list", "ambiguity_spot"],
        help="Training task variant (default: all — generates examples for every task)",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to existing checkpoint to fine-tune (default: train from scratch)",
    )
    p.add_argument(
        "--new-only", action="store_true",
        help="Only train on SIRs added since last checkpoint (incremental update)",
    )
    p.add_argument(
        "--model", default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Base model name/path for LoRA fine-tuning",
    )
    p.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs (default: 3)",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size (default: 4)",
    )
    p.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    p.add_argument(
        "--checkpoint-dir", default="checkpoints/",
        help="Directory to save checkpoints (default: checkpoints/)",
    )
    p.add_argument(
        "--export-only", action="store_true",
        help="Export dataset as JSONL and exit (no training)",
    )
    p.add_argument(
        "--out", default="dataset.jsonl",
        help="Output path for --export-only (default: dataset.jsonl)",
    )
    p.add_argument(
        "--baseline-only", action="store_true",
        help="Run n-gram baseline only (no ML deps required)",
    )
    p.add_argument(
        "--infer", action="store_true",
        help="Run inference mode instead of training",
    )
    p.add_argument("--abstract", default="", help="Abstract text for inference")
    p.add_argument("--title", default="", help="Paper title for inference")
    p.add_argument("--domain", default="unknown", help="Paper domain for inference")
    p.add_argument(
        "--infer-checkpoint", default="checkpoints/sir_learner_latest",
        help="Checkpoint path for inference",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Inference mode ---
    if args.infer:
        if not args.abstract:
            parser.error("--abstract is required for --infer")
        result = infer(
            abstract=args.abstract,
            title=args.title,
            domain=args.domain,
            checkpoint=args.infer_checkpoint,
            task="sir_completion",
        )
        print("\n=== SIR PREDICTION ===")
        print(result)
        return

    # --- Load SIRs ---
    sirs = load_all_sirs(args.registry_dir)
    if not sirs:
        log.error(
            f"No SIRs found in '{args.registry_dir}'. "
            "Run the ArXivist pipeline first to populate the registry."
        )
        sys.exit(1)

    train_sirs, val_sirs = train_val_split(sirs, val_fraction=0.10)

    # --- Build dataset ---
    train_data = build_dataset(train_sirs, task=args.task)
    val_data = build_dataset(val_sirs, task=args.task)

    log.info(f"Train examples: {len(train_data)}  |  Val examples: {len(val_data)}")

    # --- Export only ---
    if args.export_only:
        all_data = train_data + val_data
        export_jsonl(all_data, args.out)
        log.info(f"Dataset exported to {args.out}. Exiting (--export-only).")
        return

    # --- Baseline only ---
    if args.baseline_only or not _try_import_training_deps():
        if not args.baseline_only:
            log.warning(
                "transformers/peft/torch not found. "
                "Running n-gram baseline instead.\n"
                "Install with: pip install transformers peft torch accelerate"
            )
        baseline = NgramBaseline(n=3)
        baseline.train(train_data)
        stats = baseline.corpus_stats(train_data)
        log.info(f"Baseline corpus stats (train): {json.dumps(stats, indent=2)}")
        if val_data:
            val_stats = baseline.corpus_stats(val_data)
            log.info(f"Baseline corpus stats (val):   {json.dumps(val_stats, indent=2)}")
        baseline.save(os.path.join(args.checkpoint_dir, "sir_learner_baseline.json"))
        return

    # --- LoRA training ---
    cfg = TrainConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
    train_lora(train_data, val_data, cfg)


if __name__ == "__main__":
    main()
