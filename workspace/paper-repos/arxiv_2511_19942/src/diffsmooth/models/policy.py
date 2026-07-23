"""Policy model wrapper.

Loads a small instruct LLM (default: Qwen2.5-0.5B-Instruct, scaled down from the paper's
Qwen2.5-3B-Instruct — see configs/config.yaml) with optional 4-bit quantization and LoRA
adapters, and exposes sampling + log-probability computation needed by GRPO/DS-GRPO.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class PolicyModel(torch.nn.Module):
    """Wraps a HF causal LM as an RL policy: samples completions and scores log-probabilities."""

    def __init__(
        self,
        checkpoint: str,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            if load_in_4bit
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=quant_config,
            device_map=device if load_in_4bit else None,
            torch_dtype=torch.bfloat16,
        )
        if not load_in_4bit:
            self.model.to(device)

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.device = device

    @torch.no_grad()
    def generate(self, prompts: list[str], num_samples: int, temperature: float, max_new_tokens: int) -> list[list[str]]:
        """Sample `num_samples` completions per prompt. Returns [len(prompts)][num_samples] strings."""
        all_completions = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_samples,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            decoded = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            all_completions.append(decoded)
        return all_completions

    def logprob(self, prompt: str, completion: str) -> torch.Tensor:
        """Sum log pi(completion | prompt) over completion tokens. Used for Eq. 4/6 reward shaping."""
        full_text = prompt + completion
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        full_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)
        assert full_ids.shape[1] >= prompt_ids.shape[1], "completion produced no new tokens"

        with torch.no_grad():
            logits = self.model(full_ids).logits  # [1, T, V]
        log_probs = torch.log_softmax(logits, dim=-1)

        completion_start = prompt_ids.shape[1]
        target_ids = full_ids[0, completion_start:]
        token_log_probs = log_probs[0, completion_start - 1 : -1].gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs.sum()
