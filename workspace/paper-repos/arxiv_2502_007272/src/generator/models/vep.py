"""Zero-shot variant effect prediction (VEP) for GENERator.

Implements the alignment-free VEP from paper Sec 4.5:
  VEP(R -> A) = log p(s_i = R | S\\i) - log p(s_i = A | S\\i)
Because GENERator uses 6-mer tokenization, single-nucleotide probabilities are
recovered by marginalizing token-level probabilities over the tokens covering
position i:  p(s_i = X) = sum_{t in T_i} p(t) * 1[t_j = X].

This is a faithful but simplified implementation: it scores the variant using the
causal next-token distribution with the variant placed near the sequence end.
"""
from __future__ import annotations

import torch


class VEPScorer:
    """Score single-nucleotide variants with a pretrained GENERator LM.

    Args:
        model_name: HF repo id of the GENERator causal LM.
        device: torch device string.
    """

    def __init__(self, model_name: str = "GenerTeam/GENERator-eukaryote-1.2b-base", device: str = "cuda") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()

    def _nuc_logprob(self, context: str, nucleotide: str) -> float:
        """Log-prob of `nucleotide` following `context` under next-6mer prediction,
        marginalized to single-nucleotide resolution over the vocab."""
        enc = self.tok(context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1]  # next-token logits [vocab]
        probs = torch.softmax(logits.float(), dim=-1)  # [vocab]
        # Marginalize: sum prob over 6-mer tokens whose first base == nucleotide.
        vocab = self.tok.get_vocab()  # token_str -> id
        total = 0.0
        for tok_str, tid in vocab.items():
            base = tok_str[0] if tok_str else ""
            if base.upper() == nucleotide.upper():
                total += probs[tid].item()
        return float(torch.log(torch.tensor(max(total, 1e-12))))

    def score_variant(self, seq: str, pos: int, ref: str, alt: str) -> float:
        """Return VEP = log p(ref) - log p(alt) with the variant at sequence end.

        Positive => reference allele favored (stronger constraint), consistent
        with the paper's convention.
        """
        context = seq[:pos]  # causal context up to the variant position
        lp_ref = self._nuc_logprob(context, ref)
        lp_alt = self._nuc_logprob(context, alt)
        return lp_ref - lp_alt
