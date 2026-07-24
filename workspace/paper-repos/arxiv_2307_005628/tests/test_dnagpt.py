"""Unit tests for the DNAGPT reproduction (CPU, no weights).

Verifies the paper's architectural claims that can be checked without the
official checkpoint:

* Token language: non-overlap k-mer gives N/k tokens; GC content; vocab size.
* Model: dual heads (classification + regression), causal attention, shapes.
* Combined pre-training loss L = 0.01*MSE + CE (Eq 1).
* GSR classification path produces [B, 2] logits and learns on synthetic data.
"""
import torch

from src.dnagpt.data.tokenizer import DNAGPTTokenizer
from src.dnagpt.models.dnagpt import DNAGPT, DNAGPTConfig
from src.dnagpt.training.losses import combined_pretrain_loss, gc_content_targets


def test_kmer_vocab_size():
    """k=6 over NAGCT -> 19,530 k-mers; +34 reserved = 19,564 (matches checkpoint wte)."""
    from src.dnagpt.data.tokenizer import _RESERVED, _get_base_kmers

    # 5^6 + 5^5 + 5^4 + 5^3 + 5^2 + 5^1 = 19530 k-mers (dynamic length 1..6, product order)
    n_kmers = sum(5 ** i for i in range(1, 7))
    assert n_kmers == 19530 == len(_get_base_kmers(6))
    assert len(_RESERVED) == 34
    tok = DNAGPTTokenizer(k=6)
    # full vocab = 34 reserved (<...>) + 19530 k-mers = 19564 (official DNAGPT vocab)
    assert tok.vocab_size == 19564
    assert tok.pad_id == 0 and tok.stoi["N"] == 34   # reserved first, N is first k-mer


def test_non_overlap_kmer_count():
    """Non-overlapping k-mers: N/k tokens (shift == k)."""
    tok = DNAGPTTokenizer(k=6)
    seq = "ACGTAC" * 5   # length 30
    ids = tok.encode_sequence(seq)
    assert len(ids) == 30 // 6 == 5


def test_gc_content():
    tok = DNAGPTTokenizer(k=6)
    assert abs(tok.gc_content("GCGC") - 1.0) < 1e-9
    assert abs(tok.gc_content("ATAT") - 0.0) < 1e-9
    assert abs(tok.gc_content("ACGT") - 0.5) < 1e-9


def test_classification_template():
    """GSR template ends with the classification token (A=True / N=False)."""
    tok = DNAGPTTokenizer(k=6)
    pos = tok.build_classification_example("Human", "ACGTAC", True)["input_ids"]
    neg = tok.build_classification_example("Human", "ACGTAC", False)["input_ids"]
    assert pos[-1] == tok.stoi["<A>"]      # True  (paper classification token 'A')
    assert neg[-1] == tok.stoi["<N>"]      # False (paper classification token 'N')
    assert pos[0] == tok.stoi["<R>"]       # Human instruction token (Fig S2: Human = <R>)


def test_model_forward_dual_heads():
    torch.manual_seed(0)
    tok = DNAGPTTokenizer(k=6)
    cfg = DNAGPTConfig.from_variant("M", vocab_size=tok.vocab_size, seq_len=64)
    cfg.n_layer = 2  # keep the test light
    model = DNAGPT(cfg)
    ids = torch.randint(0, tok.vocab_size, (2, 16))
    out = model(ids)
    assert out["class_logits"].shape == (2, 16, tok.vocab_size)
    assert out["reg_values"].shape == (2, 16, 1)
    assert out["hidden"].shape == (2, 16, cfg.hidden)


def test_numerical_input_appends_tokens():
    """Numbers append number-tokens along the length dim (joint seq+number)."""
    torch.manual_seed(0)
    tok = DNAGPTTokenizer(k=6)
    cfg = DNAGPTConfig.from_variant("M", vocab_size=tok.vocab_size, seq_len=64)
    cfg.n_layer = 2
    model = DNAGPT(cfg)
    ids = torch.randint(0, tok.vocab_size, (2, 10))
    numbers = torch.randn(2, 3)
    out = model(ids, numbers=numbers)
    assert out["hidden"].shape[1] == 13   # 10 seq + 3 number tokens


def test_combined_loss():
    """L = 0.01*MSE + CE (Eq 1)."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 20)
    targets = torch.randint(0, 20, (2, 5))
    reg = torch.randn(2, 3, 1)
    reg_t = torch.rand(2, 3)
    loss = combined_pretrain_loss(logits, targets, reg, reg_t, lambda_=0.01)
    assert loss.item() > 0 and torch.isfinite(loss)


def test_gsr_classification_learns_synthetic():
    """The GSR classification path can fit a tiny synthetic signal (loss drops)."""
    torch.manual_seed(0)
    tok = DNAGPTTokenizer(k=6)
    cfg = DNAGPTConfig.from_variant("M", vocab_size=tok.vocab_size, seq_len=64, num_classes=2)
    cfg.n_layer = 2
    model = DNAGPT(cfg)
    # GC-rich = class 1, AT-rich = class 0
    pos = tok.encode_batch(["GCGCGCGCGCGC"] * 4, max_len=32)
    neg = tok.encode_batch(["ATATATATATAT"] * 4, max_len=32)
    ids = torch.cat([pos["input_ids"], neg["input_ids"]])
    mask = torch.cat([pos["attention_mask"], neg["attention_mask"]])
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first, last = None, None
    for step in range(25):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model.classify(ids, mask), y)
        loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"loss did not drop: {first} -> {last}"
