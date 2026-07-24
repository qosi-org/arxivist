"""The DNAGPT token language (Sec 2.2, Fig 2, Fig S2).

This reproduction matches the authors' released tokenizer
(TencentAILabHealthcare/DNAGPT `dna_gpt/tokenizer.py`) **exactly**, so the
official pretrained embedding table is indexed correctly:

    bases      = 'NAGCT'                          (N first — unknown base)
    kmers      = product(bases, repeat=i) for i in 1..k   (dynamic k-mer)
    vocab      = reserved_tokens (each wrapped <...>) + kmers
    <P> = pad, index 0 = unk

For the 0.1B models k=6 -> 19,530 k-mers; with 34 reserved tokens the vocab is
19,564, matching the checkpoint's `wte` shape (19564, 768).

The reserved (instruction / classification / connection / special) tokens follow
Fig S2: organism tokens (Human=<R>, Mouse=<S>, ...), classification <A>/<N>
(True/False), instruction <B> (classification) / <M> (number), connection
<+>/<=>, pad <P>, and reserved digits/symbols to fill 34.
"""
from __future__ import annotations

import itertools as it
from typing import Dict, List

_BASES = "NAGCT"  # N first — matches the authors' tokenizer exactly

# The 34 reserved tokens (wrapped <...> in the vocab), in a fixed order.
# Organism + task instruction + classification + connection + pad + fillers.
# Names map to the paper's Fig S2 letters so instruction ids are stable.
_RESERVED = [
    "P",                    # <P> pad (authors' pad token)
    "R", "S", "V", "L", "O", "Q", "I", "K", "U",   # organisms: Human,Mouse,Bovine,Danio,Drosophila,Ecoli,Arabidopsis,Celegans,Yeast
    "B", "M",              # instruction: classification / number
    "A", "N",              # classification: True / False
    "+", "=",              # connection: fuse / input=output
]
# pad out to 34 reserved tokens with the paper's "reserved" set (digits/symbols)
_RESERVED += [str(d) for d in range(10)]           # 0-9
_RESERVED += ["*", "/", "W", "Z", "H", "J", "X", "Y"]   # fillers
_RESERVED = _RESERVED[:34]
assert len(_RESERVED) == 34, f"reserved must be 34, got {len(_RESERVED)}"

# organism display-name -> reserved letter (Fig S2)
_SPECIES_TOKEN = {
    "Human": "R", "Mouse": "S", "Bovine": "V", "Danio": "L", "Drosophila": "O",
    "Ecoli": "Q", "Arabidopsis": "I", "Celegans": "K", "Yeast": "U",
    "FruitFly": "O",  # Drosophila melanogaster == fruit fly
}
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def _get_base_kmers(k: int) -> List[str]:
    """All k-mers of length 1..k over NAGCT, in itertools.product order."""
    kmers: List[tuple] = []
    for i in range(1, k + 1):
        kmers += list(it.product(_BASES, repeat=i))
    return ["".join(m) for m in kmers]


class DNAGPTTokenizer:
    """Non-overlapping k-mer tokenizer matching the official DNAGPT vocab."""

    def __init__(self, k: int = 6) -> None:
        self.k = k
        self.reserved = [f"<{t}>" for t in _RESERVED]           # <P>, <R>, ...
        self.kmers = _get_base_kmers(k)                          # 19,530 for k=6
        self.idx_to_token = self.reserved + self.kmers           # reserved FIRST
        self.stoi = {t: i for i, t in enumerate(self.idx_to_token)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.pad_id = self.stoi["<P>"]
        self.unk_id = 0                                          # authors: unk_id = 0

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    # --- sequence encoding (non-overlapping k-mers) ------------------------
    def encode_sequence(self, seq: str) -> List[int]:
        """Chunk DNA into non-overlapping k-mers (shift == k) -> ids."""
        seq = seq.upper()
        ids = []
        for i in range(0, len(seq), self.k):
            piece = seq[i:i + self.k]
            ids.append(self.stoi.get(piece, self.unk_id))
        return ids

    def _spec(self, species: str) -> int:
        letter = _SPECIES_TOKEN.get(species, "R")
        return self.stoi.get(f"<{letter}>", self.unk_id)

    # --- fine-tuning templates (Fig 2b) -----------------------------------
    def build_classification_example(self, species: str, seq: str, label: bool) -> Dict[str, List[int]]:
        """GSR template: <species> <seq...> <=> <B> <A|N>  (True/False)."""
        cls_tok = "<A>" if label else "<N>"
        ids = ([self._spec(species)] + self.encode_sequence(seq)
               + [self.stoi["<=>"], self.stoi["<B>"], self.stoi[cls_tok]])
        return {"input_ids": ids}

    def build_gsr_classification_input(self, species: str, seq: str) -> List[int]:
        """Input side only (label predicted by the classification head)."""
        return [self._spec(species)] + self.encode_sequence(seq) + [self.stoi["<=>"], self.stoi["<B>"]]

    def encode_batch(self, seqs: List[str], species: str = "Human", max_len: int = 512):
        import torch
        rows, masks = [], []
        for s in seqs:
            ids = self.build_gsr_classification_input(species, s)[:max_len]
            attn = [1] * len(ids)
            pad = max_len - len(ids)
            rows.append(ids + [self.pad_id] * pad)
            masks.append(attn + [0] * pad)
        return {"input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long)}

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def gc_content(seq: str) -> float:
        """GC ratio in [0,1] (GC-content pre-training target)."""
        seq = seq.upper()
        if not seq:
            return 0.0
        return (seq.count("G") + seq.count("C")) / len(seq)

    @staticmethod
    def reverse(seq: str) -> str:
        """Reverse the sequence (sequence-order pre-training target)."""
        return seq[::-1]
