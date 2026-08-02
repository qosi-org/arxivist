# ELMo: Deep Contextualized Word Representations

**Paper:** Peters et al. (2018)  
**arXiv ID:** 1802.05365  
**Venue:** ICLR 2018

## Quick Start

```bash
# Install dependencies
pip install torch pyyaml

# Train
python train.py --config configs/elmo_1b_word.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/model.pt --dataset 1b_word

# Inference
python inference.py --checkpoint checkpoints/model.pt --text "hello world"
```

## Directory Structure

```
.
├── configs/              # Training configs
├── src/
│   ├── models/          # ELMo model architecture
│   └── utils/           # Utilities (max norm constraint)
├── Results&Comparison/  # Benchmarks and reports
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── inference.py         # Inference script
```

## Paper Benchmarks (Target)

- SQuAD F1: 85.8
- SNLI Accuracy: 88.7
- SRL F1: 84.6
- Coreference F1: 70.4
- NER F1: 92.22
- SST-5 Accuracy: 54.7
