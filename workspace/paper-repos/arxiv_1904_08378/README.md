# Dynamic Evaluation of Transformer Language Models

**Paper:** Krause et al. (2019)  
**arXiv ID:** 1904.08378

## Quick Start

```bash
# Install dependencies
pip install torch pyyaml transformers

# Setup (load pretrained Transformer-XL)
python train.py --config configs/dynamic_eval_rmsprop.yaml

# Evaluate
python evaluate.py --dataset enwik8 --optimizer rmsprop --lr 0.01

# Inference
python inference.py --checkpoint checkpoints/model.pt --text "sample text" --optimizer rmsprop
```

## Directory Structure

```
.
├── configs/              # Optimizer and model configs
├── src/
│   ├── models/          # Dynamic evaluator architecture
│   └── utils/           # Metrics (bpc, perplexity)
├── Results&Comparison/  # Benchmarks and reports
├── train.py             # Setup script
├── evaluate.py          # Evaluation script
└── inference.py         # Inference script
```

## Paper Benchmarks (Target)

| Dataset | Metric | Static | Dynamic | Improvement |
|---------|--------|--------|---------|-------------|
| enwik8 | bpc | 0.993 | 0.94 | -5.3% |
| text8 | bpc | 1.085 | 1.038 | -4.3% |
| WikiText-103 | ppl | 18.3 | 16.4 | -9.4% |
