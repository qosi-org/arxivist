#!/usr/bin/env python
"""Dynamic Eval inference script."""
import argparse
import torch
from src.models.dynamic_eval_net import DynamicEvaluator

def infer(checkpoint_path, text, optimizer_type='rmsprop'):
    """Run dynamic evaluation on text."""
    print(f"Loading checkpoint: {checkpoint_path}")
    # model = load_model(checkpoint_path)
    
    evaluator = DynamicEvaluator(model=None, optimizer_type=optimizer_type)
    print(f"Computing perplexity with {optimizer_type} adaptation...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--optimizer', default='rmsprop')
    args = parser.parse_args()
    infer(args.checkpoint, args.text, args.optimizer)
