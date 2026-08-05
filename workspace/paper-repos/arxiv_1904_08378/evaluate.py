#!/usr/bin/env python
"""Dynamic Eval evaluation script."""
import argparse
import torch
from src.models.dynamic_eval_net import DynamicEvaluator
from src.utils.metrics import bits_per_character, perplexity

def evaluate(dataset, optimizer_type='rmsprop', learning_rate=0.01):
    """Evaluate with dynamic evaluation."""
    print(f"Evaluating on {dataset} with {optimizer_type}...")
    
    # Compute metrics
    # results = run_evaluation(...)
    
    print(f"Dataset: {dataset}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Learning Rate: {learning_rate}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['enwik8', 'text8', 'wikitext-103'])
    parser.add_argument('--optimizer', default='rmsprop', choices=['sgd', 'rmsprop'])
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    evaluate(args.dataset, args.optimizer, args.lr)
