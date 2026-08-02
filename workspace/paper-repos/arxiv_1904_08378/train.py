#!/usr/bin/env python
"""Dynamic Eval training (adapter setup) script."""
import argparse
import yaml
import torch
from src.models.dynamic_eval_net import DynamicEvaluator

def setup(config_path):
    """Setup pretrained model for dynamic evaluation."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load pretrained Transformer-XL
    print(f"Loading {config['model']['pretrained']}...")
    # model = load_pretrained(config['model']['pretrained'])
    
    evaluator = DynamicEvaluator(
        model=None,  # Will be loaded
        optimizer_type=config['adaptation']['optimizer'],
        learning_rate=config['adaptation']['learning_rate']
    )
    
    print("Setup complete. Ready for dynamic evaluation on test sets.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    setup(args.config)
