#!/usr/bin/env python
"""ELMo training script."""
import argparse
import yaml
import torch
import torch.nn as nn
from src.models.elmo_net import ELMo
from src.utils.max_norm import apply_max_norm

def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = ELMo(
        vocab_size=config['model']['char_vocab_size'],
        char_embed_dim=config['model']['char_embed_dim'],
        num_filters=config['model']['num_filters'],
        projection_dim=config['model']['projection_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_bilm_layers']
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['learning_rate'])
    
    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        apply_max_norm(model, 15.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    args = parser.parse_args()
    train(args.config)
