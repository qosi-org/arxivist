#!/usr/bin/env python
"""ELMo evaluation script."""
import argparse
import torch
from src.models.elmo_net import ELMo

def evaluate(checkpoint_path, dataset):
    """Evaluate model on dataset."""
    model = ELMo(vocab_size=261, char_embed_dim=50, num_filters=2048, 
                 projection_dim=512, hidden_size=4096, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Compute perplexity on dataset
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pass  # Load data and evaluate
    
    perplexity = torch.exp(torch.tensor(total_loss / num_batches))
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.dataset)
