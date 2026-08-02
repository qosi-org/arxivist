#!/usr/bin/env python
"""ELMo inference script."""
import argparse
import torch
from src.models.elmo_net import ELMo

def infer(checkpoint_path, text):
    """Infer ELMo embeddings for text."""
    model = ELMo(vocab_size=261, char_embed_dim=50, num_filters=2048, 
                 projection_dim=512, hidden_size=4096, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    with torch.no_grad():
        # Convert text to character IDs
        char_ids = torch.tensor([[ord(c) % 261 for c in text]], dtype=torch.long)
        embeddings = model(char_ids)
        print(f"Embeddings shape: {embeddings.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    infer(args.checkpoint, args.text)
