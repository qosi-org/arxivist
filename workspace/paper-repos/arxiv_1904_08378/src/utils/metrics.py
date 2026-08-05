import math

def bits_per_character(cross_entropy_loss):
    """Convert CE loss to bits per character."""
    return cross_entropy_loss / math.log(2)

def perplexity(cross_entropy_loss):
    """Convert CE loss to perplexity."""
    return math.exp(cross_entropy_loss)
