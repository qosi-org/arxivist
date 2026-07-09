import torch
import numpy as np

def parzen_log_likelihood(generator, test_data, z_dim=100, n_samples=10000, sigma=0.2, device="cpu"):
    """
    Gaussian Parzen window log-likelihood estimator (Section 5, paper's evaluation method).
    ASSUMED sigma is cross-validated on a held-out validation set (confidence 0.6) —
    paper states this procedure but exact per-dataset sigma values are not given in text.

    High variance estimator — treat results as directional, not exact (per paper's own caveat).
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        samples = generator(z)  # [n_samples, D]

    log_likelihoods = []
    for x in test_data:
        x = x.to(device)
        # squared distance from x to each generated sample
        diffs = samples - x.unsqueeze(0)
        sq_dists = (diffs ** 2).sum(dim=1)
        log_kernel = -sq_dists / (2 * sigma ** 2)
        # log-sum-exp for numerical stability
        max_log = log_kernel.max()
        log_density = max_log + torch.log(torch.exp(log_kernel - max_log).mean())
        log_likelihoods.append(log_density.item())

    return {
        "mean_log_likelihood": float(np.mean(log_likelihoods)),
        "std_error": float(np.std(log_likelihoods) / np.sqrt(len(log_likelihoods))),
        "sigma_used": sigma
    }