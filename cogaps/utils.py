import torch

def generate_test_data(num_genes, num_samples, num_patterns, noise_level=0.1):
    # Generate random patterns and loadings with positive values
    A_true = torch.abs(torch.randn(num_genes, num_patterns))
    P_true = torch.abs(torch.randn(num_patterns, num_samples))
    
    # Generate synthetic data using Poisson distribution
    D = torch.matmul(A_true, P_true)
    
    # Add Gaussian noise to the data
    noise = torch.randn_like(D) * noise_level
    D_noisy = D + noise
    
    return D_noisy, A_true, P_true
