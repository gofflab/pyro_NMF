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

def generate_structured_test_data(num_genes, num_samples, num_patterns, noise_level=0.1, sparsity=0.2):
    # Generate patterns matrix with clear linear relationships
    pattern_matrix = torch.zeros(num_genes, num_patterns)
    for i in range(num_patterns):
        pattern_matrix[i * (num_genes // num_patterns):(i + 1) * (num_genes // num_patterns), i] = 1
    
    # Generate loadings matrix with block structure
    loading_matrix = torch.zeros(num_patterns, num_samples)
    for i in range(num_patterns):
        loading_matrix[i, i * (num_samples // num_patterns):(i + 1) * (num_samples // num_patterns)] = 1
    
    # Generate synthetic data using Poisson distribution
    true_data = torch.matmul(pattern_matrix, loading_matrix)
    
    # Add Gaussian noise to the data
    noise = torch.randn_like(true_data) * noise_level
    noisy_data = true_data + noise
    
    return noisy_data, pattern_matrix, loading_matrix