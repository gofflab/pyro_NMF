import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def initialize_inducing_points_with_pca(expression_data, spatial_coords, num_components, num_inducing_points):
    # Standardize the expression data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(expression_data)

    # Perform PCA on the standardized expression data
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(data_standardized)

    # Select indices with the highest absolute scores in the first principal component
    extreme_indices = np.argsort(np.abs(principal_components[:, 0]))[-num_inducing_points:]

    # Map these indices to select corresponding spatial coordinates
    inducing_points = torch.tensor(spatial_coords[extreme_indices], dtype=torch.float)

    return inducing_points

def initialize_inducing_points_with_pca(expression_data, spatial_coords, num_components, num_inducing_points):
    # Standardize the expression data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(expression_data)

    # Perform PCA on the standardized expression data
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(data_standardized)

    # Select indices with the highest absolute scores in the first principal component
    extreme_indices = np.argsort(np.abs(principal_components[:, 0]))[-num_inducing_points:]

    # Select inducing points based on both expression data and spatial coordinates
    inducing_points_expression = torch.tensor(data_standardized[extreme_indices], dtype=torch.float)
    inducing_points_spatial = torch.tensor(spatial_coords[extreme_indices], dtype=torch.float)

    # Concatenate inducing points from expression data and spatial coordinates
    inducing_points = torch.cat((inducing_points_expression, inducing_points_spatial), dim=1)

    return inducing_points
