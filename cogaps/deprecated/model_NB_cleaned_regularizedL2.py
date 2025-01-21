#%%
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
from torch import nn
from torch.nn.functional import softplus
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad



#%% Enable Validations
pyro.enable_validation(True)

#%%
class GammaMatrixFactorization(PyroModule):
    def __init__(self,
                 num_genes,
                 num_patterns,
                 num_samples,
                 NB_probs = 0.5,
                 lambda_A = 1e-4,
                 lambda_P = 1e-4,
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]):
            ):
        super().__init__()

        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.num_samples = num_samples
        self.NB_probs = NB_probs
        self.lambda_A = lambda_A
        self.lambda_P = lambda_P
        self.device = device

        print(f"Using {self.device}")

        #### Matrix A is patterns x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device), constraint=dist.constraints.positive)
        self.scale_A = PyroParam(torch.ones(self.num_patterns, self.num_genes, device=self.device),constraint=dist.constraints.positive)

        #### Matrix P is samples x patterns ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)
        self.scale_P = PyroParam(torch.ones(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)

    def forward(self, D):
        # Nested plates for pixel-wise independence?
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A)) # sample A from Gamma

        # Nested plates for pixel-wise independence?
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P)) # sample P from Gamma

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        #D_reconstructed = softplus(torch.matmul(P, A))  # (samples x genes)
        D_reconstructed = torch.matmul(P, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

        # Add regularization terms using pyro.factor
        reg_A = torch.sum(self.loc_A ** 2)  # L2 Regularization for A
        reg_P = torch.sum(self.loc_P ** 2)  # L2 Regularization for P
        pyro.factor("A_regularization", -self.lambda_A * reg_A)
        pyro.factor("P_regularization", -self.lambda_P * reg_P)

        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) ## Changed distribution to NegativeBinomial

def guide(D):
    pass


#%% Plotting function

# make sure to change CCF coords -1*y

def plot_grid(patterns, coords, nrows, ncols, savename = None):
    fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*4, nrows*4))
    num_patterns = patterns.shape[1]
    x, y = coords['x'], coords['y']
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                p5 = np.percentile(patterns[:,i], 5)
                p95 = np.percentile(patterns[:,i], 95)

                #pattern_min = np.min(patterns[:, i])
                #pattern_max = np.max(patterns[:, i])
                #alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))
                #axes[r,c].scatter(x, y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                p = axes[r,c].scatter(x, y, c=patterns[:,i], s=6,alpha=0.7, vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                i += 1

    if savename != None:
        plt.savefig(savename)

def plot_grid_NSF(patterns, coords, nrows, ncols, savename = None):
    fig, axes = plt.subplots(nrows,ncols)
    num_patterns = patterns.shape[1]
    x, y = coords['x'], coords['y']
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    resolution = 0.114286
    x_grid = np.arange(x_min, x_max + resolution, resolution)
    y_grid = np.arange(y_min, y_max + resolution, resolution)

    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                # Initialize an empty grid (NaN-filled)
                grid = np.full((len(y_grid), len(x_grid)), np.nan)
                # Map the patterns to the grid; find nearest grid indices for each x, y pair
                ix = np.searchsorted(x_grid, x) - 1
                iy = np.searchsorted(y_grid, y) - 1
                for j in range(len(patterns[:,i])):
                    if 0 <= ix[j] < len(x_grid) and 0 <= iy[j] < len(y_grid):
                        grid[iy[j], ix[j]] = patterns[j,i]
                axes[r,c].imshow(grid, cmap='viridis')
                i += 1
    if savename != None:
        plt.savefig(savename)


def plot_correlations(true_vals, inferred_vals, savename = None):
    true_vals_df = pd.DataFrame(true_vals)
    true_vals_df.columns = ['True_' + str(x) for x in true_vals_df.columns]
    inferred_vals_df = pd.DataFrame(inferred_vals)
    inferred_vals_df.columns = ['Inferred_' + str(x) for x in inferred_vals_df.columns]
    correlations = true_vals_df.merge(inferred_vals_df, left_index=True, right_index=True).corr().round(2)
    plt.figure()
    sns.clustermap(correlations.iloc[:true_vals.shape[1], true_vals.shape[1]:], annot=False, cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
    if savename != None:
        plt.savefig(savename + '_correlations.png')


def prep_anndata(ad_data, counts_layer = None):
        if not(counts_layer):
            print("Using raw counts from anndata.X")
            D = torch.tensor(ad_data.X) ## RAW COUNT DATA
        else:
            print(f"Retrieve raw counts from {counts_layer} layer NOT YET IMPLEMENTED")
