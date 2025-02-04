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
class SSGammaMatrixFactorization(PyroModule):
    def __init__(self,
                 num_genes,
                 num_samples,
                 num_patterns, # number of additional unsupervised patterns
                 fixed_patterns, # samples x fixed_patterns
                 NB_probs = 0.5,
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]):
            ):
        super().__init__()

        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.fixed_patterns = fixed_patterns # of shape samples x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]
        self.num_samples = num_samples
        self.NB_probs = NB_probs
        self.device = device

        print(f"Using {self.device}")

        #### Matrix A is patterns x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns + self.num_fixed_patterns, self.num_genes, device=self.device), constraint=dist.constraints.positive)
        self.scale_A = PyroParam(torch.ones(self.num_patterns + self.num_fixed_patterns, self.num_genes, device=self.device),constraint=dist.constraints.positive)

        #### Matrix P is samples x patterns ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)
        self.scale_P = PyroParam(torch.ones(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)

        self.fixed_P = torch.tensor(fixed_patterns, device=self.device,dtype=torch.float32)


    def forward(self, D):
        # Nested plates for pixel-wise independence?
        with pyro.plate("patterns", self.num_patterns + self.num_fixed_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A)) # sample A from Gamma

        # Nested plates for pixel-wise independence?
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P)) # sample P from Gamma
                
        P_total = torch.cat((self.fixed_P, P), dim=1) # concatenate fixed_patterns to learned patterns
        self.P_total = P_total # save P_total

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P_total, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

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
                #p5 = np.percentile(patterns[:,i], 5)
                #p95 = np.percentile(patterns[:,i], 95)
                pattern_min = patterns[:,i].min()
                pattern_max = patterns[:,i].max()
                #pattern_min = np.min(patterns[:, i])
                #pattern_max = np.max(patterns[:, i])
                alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))
                #axes[r,c].scatter(x, y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                p = axes[r,c].scatter(x, y, c=patterns[:,i], s=4,alpha=alpha_values, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                #axes[r,c].set_title(patterns.columns[i])
                i += 1

    if savename != None:
        plt.savefig(savename)


'''
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
                pattern_min = patterns[:,i].min()
                pattern_max = patterns[:,i].max()
                #pattern_min = np.min(patterns[:, i])
                #pattern_max = np.max(patterns[:, i])

                #alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))

                #axes[r,c].scatter(x, y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                p = axes[r,c].scatter(x, y, c=patterns[:,i], s=4,alpha=alpha_values, vmin=p5, vmax=max(1,p95), cmap='viridis',edgecolors='none')
                cbar = fig.colorbar(scatter, ax=axes[r, c], shrink=0.7)  # Adjust shrink to fit the grid nicely

                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                #axes[r,c].set_title(patterns.columns[i])
                i += 1

    if savename != None:
        plt.savefig(savename)
'''

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

# %%
