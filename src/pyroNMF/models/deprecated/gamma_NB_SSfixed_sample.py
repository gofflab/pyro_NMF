"""Deprecated semi-supervised Gamma-NB model with sampling (kept for reference)."""
#%%
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
import torch
import matplotlib.pyplot as plt
from .gamma_NB_base import Gamma_NegBinomial_base

#%% Enable Validations
pyro.enable_validation(True)

#%%
class Gamma_NegBinomial_SSFixed(Gamma_NegBinomial_base):
    """Deprecated Gamma-Negative Binomial model with fixed patterns."""
    def __init__(self,
                 num_genes,
                 num_samples,
                 num_patterns, # these are additional unsupervised patterns
                 fixed_patterns,
                 NB_probs = 0.5,
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]):
            ):
        """Initialize the deprecated semi-supervised model.

        Parameters
        ----------
        num_genes : int
            Number of genes/features (columns in ``D``).
        num_samples : int
            Number of samples (rows in ``D``).
        num_patterns : int
            Number of additional patterns to learn.
        fixed_patterns : array-like
            Fixed patterns with shape ``(num_samples, num_fixed_patterns)``.
        NB_probs : float, optional
            Negative Binomial probability parameter.
        device : torch.device, optional
            Device for parameters and tensors.
        """
        super().__init__(num_genes, num_samples, num_patterns, NB_probs, device)
        self.fixed_patterns = fixed_patterns # of shape samples x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]

        #### Matrix A is patterns (supervised+unsupervised) x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device), constraint=dist.constraints.positive)
        self.scale_A = PyroParam(torch.ones(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device),constraint=dist.constraints.positive)
        
        print(f"Using {self.device}")

        #### Fixed patterns are samples x patterns ####
        self.fixed_P = torch.tensor(fixed_patterns, device=self.device,dtype=torch.float32) # tensor, not updatable

        #### Save samples ####
        self.sumA = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device)
        self.sumP = torch.zeros(self.num_samples, self.num_patterns, device=self.device)

    def forward(self, D, samp = False):
        """Run a forward pass of the model.

        Parameters
        ----------
        D : torch.Tensor
            Observed count matrix with shape ``(num_samples, num_genes)``.
        samp : bool, optional
            If True, accumulate sampled A/P values into running sums.
        """
        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_fixed_patterns + self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A)) # sample A from Gamma
                self.A = A

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P)) # sample P from Gamma
                self.P = P

        P_total = torch.cat((self.fixed_P, P), dim=1)
        self.P_total = P_total # save P_total

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P_total, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

        if samp:
            self.sumA += A
            self.sumP += P 

        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) ## Changed distribution to NegativeBinomial


    
    def plot_grid_noAlpha(self, patterns, coords, nrows, ncols, s=4, savename = None):
        """Plot spatial patterns without alpha scaling.

        Parameters
        ----------
        patterns : array-like
            Pattern matrix with shape ``(n_samples, n_patterns)``.
        coords : Mapping or array-like
            Spatial coordinates; expects ``coords['x']`` and ``coords['y']``.
        nrows : int
            Number of rows in the plot grid.
        ncols : int
            Number of columns in the plot grid.
        s : float, optional
            Marker size for scatter points.
        savename : str or None, optional
            If provided, save the figure to this path.
        """
        fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*5, nrows*4))
        num_patterns = patterns.shape[1]
        x, y = coords['x'], coords['y']
        i = 0
        for r in range(nrows):
            for c in range(ncols):
                if i < num_patterns:
                    pattern_min = patterns[:,i].min()
                    pattern_max = patterns[:,i].max()
                    p = axes[r,c].scatter(x, y, c=patterns[:,i], s=s, alpha=1, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                    axes[r,c].set_yticklabels([])
                    axes[r,c].set_xticklabels([])
                    fig.colorbar(p, ax=axes[r,c])

                    #axes[r,c].set_title(patterns.columns[i])
                    i += 1
        if savename != None:
            plt.savefig(savename)

def guide(D):
    """Placeholder guide (not implemented)."""
    pass
