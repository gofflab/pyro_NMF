"""Deprecated Gamma-NB model with chi-squared tracking (kept for reference)."""
#%%
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
import torch
import matplotlib.pyplot as plt

#%% Enable Validations
pyro.enable_validation(True)

#%%
class Gamma_NegBinomial_base(PyroModule):
    """Deprecated Gamma-Negative Binomial model with chi-squared tracking."""
    def __init__(self,
                 num_genes,
                 num_samples,
                 num_patterns,
                 use_chisq = False,
                 NB_probs = 0.5,
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):
        """Initialize the deprecated model.

        Parameters
        ----------
        num_genes : int
            Number of genes/features (columns in ``D``).
        num_samples : int
            Number of samples (rows in ``D``).
        num_patterns : int
            Number of latent patterns to learn.
        use_chisq : bool, optional
            If True, include chi-squared loss term.
        NB_probs : float, optional
            Negative Binomial probability parameter.
        device : torch.device, optional
            Device for parameters and tensors.
        """
        super().__init__()

        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.num_samples = num_samples
        self.NB_probs = NB_probs
        self.device = device
        self.use_chisq = use_chisq
        self.best_chisq = None
        self.best_chisq_iter = 0

        print(f"Using {self.device}")
        print("Original")
        #### Matrix A is patterns x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device), constraint=dist.constraints.positive)
        self.scale_A = PyroParam(torch.ones(self.num_patterns, self.num_genes, device=self.device),constraint=dist.constraints.positive)
        
        #### Matrix P is samples x patterns ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)
        self.scale_P = PyroParam(torch.ones(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)
    
        #### Save samples ####
        self.sumA = torch.zeros(self.num_patterns, self.num_genes, device=self.device)
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
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A)) # sample A from Gamma
        self.A = A
        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P)) # sample P from Gamma
        self.P = P

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

        chi2 = torch.sum(((D_reconstructed - D))**2)
        self.chi2  = chi2
        self.best_chisq_iter += 1
        
        if self.best_chisq == None: # first chi squared saved
            self.best_chisq = chi2
        elif chi2 < self.best_chisq:
            self.best_chisq = chi2
            self.best_A = A
            self.best_P = P

        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        # Save sampled values
        if samp:
            self.sumA += A
            self.sumP += P 

        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) ## Changed distribution to NegativeBinomial
        
        

    #%%
    def guide(D):
        """Placeholder guide (not implemented)."""
        pass


    def plot_grid(self, patterns, coords, nrows, ncols, s=4, savename = None):
        """Plot spatial patterns with alpha scaling.

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
                    #p5 = np.percentile(patterns[:,i], 5)
                    #p95 = np.percentile(patterns[:,i], 95)
                    pattern_min = patterns[:,i].min()
                    pattern_max = patterns[:,i].max()
                    #pattern_min = np.min(patterns[:, i])
                    #pattern_max = np.max(patterns[:, i])
                    alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))
                    #axes[r,c].scatter(x, y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                    p = axes[r,c].scatter(x, y, c=patterns[:,i], s=s, alpha=alpha_values, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                    axes[r,c].set_yticklabels([])
                    axes[r,c].set_xticklabels([])
                    fig.colorbar(p, ax=axes[r,c])

                    #axes[r,c].set_title(patterns.columns[i])
                    i += 1

        if savename != None:
            plt.savefig(savename)


    
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
# %%

# %%
