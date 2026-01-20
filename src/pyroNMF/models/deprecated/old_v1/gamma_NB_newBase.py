#%%
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
import torch
import matplotlib.pyplot as plt
import numpy as np

#%% Enable Validations
pyro.enable_validation(True)

#%%
class Gamma_NegBinomial_base(PyroModule):
    def __init__(self,
                num_samples,
                num_genes,
                num_patterns,
                use_chisq = False,
                scale = 1,
                NB_probs = 0.5,
                device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):
        
        super().__init__()

        ## Initialize parameters
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.use_chisq = use_chisq
        self.scale = scale
        self.use_chisq = use_chisq
        self.NB_probs = NB_probs
        self.device = device

        ## Print settings
        print(f"Using {self.device}")
        print(f"Data is {self.num_samples} samples x {self.num_genes} genes")
        print(f"Running for {self.num_patterns} patterns")
        print(f"Using scale of {self.scale} for the gamma distribution")
        print(f"Using Negative Binomial with probs of {self.NB_probs}")

        if use_chisq:
            print(f"Using chi squared")
        else:
            print(f"Not using chi squared")

        ## Set some initial values
        self.best_chisq = np.inf
        self.best_chisq_iter = 0
        self.iter = 0

        #### Matrix A is patterns x genes ####
        #### Initialize randomly, but with positive constraint ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device, dtype=torch.float32), constraint=dist.constraints.positive)
        
        #### Matrix P is samples x patterns ####
        #### Initialize randomly, but with positive constraint ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device, dtype=torch.float32),constraint=dist.constraints.positive)

        #### Single fixed scale parameter for gammas ####
        self.scale = scale

        self.best_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=torch.float32)
        self.best_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=torch.float32)
        self.best_locA = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=torch.float32)
        self.best_locP = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=torch.float32)

        #self.A = torch.ones(self.num_patterns, self.num_genes, device=self.device, dtype=torch.float32)
        #self.P = torch.ones(self.num_samples, self.num_patterns, device=self.device, dtype=torch.float32)
        self.A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=torch.float32) 
        self.P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=torch.float32)

    def forward(self, D, U):

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale)) # sample A from Gamma
        self.A = A

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale)) # sample P from Gamma
        self.P = P

        # D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

        # Calculate chi squared
        chi2 = torch.sum((D_reconstructed-D)**2/U**2)
        self.chi2  = chi2

        if chi2 < self.best_chisq: # if this is a better chi squared, save it
            self.best_chisq = chi2
            self.best_chisq_iter = self.iter
            self.best_A = A
            self.best_P = P
            self.best_locA = self.loc_A
            self.best_locP = self.loc_P

        # Include chi squared loss in the model
        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) 
        

    #%%
    def guide(D):
        pass


    def plot_grid(self, patterns, coords, nrows, ncols, savename = None):
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
                    alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))
                    #axes[r,c].scatter(x, y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                    p = axes[r,c].scatter(x, y, c=patterns[:,i], s=4,alpha=alpha_values, vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                    axes[r,c].set_yticklabels([])
                    axes[r,c].set_xticklabels([])
                    #axes[r,c].set_title(patterns.columns[i])
                    i += 1

        if savename != None:
            plt.savefig(savename)
# %%
