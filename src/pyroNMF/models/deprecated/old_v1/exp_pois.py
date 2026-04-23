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
class ExpMatrixFactorization(PyroModule):
    def __init__(self,
                num_samples,
                 num_genes,
                 num_patterns,
                 sparsity,
                 use_chisq = False,
                 device=torch.device('cpu')
            ):
        super().__init__()

        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.num_samples = num_samples
        self.sparsity = sparsity
        self.device = device

        print(f"Using {self.device}")

        #self.minchi2 = 1000000000000

        self.best_chisq = np.inf
        self.best_chisq_iter = 0
        self.iter = 0

        ## A is patterns x genes
        ## P is samples x patterns
        self.sumA = torch.zeros(self.num_patterns, self.num_genes, device=self.device)
        self.sumP = torch.zeros(self.num_samples, self.num_patterns, device=self.device)

        self.sumA2 = torch.zeros(self.num_patterns, self.num_genes, device=self.device)
        self.sumP2 = torch.zeros(self.num_samples, self.num_patterns, device=self.device)

        self.best_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device)
        self.best_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device)


        #### A parameter for Exponential to Populate A Matrix ####
        self.scale_A = PyroParam(torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)

        #### P parameter for Exponential to Populate P Matrix ####
        self.scale_P = PyroParam(torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)

    # forward when no uncertainty matrix, no chi2 calculation done
    def forward(self, D, U):
        self.iter += 1

        # Nested plates 
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Exponential(self.scale_A)) # sample A from Gamma
                self.A = A

        # Nested plates 
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Exponential(self.scale_P)) # sample P from Gamma
                self.P = P

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
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
            #self.best_locA = self.loc_A
            #self.best_locP = self.loc_P

        # Error Model Poisson
        theta = self.D_reconstructed
        poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
   
        # Addition to Elbow Loss - should make this at least as large as Elbow
        pyro.factor("pois.loss",10.*poisL)

        #if samp:
            # try to normalize by pattern max, here in column
        with torch.no_grad():
            correction = P.max(axis=0).values
            Pn = P / correction
            An = A * correction.unsqueeze(1)

            self.sumA += An
            self.sumP += Pn
            self.sumA2 += torch.square(An)
            self.sumP2 += torch.square(Pn) 
    
        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.sparsity).to_event(2),obs=D)

    #def loss_chi2(self, D, U):
    #    chi2 = torch.sum(((self.D_reconstructed - D)/U)**2)
    #    return chi2/2

    #def pois_Loss(self, D, U):
    #    theta = self.D_reconstructed
    #    pLoss = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
    #    return -pLoss
        
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
