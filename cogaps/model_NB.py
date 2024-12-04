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


#%% Enable Validations
pyro.enable_validation(True)

#%%
class GammaMatrixFactorization(PyroModule):
    def __init__(self,
                 num_genes,
                 num_patterns,
                 num_samples,
                 D_NegBinomial_probs = 0.5,
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]):
            ):
        super().__init__()

        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.num_samples = num_samples
        self.D_NegBinomial_probs = D_NegBinomial_probs
        self.device = device
        print(self.device)

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
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A))

        # Nested plates for pixel-wise independence?
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P))

        #### Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = softplus(torch.matmul(P, A))  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.D_NegBinomial_probs).to_event(2), obs=D) ## Changed distribution to NegativeBinomial

def guide(D):
    pass

