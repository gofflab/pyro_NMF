#%%
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
import torch
from .gamma_NB_base import Gamma_NegBinomial_base



#%% Enable Validations
pyro.enable_validation(True)

#%%
class Gamma_NegBinomial_L2reg(Gamma_NegBinomial_base):
    def __init__(self,
                 num_genes,
                 num_samples,
                 num_patterns,
                 NB_probs = 0.5,
                 lambda_A = 1e-4,
                 lambda_P = 1e-4,
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):
        
        super().__init__(num_genes, num_samples, num_patterns, NB_probs, device)

        self.lambda_A = lambda_A
        self.lambda_P = lambda_P


    def forward(self, D):
        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A)) # sample A from Gamma

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P)) # sample P from Gamma

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
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

