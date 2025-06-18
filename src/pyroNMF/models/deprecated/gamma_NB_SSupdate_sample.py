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
class Gamma_NegBinomial_SSUpdate(Gamma_NegBinomial_base):
    def __init__(self,
                 num_genes,
                 num_samples,
                 num_patterns, # these are additional unsupervised patterns
                 fixed_patterns,
                 NB_probs = 0.5,
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]):
            ):
        
        super().__init__(num_genes, num_samples, num_patterns, NB_probs, device)
        
        self.num_fixed_patterns = fixed_patterns.shape[1]

        #### Matrix A is patterns (supervised+unsupervised) x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device), constraint=dist.constraints.positive)
        self.scale_A = PyroParam(torch.ones(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device),constraint=dist.constraints.positive)

        self.fixed_patterns = fixed_patterns # of shape samples x fixed patterns

        print(f"Using {self.device}")

        #### Fixed patterns are samples x patterns ####
        self.fixed_P = PyroParam(
                torch.tensor(fixed_patterns, device=self.device, dtype=torch.float32),
                constraint=dist.constraints.positive
            ) # PyroParam is updatable
        
        #### Save samples ####
        self.sumA = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device)
        self.sumP = torch.zeros(self.num_samples, self.num_patterns, device=self.device)

        
    def forward(self, D,samp=False):
        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_fixed_patterns + self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A)) # sample A from Gamma

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P)) # sample P from Gamma
        
        P_total = torch.cat((self.fixed_P, P), dim=1)
        self.P_total = P_total # save P_total

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P_total, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

        if samp:
            self.sumA += A
            self.sumP += P 

        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) ## Changed distribution to NegativeBinomial

def guide(D):
    pass

