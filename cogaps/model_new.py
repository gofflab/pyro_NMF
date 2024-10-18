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
                 device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]):
            ):
        super().__init__()

        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.num_samples = num_samples
        self.device = device
        print(self.device)

        #### A_mean should be patterns x genes ####
        #### P_mean should be samples x patterns ####

        #### Both scale and loc are learned/updated parameters
        #### Should we go back to learning only scale and calculating loc?
        #### Removed all initialization method options

        # Matrix A (patterns x genes)
        #self.scale_A = PyroParam(torch.ones(num_patterns, num_genes), constraint=dist.constraints.positive)  # learnable scale
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device), constraint=dist.constraints.positive)  # loc is mean for normal
        self.scale_A = PyroParam(torch.ones(self.num_patterns, self.num_genes, device=self.device),constraint=dist.constraints.positive)    # scale is std for normal

        # Matrix P (samples x patterns)
        #self.scale_P = PyroParam(torch.ones(num_samples, num_patterns),constraint=dist.constraints.positive)  # learnable scale
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)  # loc is mean for normal
        self.scale_P = PyroParam(torch.ones(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)    # scale is std for normal


    def forward(self, D):
        # Nested plates for pixel-wise independence?
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A))

        # Nested plates for pixel-wise independence?
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P))

        # Reconstruct D as the product of P and A
        D_reconstructed = torch.matmul(P, A)  # (samples x genes) # move soft plus up here? KW TODO

        # Model the data D as being Normal distributed around the reconstructed D
        # normal? Try saving D at every step
        # compare sample reconstructed to previous step to see whats changing?
        #print("no return")
        pyro.sample("D", dist.Normal(softplus(D_reconstructed),torch.ones_like(D_reconstructed)).to_event(2), obs=D) # Check std ones KW TODO; do we want gen specific standard deviation

        #return D_reconstructed

def guide(D):
    pass

#%%

############################################################
####################### TEST RUN ###########################
############################################################

######## Data example
#samples, genes, patterns = 50, 100, 10  # Example dimensions for D, A, and P
#D = torch.randn(samples, genes).abs()  # Example data matrix (samples x genes)

######## Initialize model
#model = GammaMatrixFactorization(genes, patterns, samples)

######## Optimization setup
#optimizer = Adam({"lr": 0.01})  # Learning rate for SVI
#svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

######## Training loop
#n_steps = 10000
#for step in range(n_steps):
#    loss = svi.step(D)  # One step of SVI optimization
#    if step % 100 == 0:
#        print(f"Step {step} - Loss: {loss:.4f}")

######## After training, the optimized parameters for A and P can be accessed as:
#A_optimized = model.scale_A / model.loc_A
#P_optimized = model.scale_P / model.loc_P

