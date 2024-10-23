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


        #### Matrix A is patterns x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device), constraint=dist.constraints.positive)
        self.scale_A = PyroParam(torch.ones(self.num_patterns, self.num_genes, device=self.device),constraint=dist.constraints.positive)

        #### Matrix P is samples x patterns ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)
        self.scale_P = PyroParam(torch.ones(self.num_samples, self.num_patterns, device=self.device),constraint=dist.constraints.positive)

        #### scale D as a learnable parameter ###
        #self.scale_D = PyroParam(torch.ones(self.num_samples, self.num_genes, device=self.device), constraint=dist.constraints.positive) # no constraints blows up values
        self.scale_D = PyroParam(torch.ones(self.num_samples, self.num_genes, device=self.device), constraint=dist.constraints.interval(0.1, 18)) # KW TODO: how to determine constraints? rn hardcoded as max SD of a gene from input data

        #### Store D reconstructed and D sampled ? ####
        #self.D_reconstructed = torch.ones(self.num_samples, self.num_genes, device=self.device)
        #self.D_sampled = torch.ones(self.num_samples, self.num_genes, device=self.device)

    def forward(self, D):
        # Nested plates for pixel-wise independence?
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale_A))

        # Nested plates for pixel-wise independence?
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale_P))

        #### Matrix D_reconstucted is samples x genes
        # Calculate as the product of P and A
        D_reconstructed = softplus(torch.matmul(P, A))  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed

        #### How should we sample D? ####

        ## Try using gene std
        #std_D_per_gene = torch.std(D_reconstructed, dim=0, keepdim=True)  # (1 x genes)
        #std_D_broadcasted = torch.clamp(std_D_per_gene.expand(self.num_samples, -1), min=0.5)  # (-1 preserves the size of the dim) so now samples x genes
        #pyro.sample("D", dist.Normal(D_reconstructed, std_D_broadcasted).to_event(2), obs=D)

        ## Try original method, use scale of ones
        #pyro.sample("D", dist.Normal(D_reconstructed,torch.ones_like(D_reconstructed)).to_event(2), obs=D)
        #D_sampled = pyro.sample("D", dist.Normal(D_reconstructed,torch.ones_like(D_reconstructed)).to_event(2), obs=D)

        ## Try making scale_D learnable
        pyro.sample("D", dist.GammaPoisson(D_reconstructed, self.scale_D).to_event(2), obs=D) ## Changed distribution to GammaPoisson
        #pyro.sample("D", dist.Normal(D_reconstructed,self.scale_D).to_event(2), obs=D)

        ## Try using percentage of expression as uncertainty
        #uncertainty = (D * 0.1) + 1
        #pyro.sample("D", dist.Normal(D_reconstructed, uncertainty).to_event(2), obs=D)

        ## Options to save or return
        #D_sampled = pyro.sample("D", dist.Normal(D_reconstructed,torch.ones_like(D_reconstructed)).to_event(2), obs=D) # Check std ones KW TODO; do we want gen specific standard deviation
        #self.D_sampled = D_sampled
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

