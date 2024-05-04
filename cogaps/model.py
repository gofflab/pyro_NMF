import torch
import pyro
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam


#%% Enable Validations
pyro.enable_validation(True)

#%%
# Deprecated
# def cogaps_model(D, num_patterns):
#     num_genes, num_samples = D.shape
#     #print(f"num_genes: {num_genes}\nnum_samples: {num_samples}\nnum_patterns: {num_patterns}")
#     # Parameters should be defined outside the plate if they do not depend on it
#     A_scale = pyro.param("A_scale", torch.ones((num_genes, num_patterns)), constraint=dist.constraints.positive)
#     #print(A_scale.shape)
#     P_scale = pyro.param("P_scale", torch.ones((num_patterns, num_samples)), constraint=dist.constraints.positive)

#     with pyro.plate("genes", num_genes):
#         #print(dist.Exponential(rate=A_scale))
#         A = pyro.sample("A", dist.Exponential(rate=A_scale).to_event(1))

#     with pyro.plate("patterns", num_patterns):
#         P = pyro.sample("P", dist.Exponential(rate=P_scale).to_event(1))

#     prediction = torch.matmul(A, P)
    
#     with pyro.plate("data", num_genes):
#         #pyro.sample("obs", dist.Poisson(prediction), obs=D.T)
#         # Use Negative Binomial distribution instead of Poisson
#         #pyro.sample("obs", dist.NegativeBinomial(total_count=1, logits=prediction), obs=D.T)
#         # Use Gaussian (normal) distribution instead of Poisson or Negative Binomial
#         pyro.sample("obs", dist.Normal(prediction, torch.ones_like(prediction)).to_event(1), obs=D)

class CoGAPSModel(PyroModule):
    def __init__(self, D, num_patterns, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_genes = D.shape[0]
        self.num_samples = D.shape[1]
        self.D_gene_means = D.mean(axis=1)
        self.D_mean = D.mean()
        self.num_patterns = num_patterns

        # Initialize scale parameters to 10% of each gene's mean
        self.initial_scale = 0.1 * self.D_gene_means
        self.init_vals = self.initial_scale.unsqueeze(-1)
        self.init_vals = self.init_vals.expand(-1, self.num_patterns)
        
        # Define parameters
        #self.A_scale = PyroParam(torch.full((self.num_genes, self.num_patterns), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
        self.A_scale = PyroParam(self.init_vals.to(device), constraint=dist.constraints.positive)
        #self.P_scale = PyroParam(torch.full((self.num_patterns, self.num_samples), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
        self.P_scale = PyroParam(torch.ones((self.num_patterns, self.num_samples), device=self.device), constraint=dist.constraints.positive)
        
    def forward(self, D):
        with pyro.plate("genes", self.num_genes):
            A = pyro.sample("A", dist.Exponential(rate=self.A_scale).to_event(1))

        with pyro.plate("patterns", self.num_patterns):
            P = pyro.sample("P", dist.Exponential(rate=self.P_scale).to_event(1))

        prediction = torch.matmul(A, P)

        with pyro.plate("data", self.num_genes):
            pyro.sample("obs", dist.Normal(prediction, torch.ones_like(prediction)).to_event(1), obs=D)
