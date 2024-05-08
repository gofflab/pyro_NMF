import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import lkj
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
from torch import nn
from torch.nn.functional import softplus

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
        #self.A_scale = PyroParam(self.init_vals.to(device), constraint=dist.constraints.positive)
        #self.P_scale = PyroParam(torch.full((self.num_patterns, self.num_samples), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
        #self.P_scale = PyroParam(torch.ones((self.num_patterns, self.num_samples), device=self.device), constraint=dist.constraints.positive)
        self.A_scale = PyroParam(torch.tensor(1.0, device=device), constraint=dist.constraints.positive)
        self.P_scale = PyroParam(torch.tensor(1.0, device=device), constraint=dist.constraints.positive)

    def forward(self, D):
        with pyro.plate("genes", self.num_genes):
            A = pyro.sample("A", dist.Exponential(rate=self.A_scale).expand([self.num_genes, self.num_patterns]).to_event(1))

        with pyro.plate("patterns", self.num_patterns):
            P = pyro.sample("P", dist.Exponential(rate=self.P_scale).expand([self.num_patterns, self.num_samples]).to_event(1))

        prediction = torch.matmul(A, P)

        with pyro.plate("data", self.num_genes):
            pyro.sample("obs", dist.Normal(prediction, torch.ones_like(prediction)).to_event(1), obs=D)

        return prediction

# Starting below from scratch
#%%
class ProbNMFModel(PyroModule):
    def __init__(self, D, num_patterns, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_genes = D.shape[0]
        self.num_samples = D.shape[1]
        self.num_patterns = num_patterns
        self.D_gene_means = D.mean(axis=1)
        #self.D_mean = D.mean()
        self.num_patterns = num_patterns

        # Initialize scale parameters to 10% of each gene's mean
        self.initial_scale = 0.1 * self.D_gene_means
        self.init_vals = self.initial_scale.unsqueeze(-1)
        self.init_vals = self.init_vals.expand(-1, self.num_patterns)
        #Clamp to min of 0.01
        self.init_vals = torch.clamp(self.init_vals, min=0.01)


        # Define parameters
        self.A_mean = PyroParam(torch.rand((self.num_genes, self.num_patterns), device=self.device),constraint=dist.constraints.positive)
        #self.A_mean = PyroParam(self.init_vals.to(device), constraint=dist.constraints.positive)
        self.A_scale = PyroParam(torch.ones((self.num_genes, self.num_patterns), device=self.device),constraint=dist.constraints.positive)

        self.P_mean = PyroParam(torch.rand((self.num_patterns, self.num_samples), device=self.device),constraint=dist.constraints.positive)
        self.P_scale = PyroParam(torch.ones((self.num_patterns, self.num_samples), device=self.device),constraint=dist.constraints.positive)

        #self.D_mean = PyroParam(torch.tensor(D.mean(), device=self.device))

    def forward(self, D):
        # Priors
        genes_plate = pyro.plate("Genes", self.num_genes, dim=-2)
        patterns_plate = pyro.plate("Patterns", self.num_patterns, dim=-3)

        with genes_plate:
            A = pyro.sample("A", dist.Normal(self.A_mean, self.A_scale).to_event(1))

        with patterns_plate:
            P = pyro.sample("P", dist.Normal(self.P_mean, self.P_scale).to_event(1))

        prediction = torch.matmul(A, P)
        # Likelihood
        #with pyro.plate("data", self.num_genes):
        with genes_plate:
            with patterns_plate:

                pyro.sample("D", dist.Normal(softplus(prediction),torch.ones_like(prediction)).to_event(1), obs=D)
        return prediction