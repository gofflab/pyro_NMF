
import torch
import pyro
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam

# Deprecated
# def guide(D, num_patterns):
#     num_genes, num_samples = D.shape
#     A_loc = pyro.param("A_loc", torch.rand((num_genes, num_patterns)))
#     A_scale = pyro.param("A_scale_guide", torch.ones((num_genes, num_patterns)), constraint=dist.constraints.positive)
#     P_loc = pyro.param("P_loc", torch.rand((num_patterns, num_samples)))
#     P_scale = pyro.param("P_scale_guide", torch.ones((num_patterns, num_samples)), constraint=dist.constraints.positive)

#     with pyro.plate("genes", num_genes):
#         pyro.sample("A", dist.LogNormal(A_loc, A_scale).to_event(1))

#     with pyro.plate("patterns", num_patterns):
#         pyro.sample("P", dist.LogNormal(P_loc, P_scale).to_event(1))

class CoGAPSGuide(PyroModule):
    def __init__(self, D, num_patterns, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_genes = D.shape[0]
        self.num_samples = D.shape[1]
        self.D_mean = D.mean()
        self.D_gene_means = D.mean(axis=1)
        self.num_patterns = num_patterns

        # Initialize scale parameters to 10% of each gene's mean
        self.initial_scale = 0.1 * self.D_gene_means
        self.init_vals = self.initial_scale.unsqueeze(-1)
        self.init_vals = self.init_vals.expand(-1, self.num_patterns)

        # Define parameters for the guide
        self.A_loc = PyroParam(torch.rand((self.num_genes, self.num_patterns), device=self.device)*0.1)
        #self.A_scale_guide = PyroParam(torch.full((self.num_genes, self.num_patterns), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
        self.A_scale_guide = PyroParam(self.init_vals.to(self.device), constraint=dist.constraints.positive)

        self.P_loc = PyroParam(torch.rand((self.num_patterns, self.num_samples), device=self.device)*0.1)
        #self.P_scale_guide = PyroParam(torch.full((self.num_patterns, self.num_samples), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
        self.P_scale_guide = PyroParam(torch.ones((self.num_patterns, self.num_samples), device=self.device), constraint=dist.constraints.positive)
        
    def forward(self, D):
        with pyro.plate("genes", self.num_genes):
            pyro.sample("A", dist.LogNormal(self.A_loc, self.A_scale_guide).to_event(1))

        with pyro.plate("patterns", self.num_patterns):
            pyro.sample("P", dist.LogNormal(self.P_loc, self.P_scale_guide).to_event(1))

