
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
        self.initial_scale = torch.clamp(self.initial_scale, min=0.01) # set to max of 10% gene mean or 0.1 whichever is higher
        self.init_vals = self.initial_scale.unsqueeze(-1)
        self.init_vals = self.init_vals.expand(-1, self.num_patterns)

        # Define parameters for the guide
        self.A_loc = PyroParam(torch.randn((self.num_genes, self.num_patterns), device=self.device)*0.1)
        #self.A_scale_guide = PyroParam(torch.full((self.num_genes, self.num_patterns), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
        self.A_scale_guide = PyroParam(self.init_vals.to(self.device), constraint=dist.constraints.positive)

        self.P_loc = PyroParam(torch.randn((self.num_patterns, self.num_samples), device=self.device)*0.1)
        #self.P_scale_guide = PyroParam(torch.full((self.num_patterns, self.num_samples), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
        self.P_scale_guide = PyroParam(torch.ones((self.num_patterns, self.num_samples), device=self.device), constraint=dist.constraints.positive)
        
    def forward(self, D):
        with pyro.plate("genes", self.num_genes):
            pyro.sample("A", dist.LogNormal(self.A_loc, self.A_scale_guide).to_event(1))

        with pyro.plate("patterns", self.num_patterns):
            pyro.sample("P", dist.LogNormal(self.P_loc, self.P_scale_guide).to_event(1))

# This does not work yet
class CoGAPSGuideWithAtomicPriorGibbs(PyroModule):
    def __init__(self, D, num_patterns, lA, lP, num_atoms_A, num_atoms_P, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_genes = D.shape[0]
        self.num_samples = D.shape[1]
        self.num_patterns = num_patterns
        self.lA = lA
        self.lP = lP
        self.num_atoms_A = num_atoms_A
        self.num_atoms_P = num_atoms_P

        # Initialize coordinates of atoms in the atomic domain for matrices A and P
        self.atom_coordinates_A = pyro.param("atom_coordinates_A", torch.rand((self.num_genes, self.num_patterns, self.num_atoms_A), device=self.device))
        self.atom_coordinates_P = pyro.param("atom_coordinates_P", torch.rand((self.num_patterns, self.num_samples, self.num_atoms_P), device=self.device))
        self.counter = 0

    def forward(self, D):
        # Update the coordinates of atoms within the atomic domain using Gibbs sampling
        self.atom_coordinates_A = self.update_atom_coordinates(self.atom_coordinates_A, self.lA)
        self.atom_coordinates_P = self.update_atom_coordinates(self.atom_coordinates_P, self.lP)

    def update_atom_coordinates(self, atom_coordinates, rate):
        updated_atom_coordinates = atom_coordinates.clone()
        for i in range(atom_coordinates.shape[0]):
            for k in range(atom_coordinates.shape[1]):
                with pyro.plate(f"data_A_{self.counter}", size=atom_coordinates.shape[2], dim=-1):
                    updated_atom_coordinates[i, k] = pyro.sample(f"atom_coordinates_A_{self.counter}", dist.Exponential(rate))
                    self.counter += 1
        return updated_atom_coordinates