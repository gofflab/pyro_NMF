
import torch
import pyro
import pyro.distributions as dist

def guide(D, num_patterns):
    num_genes, num_samples = D.shape
    A_loc = pyro.param("A_loc", torch.rand((num_genes, num_patterns)))
    A_scale = pyro.param("A_scale_guide", torch.ones((num_genes, num_patterns)), constraint=dist.constraints.positive)
    P_loc = pyro.param("P_loc", torch.rand((num_patterns, num_samples)))
    P_scale = pyro.param("P_scale_guide", torch.ones((num_patterns, num_samples)), constraint=dist.constraints.positive)

    with pyro.plate("genes", num_genes):
        pyro.sample("A", dist.LogNormal(A_loc, A_scale).to_event(1))

    with pyro.plate("patterns", num_patterns):
        pyro.sample("P", dist.LogNormal(P_loc, P_scale).to_event(1))
