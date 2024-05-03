import torch
import pyro
import pyro.distributions as dist

#%% Enable Validations
pyro.enable_validation(True)

#%%
def cogaps_model(D, num_patterns):
    num_genes, num_samples = D.shape
    #print(f"num_genes: {num_genes}\nnum_samples: {num_samples}\nnum_patterns: {num_patterns}")
    # Parameters should be defined outside the plate if they do not depend on it
    A_scale = pyro.param("A_scale", torch.ones((num_genes, num_patterns)), constraint=dist.constraints.positive)
    #print(A_scale.shape)
    P_scale = pyro.param("P_scale", torch.ones((num_patterns, num_samples)), constraint=dist.constraints.positive)

    with pyro.plate("genes", num_genes):
        #print(dist.Exponential(rate=A_scale))
        A = pyro.sample("A", dist.Exponential(rate=A_scale).to_event(1))

    with pyro.plate("patterns", num_patterns):
        P = pyro.sample("P", dist.Exponential(rate=P_scale).to_event(1))

    prediction = torch.matmul(A, P)
    
    with pyro.plate("data", num_genes):
        #pyro.sample("obs", dist.Poisson(prediction), obs=D.T)
        # Use Negative Binomial distribution instead of Poisson
        #pyro.sample("obs", dist.NegativeBinomial(total_count=1, logits=prediction), obs=D.T)
        # Use Gaussian (normal) distribution instead of Poisson or Negative Binomial
        pyro.sample("obs", dist.Normal(prediction, torch.ones_like(prediction)).to_event(1), obs=D)

