import pyro
import pyro.distributions as dist
import torch
#from pyro.distributions import lkj
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


#%% Is this used? Not updated for A and P yet
#class CoGAPSModel(PyroModule):
#    def __init__(self, D, num_patterns, device=torch.device('cpu')):
#        super().__init__()
#        self.device = device
#        self.num_genes = D.shape[0]
#        self.num_samples = D.shape[1]
#        self.D_gene_means = D.mean(axis=1)
#        self.D_mean = D.mean()
#        self.num_patterns = num_patterns
#
#        # Initialize scale parameters to 10% of each gene's mean
#        self.initial_scale = 0.1 * self.D_gene_means
#        self.init_vals = self.initial_scale.unsqueeze(-1)
#        self.init_vals = self.init_vals.expand(-1, self.num_patterns)
#
#        # Define parameters
#        #self.A_scale = PyroParam(torch.full((self.num_genes, self.num_patterns), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
#        #self.A_scale = PyroParam(self.init_vals.to(device), constraint=dist.constraints.positive)
#        #self.P_scale = PyroParam(torch.full((self.num_patterns, self.num_samples), self.initial_scale, device=self.device), constraint=dist.constraints.positive)
#        #self.P_scale = PyroParam(torch.ones((self.num_patterns, self.num_samples), device=self.device), constraint=dist.constraints.positive)
#        self.A_scale = PyroParam(torch.tensor(1.0, device=device), constraint=dist.constraints.positive,requires_grad=False)
#        self.P_scale = PyroParam(torch.tensor(1.0, device=device), constraint=dist.constraints.positive,requires_grad=False)
#
#    def forward(self, D):
#        with pyro.plate("genes", self.num_genes):
#            A = pyro.sample("A", dist.Exponential(rate=self.A_scale).expand([self.num_genes, self.num_patterns]).to_event(1))
#
#        with pyro.plate("patterns", self.num_patterns):
#            P = pyro.sample("P", dist.Exponential(rate=self.P_scale).expand([self.num_patterns, self.num_samples]).to_event(1))
#
#        prediction = torch.matmul(A, P)
#
#        with pyro.plate("data", self.num_genes):
#            pyro.sample("obs", dist.Normal(prediction, torch.ones_like(prediction)).to_event(1), obs=D)
#
#        return prediction
#
# Starting below from scratch
#%%
class ProbNMFModel(PyroModule):
    def __init__(self,
                 D,
                 num_patterns,
                 device=torch.device('cpu'),
                 init_method="mean", # Options: (["mean", "svd", None]):
            ):
        super().__init__()
        self.device = device
        self.D = D
        self.num_genes = self.D.shape[1] # checked
        self.num_samples = self.D.shape[0] # checked
        self.num_patterns = num_patterns
        self.D_gene_means = self.D.mean(axis=0) # checked
        self.D_mean = self.D.mean() # is this used
        self.D_gene_sd = self.D.std(axis=0) # checked
        self.D_sample_means = self.D.mean(axis=1) # checked
        self.D_sample_sd = self.D.std(axis=1) # checked
        #self.D_mean = D.mean()


        #### A_mean should be patterns x genes ####
        #### P_mean should be samples x patterns ####

        if init_method == "svd":
            print('init_method svd')
            #self.initial_A_mean, self.initial_P_mean = self.nnsvd_initialization(self.D, self.num_patterns)
            _, self.initial_P_mean = self.nnsvd_initialization(self.D, self.num_patterns) ## didn't change this but might be right already?
            self.initial_A_mean = torch.rand((self.num_patterns, self.num_genes), device=self.device)
        elif init_method == "mean":
            print('init_method mean')
            # Initialize A_mean and P_mean to gene means and sample_means
            self.initial_A_mean = self.D_gene_means.unsqueeze(-2).expand(self.num_patterns, -1)
            #self.initial_A_mean = torch.full((self.num_genes,self.num_patterns),self.D_mean)
            # Redefine initial_P_mean to sample means
            self.initial_P_mean = self.D_sample_means.unsqueeze(-1).expand(-1, self.num_patterns)
            #self.initial_P_mean = torch.full((self.num_patterns,self.num_samples),self.D_mean)
        elif init_method == "zeros":
            print('init_method zeros')
            self.initial_A_mean = torch.zeros((self.num_patterns, self.num_genes), device=self.device)
            self.initial_P_mean = torch.zeros((self.num_samples, self.num_patterns), device=self.device)
        else:
            print('init_method random')
            self.initial_A_mean = torch.rand((self.num_patterns, self.num_genes), device=self.device) # try with sparse random
            self.initial_P_mean = torch.rand((self.num_samples, self.num_patterns), device=self.device)
            ## maybe try initializing based on gamma (this would give us sparse); negative
            # NB is poisson with gamma prior on mean (lambda); overdispersed poisson
            # A and P is gamma distributed because sum of exponentials

        # Initialize A_scale and P_scale parameters to gene and sample standard deviations
        #self.initial_A_scale = self.D_gene_sd.unsqueeze(-2).expand(self.num_patterns, -1).clamp(min=0.01)
        #self.initial_P_scale = self.D_sample_sd.unsqueeze(-1).expand(-1, self.num_patterns).clamp(min=0.01)
        # Try initializing std as 1
        self.initial_A_scale = torch.ones((self.num_patterns, self.num_genes), device=self.device, requires_grad=False)
        self.initial_P_scale = torch.ones((self.num_samples, self.num_patterns), device=self.device, requires_grad=False)



        #Clamp to min of 0.01
        #self.init_vals = torch.clamp(self.initial_A_scale, min=0.01)

        # Define parameters
        #self.A_mean = PyroParam(torch.rand((self.num_genes, self.num_patterns), device=self.device),constraint=dist.constraints.positive)
        self.A_mean = PyroParam(self.initial_A_mean.to(device), constraint=dist.constraints.positive)
        #print('self.A_mean param')
        #print(self.A_mean[:,:5])

        #self.A_scale = PyroParam(torch.ones((self.num_genes, self.num_patterns), device=self.device),constraint=dist.constraints.positive)
        #self.A_scale = PyroParam(self.initial_A_scale.to(device), constraint=dist.constraints.positive) #,requires_grad=False
        #self.A_scale = pyro.param(self.initial_A_scale.to(device), constraint=dist.constraints.positive, requires_grad=False) #,requires_grad=False
        self.A_scale =self.initial_A_scale.to(device) #,requires_grad=False

        #self.P_mean = PyroParam(torch.rand((self.num_patterns, self.num_samples), device=self.device),constraint=dist.constraints.positive)
        self.P_mean = PyroParam(self.initial_P_mean.to(device), constraint=dist.constraints.positive)

        #self.P_scale = PyroParam(torch.ones((self.num_patterns, self.num_samples), device=self.device),constraint=dist.constraints.positive)
        #self.P_scale = PyroParam(self.initial_P_scale.to(device), constraint=dist.constraints.positive) # ,requires_grad=False
        #self.P_scale = pyro.param(self.initial_P_scale.to(device), constraint=dist.constraints.positive, requires_grad=False) # ,requires_grad=False
        self.P_scale = self.initial_P_scale.to(device) # ,requires_grad=False

        # Add an error parameter sigma to the matrix multiplication
        #self.sigma = PyroParam(torch.tensor(0.1, device=self.device), constraint=dist.constraints.positive)
        #self.D_mean = PyroParam(torch.tensor(D.mean(), device=self.device))

    def nnsvd_initialization(self, D, num_patterns):
        # Compute SVD
        U, S, V = torch.svd(D)

        # Keep only the first 'num_patterns' components
        U = U[:, :num_patterns]
        S = S[:num_patterns]
        V = V[:, :num_patterns]

        # Split into positive and negative parts
        U_pos = torch.clamp(U, min=0)
        V_pos = torch.clamp(V, min=0)

        # Initialize A and P using positive parts
        S = torch.sqrt(S).unsqueeze(0)
        A_init = (U_pos * S.expand_as(U_pos)).t # transpose to match patterns x genes
        P_init = V_pos * S.expand_as(V_pos) # expand to match samples x patterns

        return A_init, P_init

    def forward(self, D):
        # Priors
        ## TODO recalculate A scale and P scale here baseed off mean
        self.gene_sd = self.A_mean.std(dim=0)
        self.sample_sd = self.P_mean.std(dim=1)
        #print('A_mean in forward')
        #print(self.A_mean[:,:5])
        #print('A sd')
        #print(self.gene_sd)
        self.A_scale = self.gene_sd.unsqueeze(-2).expand(self.num_patterns, -1).clamp(min=0.01) #.unsqueeze(-2).expand(self.num_patterns, -1)
        self.P_scale = self.sample_sd.unsqueeze(-1).expand(-1, self.num_patterns).clamp(min=0.01)
        #print('A scale defined')
        #print(self.A_scale[:,1:5])

        genes_plate = pyro.plate("Genes", self.num_genes, dim=-2)
        patterns_plate = pyro.plate("Patterns", self.num_patterns, dim=-3)

        with genes_plate:
            A = pyro.sample("A", dist.Gamma(self.A_mean, 1/self.A_scale).to_event(1)) # try switching to gamma? updates both the prior (here) and the posterior which is autoguide

        with patterns_plate:
            P = pyro.sample("P", dist.Gamma(self.P_mean, 1/self.P_scale).to_event(1)) # try switching to gamma?

        prediction = torch.matmul(P, A)#+self.sigma
        # Likelihood
        #with pyro.plate("data", self.num_genes):
        with genes_plate:
            with patterns_plate:
                pyro.sample("D", dist.Normal(softplus(prediction),torch.ones_like(prediction)).to_event(1), obs=D)
        return prediction