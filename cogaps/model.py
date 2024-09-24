#%%
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
from torch import nn
from torch.nn.functional import softplus

#%% Enable Validations
pyro.enable_validation(True)


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
        self.D_mean = self.D.mean() # is this used ?
        self.D_gene_sd = self.D.std(axis=0) # checked
        self.D_sample_means = self.D.mean(axis=1) # checked
        self.D_sample_sd = self.D.std(axis=1) # checked


        #### A_mean should be patterns x genes ####
        #### P_mean should be samples x patterns ####

        if init_method == "svd":
            print('init_method svd')
            #self.initial_A_mean, self.initial_P_mean = self.nnsvd_initialization(self.D, self.num_patterns)
            _, self.initial_P_mean = self.nnsvd_initialization(self.D, self.num_patterns)
            self.initial_A_mean = torch.rand((self.num_patterns, self.num_genes), device=self.device)
        elif init_method == "mean":
            print('init_method mean')
            # Initialize A_mean and P_mean to gene means and sample_means
            self.initial_A_mean = self.D_gene_means.unsqueeze(-2).expand(self.num_patterns, -1)
            self.initial_P_mean = self.D_sample_means.unsqueeze(-1).expand(-1, self.num_patterns)
        elif init_method == "zeros":
            print('init_method zeros')
            self.initial_A_mean = torch.zeros((self.num_patterns, self.num_genes), device=self.device)
            self.initial_P_mean = torch.zeros((self.num_samples, self.num_patterns), device=self.device)
        else:
            print('init_method random')
            self.initial_A_mean = torch.rand((self.num_patterns, self.num_genes), device=self.device) # try with sparse random
            self.initial_P_mean = torch.rand((self.num_samples, self.num_patterns), device=self.device)

        # Try initializing std as 1
        self.initial_A_scale = torch.ones((self.num_patterns, self.num_genes), device=self.device, requires_grad=False)
        self.initial_P_scale = torch.ones((self.num_samples, self.num_patterns), device=self.device, requires_grad=False)

        # Define pyro parameters
        self.A_mean = PyroParam(self.initial_A_mean.to(device), constraint=dist.constraints.positive)
        self.A_scale =self.initial_A_scale.to(device) #,requires_grad=False
        self.P_mean = PyroParam(self.initial_P_mean.to(device), constraint=dist.constraints.positive)
        self.P_scale = self.initial_P_scale.to(device) #,requires_grad=False


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
        ## Priors

        ## Recalculate A scale and P scale here based off mean
        self.gene_sd = self.A_mean.std(dim=0)
        self.sample_sd = self.P_mean.std(dim=1)

        self.A_scale = self.gene_sd.unsqueeze(-2).expand(self.num_patterns, -1).clamp(min=0.01)
        self.P_scale = self.sample_sd.unsqueeze(-1).expand(-1, self.num_patterns).clamp(min=0.01)

        genes_plate = pyro.plate("Genes", self.num_genes, dim=-2)
        patterns_plate = pyro.plate("Patterns", self.num_patterns, dim=-3)

        with genes_plate:
            A = pyro.sample("A", dist.Gamma(self.A_mean, 1/self.A_scale).to_event(1))

        with patterns_plate:
            P = pyro.sample("P", dist.Gamma(self.P_mean, 1/self.P_scale).to_event(1))

        prediction = torch.matmul(P, A) #+self.sigma ?

        with genes_plate:
            with patterns_plate:
                pyro.sample("D", dist.Normal(softplus(prediction),torch.ones_like(prediction)).to_event(1), obs=D)
        return prediction

