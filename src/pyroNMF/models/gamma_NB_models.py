#%%
# Consolidate all the gamma NB models
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
import torch
import matplotlib.pyplot as plt
import numpy as np

#%% Enable Validations
pyro.enable_validation(True)

default_dtype = torch.float32

#%%
class Gamma_NegBinomial_base(PyroModule):
    def __init__(self,
                num_samples,
                num_genes,
                num_patterns,
                use_chisq = False,
                use_pois = False,
                scale = 1,
                NB_probs = 0.5,
                device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):
    
        super().__init__()

        ## Initialize parameters
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.use_chisq = use_chisq
        self.use_pois = use_pois
        self.scale = scale
        self.NB_probs = NB_probs
        self.device = device

        ## Print settings
        print(f" ################# Running Gamma-Negative Binomial Model #################")
        print(f"Using {self.device}")
        print(f"Data is {self.num_samples} samples x {self.num_genes} genes")
        print(f"Running for {self.num_patterns} patterns")
        print(f"Using scale of {self.scale} for the gamma distribution")
        print(f"Using Negative Binomial with probs of {self.NB_probs}")

        if use_chisq:
            print(f"Using chi squared")
        else:
            print(f"Not using chi squared")

        ## Set some initial values to update
        self.best_chisq = np.inf
        self.best_chisq_iter = 0
        self.iter = 0

        self.scale = torch.tensor(scale, device=self.device, dtype=default_dtype)

        self.best_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.best_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)
        self.best_locA = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.best_locP = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.sum_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)
        
        self.sum_A2 = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)
        self.A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        ## Set up the pyro parameters
        #### Matrix A is patterns x genes ####
        #### Initialize randomly, but with positive constraint ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.nonnegative)
        
        #### Matrix P is samples x patterns ####
        #### Initialize randomly, but with positive constraint ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype),constraint=dist.constraints.nonnegative)
        #### Single fixed scale parameter for gammas ####
        self.scale = scale


    def forward(self, D, U, samp=False):

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale)) # sample A from Gamma
        self.A = A # save A to model

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale)) # sample P from Gamma
        self.P = P # save P to model

        # D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed to model

        # Calculate chi squared
        chi2 = torch.sum((D_reconstructed-D)**2/U**2)
        self.chi2  = chi2
        theta = self.D_reconstructed
        poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
        self.pois  = poisL

        if chi2 < self.best_chisq: # if this is a better chi squared, save it
            self.best_chisq = chi2
            self.best_chisq_iter = self.iter
            self.best_A = A
            self.best_P = P
            self.best_locA = self.loc_A
            self.best_locP = self.loc_P

        # Include chi squared loss in the model
        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            # Addition to Elbow Loss - should make this at least as large as Elbow
            pyro.factor("pois.loss",10.*poisL)
        if samp:
            with torch.no_grad():
                correction = P.max(axis=0).values
                Pn = P / correction
                An = A * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn) 

        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) 

    def guide(D):
        pass


class Gamma_NegBinomial_SSFixedGenes(Gamma_NegBinomial_base):
    def __init__(self,
                num_samples,
                num_genes,
                num_patterns, # num unsupervised
                fixed_patterns, # of shape genes x fixed patterns
                use_chisq = False,
                use_pois = False,
                scale = 1,
                NB_probs = 0.5,
                device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):

        super().__init__(num_samples, num_genes, num_patterns, use_chisq, use_pois, scale, NB_probs, device) 

        ## This is the same as unsupervised but with a set of fixed A, and P extended by this amount ##

        self.fixed_patterns = fixed_patterns # of shape genes x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]
        print(f"################# Running Gamma-Negative Binomial Model with fixed genes #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")


        #### Matrix P is samples x patterns (supervised+unsupervised) ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype), constraint=dist.constraints.nonnegative)        
        self.best_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.best_locP = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        #### Matrix A total is expanded ###
        self.sum_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### Fixed patterns are samples x patterns ####
        self.fixed_A = torch.tensor(fixed_patterns, device=self.device,dtype=default_dtype) # tensor, not updatable

    def forward(self, D, U, samp=False):

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale)) # sample A from Gamma
        self.A = A

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_fixed_patterns + self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale)) # sample P from Gamma
        self.P = P

        A_total = torch.cat((self.fixed_A.T, A), dim=0)
        self.A_total = A_total # save P_total

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P, A_total)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed
        
        # Calculate chi squared
        chi2 = torch.sum((D_reconstructed-D)**2/U**2)
        self.chi2  = chi2        
        theta = self.D_reconstructed
        poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
        self.pois  = poisL

        if chi2 < self.best_chisq: # if this is a better chi squared, save it
            self.best_chisq = chi2
            self.best_chisq_iter = self.iter
            self.best_A = A
            self.best_P = P
            self.best_locA = self.loc_A
            self.best_locP = self.loc_P

        # Include chi squared loss in the model
        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            # Addition to Elbow Loss - should make this at least as large as Elbow
            pyro.factor("pois.loss",10.*poisL)
        
        if samp:
            with torch.no_grad():
                correction = P.max(axis=0).values
                Pn = P / correction
                An = A_total * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn) 


        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) 

    def guide(D):
        pass




class Gamma_NegBinomial_SSFixedSamples(Gamma_NegBinomial_base):
    def __init__(self,
                num_samples,
                num_genes,
                num_patterns, # num unsupervised
                fixed_patterns, # of shape samples x fixed patterns
                use_chisq = False,
                use_pois = False,
                scale = 1,
                NB_probs = 0.5,
                device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):

        super().__init__(num_samples, num_genes, num_patterns, use_chisq, use_pois, scale, NB_probs, device) 

        ## This is the same as unsupervised but with a set of fixed P and A extended by this amount ##

        self.fixed_patterns = fixed_patterns # of shape samples x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]
        print(f"################# Running Gamma-Negative Binomial Model with fixed samples #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")


        #### Matrix A is patterns (supervised+unsupervised) x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.nonnegative)        
        self.best_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.best_locA = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### Matrix P total is expanded ###
        self.sum_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        #### Fixed patterns are samples x patterns ####
        self.fixed_P = torch.tensor(fixed_patterns, device=self.device,dtype=default_dtype) # tensor, not updatable

    def forward(self, D, U, samp=False):

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_fixed_patterns + self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale)) # sample A from Gamma
        self.A = A

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale)) # sample P from Gamma
        self.P = P

        P_total = torch.cat((self.fixed_P, P), dim=1)
        self.P_total = P_total # save P_total

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P_total, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed # save D_reconstructed
        
        # Calculate chi squared
        chi2 = torch.sum((D_reconstructed-D)**2/U**2)
        self.chi2  = chi2
        theta = self.D_reconstructed
        poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
        self.pois  = poisL

        if chi2 < self.best_chisq: # if this is a better chi squared, save it
            self.best_chisq = chi2
            self.best_chisq_iter = self.iter
            self.best_A = A
            self.best_P = P
            self.best_locA = self.loc_A
            self.best_locP = self.loc_P

        # Include chi squared loss in the model
        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            # Addition to Elbow Loss - should make this at least as large as Elbow
            pyro.factor("pois.loss",10.*poisL)

        if samp:
            with torch.no_grad():
                correction = P_total.max(axis=0).values
                Pn = P_total / correction
                An = A * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn)
        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) 


def guide(D):
    pass

