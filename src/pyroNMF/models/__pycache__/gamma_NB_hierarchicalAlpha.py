#%%
# Gamma NB models with hierarchical hyperprior on alpha (concentration)
# alpha_A and alpha_P are sampled latent variables drawn from Gamma(0.1, 1)
# This places a sparse hyperprior on the concentration, with the KL term
# in the ELBO penalizing entries whose alpha drifts above the prior.
# loc_A and loc_P remain the learned rate parameters (as in fixedAlphas versions).
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
class GammaHierarchicalAlpha_NegBinomial_base(PyroModule):
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
        print(f" ################# Running Gamma-Negative Binomial Model (Hierarchical Alpha) #################")
        print(f"Using {self.device}")
        print(f"Data is {self.num_samples} samples x {self.num_genes} genes")
        print(f"Running for {self.num_patterns} patterns")
        print(f"Using scale of {self.scale} for the gamma distribution")
        print(f"Using Negative Binomial with probs of {self.NB_probs}")
        print(f"Alpha drawn from Gamma(0.1, 1) hyperprior - sparse by default")

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

        self.markers_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Ascaled = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Pscaled = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Asoftmax = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Psoftmax = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        ## Set up the pyro parameters
        #### loc_A and loc_P are learned rate parameters (as in fixedAlphas versions) ####
        #### alpha_A and alpha_P are NOT PyroParams - they are sampled in forward() ####

        #### Matrix A is patterns x genes ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)
        
        #### Matrix P is samples x patterns ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)

        ## Hyperprior constants - Gamma(0.1, 1) puts most mass below 1, favouring sparse alphas
        ## These are not learned; they define the prior. Change them to adjust sparsity strength.
        self.hyperprior_concentration = torch.tensor(0.1, device=self.device, dtype=default_dtype)
        self.hyperprior_rate = torch.tensor(1.0, device=self.device, dtype=default_dtype)


    def forward(self, D, U, samp=False):

        self.iter += 1 # keep a running total of iterations

        # Sample alpha_A from sparse hyperprior - shape (num_patterns, num_genes)
        # Each entry gets its own concentration drawn from Gamma(0.1, 1)
        # The KL term penalises alphas that drift away from the hyperprior
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                alpha_A = pyro.sample("alpha_A", dist.Gamma(self.hyperprior_concentration, self.hyperprior_rate))
                A = pyro.sample("A", dist.Gamma(alpha_A, self.loc_A))
        self.A = A # save A to model

        # Sample alpha_P from sparse hyperprior - shape (num_samples, num_patterns)
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                alpha_P = pyro.sample("alpha_P", dist.Gamma(self.hyperprior_concentration, self.hyperprior_rate))
                P = pyro.sample("P", dist.Gamma(alpha_P, self.loc_P))
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
                #correction = P.max(axis=0).values
                correction = P.sum(axis=0)
                Pn = P / correction
                An = A * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn) 

                max_pat_per_gene = A.argmax(dim=0)  # shape: (Gene,)
                A_binary = torch.zeros_like(A)
                A_binary[max_pat_per_gene, torch.arange(A.shape[1])] = 1
                self.markers_A += A_binary

                max_pat_per_samp = P.argmax(dim=1)  # shape: (Samp,)
                P_binary = torch.zeros_like(P)
                P_binary[torch.arange(P.shape[0]), max_pat_per_samp] = 1
                self.markers_P += P_binary

                max_pat_per_gene_scaled = An.argmax(dim=0)  # shape: (Gene,)
                A_binaryscaled = torch.zeros_like(An)
                A_binaryscaled[max_pat_per_gene_scaled, torch.arange(An.shape[1])] = 1
                self.markers_Ascaled += A_binaryscaled

                max_pat_per_samp_scaled = Pn.argmax(dim=1)  # shape: (Samp,)
                P_binaryscaled = torch.zeros_like(Pn)
                P_binaryscaled[torch.arange(Pn.shape[0]), max_pat_per_samp_scaled] = 1
                self.markers_Pscaled += P_binaryscaled

                sumPerPat = Pn.sum(dim=1)  # shape: (Samp,)
                self.markers_Psoftmax += (Pn / sumPerPat.unsqueeze(1))

                sumPerGene = An.sum(dim=0)  # shape: (Samp,)
                self.markers_Asoftmax += (An / sumPerGene)


        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) 

    def guide(D):
        pass




class GammaHierarchicalAlpha_NegBinomial_SSFixedSamples(GammaHierarchicalAlpha_NegBinomial_base):
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
        print(f"################# Running Gamma-Negative Binomial Model (Hierarchical Alpha) with fixed samples #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")

        total_patterns = self.num_fixed_patterns + self.num_patterns

        #### Matrix A is patterns (supervised+unsupervised) x genes ####
        self.loc_A = PyroParam(torch.rand(total_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)        
        self.best_A = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.best_locA = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### Matrix P total is expanded ###
        self.sum_P = torch.zeros(self.num_samples, total_patterns, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, total_patterns, device=self.device, dtype=default_dtype)

        #### Fixed patterns are samples x patterns ####
        self.fixed_P = torch.tensor(fixed_patterns, device=self.device, dtype=default_dtype) # tensor, not updatable

        self.markers_A = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_P = torch.zeros(self.num_samples, total_patterns, device=self.device, dtype=default_dtype)

        self.markers_Ascaled = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Pscaled = torch.zeros(self.num_samples, total_patterns, device=self.device, dtype=default_dtype)

        self.markers_Asoftmax = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Psoftmax = torch.zeros(self.num_samples, total_patterns, device=self.device, dtype=default_dtype)


    def forward(self, D, U, samp=False):

        self.iter += 1 # keep a running total of iterations

        total_patterns = self.num_fixed_patterns + self.num_patterns

        # Sample alpha_A from sparse hyperprior for all patterns x genes
        with pyro.plate("patterns", total_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                alpha_A = pyro.sample("alpha_A", dist.Gamma(self.hyperprior_concentration, self.hyperprior_rate))
                A = pyro.sample("A", dist.Gamma(alpha_A, self.loc_A))
        self.A = A

        # Sample alpha_P from sparse hyperprior for unsupervised patterns only
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                alpha_P = pyro.sample("alpha_P", dist.Gamma(self.hyperprior_concentration, self.hyperprior_rate))
                P = pyro.sample("P", dist.Gamma(alpha_P, self.loc_P))
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
                #correction = P_total.max(axis=0).values
                correction = P_total.sum(axis=0)

                Pn = P_total / correction
                An = A * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn)

                max_pat_per_gene = A.argmax(dim=0)  # shape: (Gene,)
                A_binary = torch.zeros_like(A)
                A_binary[max_pat_per_gene, torch.arange(A.shape[1])] = 1
                self.markers_A += A_binary

                max_pat_per_samp = P_total.argmax(dim=1)  # shape: (Samp,)
                P_binary = torch.zeros_like(P_total)
                P_binary[torch.arange(P_total.shape[0]), max_pat_per_samp] = 1
                self.markers_P += P_binary

                max_pat_per_gene_scaled = An.argmax(dim=0)  # shape: (Gene,)
                A_binaryscaled = torch.zeros_like(An)
                A_binaryscaled[max_pat_per_gene_scaled, torch.arange(An.shape[1])] = 1
                self.markers_Ascaled += A_binaryscaled

                max_pat_per_samp_scaled = Pn.argmax(dim=1)  # shape: (Samp,)
                P_binaryscaled = torch.zeros_like(Pn)
                P_binaryscaled[torch.arange(Pn.shape[0]), max_pat_per_samp_scaled] = 1
                self.markers_Pscaled += P_binaryscaled

                sumPerPat = Pn.sum(dim=1)  # shape: (Samp,)
                self.markers_Psoftmax += (Pn / sumPerPat.unsqueeze(1))

                sumPerGene = An.sum(dim=0)  # shape: (Samp,)
                self.markers_Asoftmax += (An / sumPerGene)


        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) 


def guide(D):
    pass
