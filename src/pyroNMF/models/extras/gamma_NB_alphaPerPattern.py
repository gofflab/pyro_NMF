#%%
# Gamma NB models with one alpha (concentration) per pattern
# For A (patterns x genes): one alpha per pattern (row) -> shape (num_patterns,)
# For P (samples x patterns): one alpha per pattern (column) -> shape (num_patterns,)
# loc_A and loc_P remain the learned rate parameters
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
class GammaAlphaPerPattern_NegBinomial_base(PyroModule):
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
        print(f" ################# Running Gamma-Negative Binomial Model (Alpha Per Pattern) #################")
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

        self.markers_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Ascaled = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Pscaled = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Asoftmax = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Psoftmax = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        ## Set up the pyro parameters
        #### Matrix A is patterns x genes ####
        #### loc_A is the learned rate; shape (patterns, genes) ####
        self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)
        
        #### Matrix P is samples x patterns ####
        #### loc_P is the learned rate; shape (samples, patterns) ####
        self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)

        #### One alpha per pattern for A (row of A = one pattern) -> shape (num_patterns, 1) ####
        #### Broadcast over genes automatically ####
        self.alpha_A = PyroParam(2.0 * torch.ones(self.num_patterns, 1, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)

        #### One alpha per pattern for P (column of P = one pattern) -> shape (1, num_patterns) ####
        #### Broadcast over samples automatically ####
        self.alpha_P = PyroParam(0.5 * torch.ones(1, self.num_patterns, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)


    def forward(self, D, U, samp=False):

        self.iter += 1 # keep a running total of iterations

        # alpha_A is (num_patterns, 1), broadcasts to (num_patterns, num_genes)
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.alpha_A, self.loc_A)) # one alpha per pattern row
        self.A = A # save A to model

        # alpha_P is (1, num_patterns), broadcasts to (num_samples, num_patterns)
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.alpha_P, self.loc_P)) # one alpha per pattern column
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




class GammaAlphaPerPattern_NegBinomial_SSFixedSamples(GammaAlphaPerPattern_NegBinomial_base):
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
        print(f"################# Running Gamma-Negative Binomial Model (Alpha Per Pattern) with fixed samples #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")

        total_patterns = self.num_fixed_patterns + self.num_patterns

        #### Matrix A is patterns (supervised+unsupervised) x genes ####
        self.loc_A = PyroParam(torch.rand(total_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)        
        self.best_A = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.best_locA = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(total_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### alpha_A: one per pattern row, expanded for total patterns ####
        self.alpha_A = PyroParam(2.0 * torch.ones(total_patterns, 1, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)

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

        # alpha_A is (total_patterns, 1), broadcasts over genes
        with pyro.plate("patterns", total_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.alpha_A, self.loc_A)) # one alpha per pattern row
        self.A = A

        # alpha_P is (1, num_patterns), broadcasts over samples (unsupervised patterns only)
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.alpha_P, self.loc_P)) # one alpha per pattern column
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
