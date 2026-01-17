"""Exponential-prior NMF models with Negative Binomial likelihoods."""
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
class Exponential_base(PyroModule):
    """Exponential-prior NMF model with Negative Binomial likelihood.

    Factorizes the observed count matrix ``D`` (samples x genes) into
    ``P @ A`` with Exponential priors on both factors. Optionally adds
    chi-squared and/or Poisson terms to the loss, and tracks the best
    parameters by chi-squared during training.

    Parameters
    ----------
    num_samples : int
        Number of samples (rows in ``D``).
    num_genes : int
        Number of genes/features (columns in ``D``).
    num_patterns : int
        Number of latent patterns to learn.
    use_chisq : bool, optional
        If True, adds a chi-squared loss term via ``pyro.factor``.
    use_pois : bool, optional
        If True, adds a Poisson log-likelihood term via ``pyro.factor``.
    NB_probs : float, optional
        Probability parameter for the Negative Binomial likelihood.
    device : torch.device, optional
        Device for parameters and intermediate tensors.
    """
    def __init__(self,
                num_samples,
                num_genes,
                num_patterns,
                use_chisq = False,
                use_pois = False,
                #scale = 1,
                NB_probs = 0.5,
                device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):
    
        """Initialize the model and tracking buffers."""
        super().__init__()

        ## Initialize parameters
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.num_patterns = num_patterns
        self.use_chisq = use_chisq
        self.use_pois = use_pois
        self.NB_probs = NB_probs
        self.device = device

        ## Print settings
        print(f" ################# Running Exponential Model #################")
        print(f"Using {self.device}")
        print(f"Data is {self.num_samples} samples x {self.num_genes} genes")
        print(f"Running for {self.num_patterns} patterns")
        print(f"Using Negative Binomial with probs of {self.NB_probs}")

        if use_chisq:
            print(f"Using chi squared")
        else:
            print(f"Not using chi squared")

        ## Set some initial values to update
        self.best_chisq = np.inf
        self.best_chisq_iter = 0
        self.iter = 0

        self.best_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.best_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)
        
        self.sum_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.sum_A2 = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        ## Set up the pyro parameters
        #### A parameter for Exponential to Populate A Matrix ####
        self.scale_A = PyroParam(torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)

        #### P parameter for Exponential to Populate P Matrix ####
        self.scale_P = PyroParam(torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)


    def forward(self, D, U):
        """Run one stochastic forward pass of the model.

        Parameters
        ----------
        D : torch.Tensor
            Observed count matrix with shape ``(num_samples, num_genes)``.
        U : torch.Tensor
            Per-entry scale/uncertainty for chi-squared computation, same
            shape as ``D``.
        """
        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Exponential(self.scale_A)) # sample A from Exponential
        self.A = A # save A to model

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Exponential(self.scale_P)) # sample P from Exponential
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
            self.best_scaleA = self.scale_A
            self.best_scaleP = self.scale_P

            
        # Include chi squared loss in the model
        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            # Addition to Elbow Loss - should make this at least as large as Elbow
            pyro.factor("pois.loss",10.*poisL)
     
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
        """Placeholder guide (use AutoNormal externally)."""
        pass


class Exponential_SSFixedGenes(Exponential_base):
    """Semi-supervised Exponential model with fixed gene patterns.

    Extends ``Exponential_base`` by fixing a set of patterns over genes and
    learning additional patterns. The fixed patterns are concatenated to the
    learned ``A`` matrix during reconstruction.

    Parameters
    ----------
    num_samples : int
        Number of samples (rows in ``D``).
    num_genes : int
        Number of genes/features (columns in ``D``).
    num_patterns : int
        Number of additional patterns to learn.
    fixed_patterns : array-like
        Fixed patterns with shape ``(num_genes, num_fixed_patterns)``.
    use_chisq : bool, optional
        If True, adds a chi-squared loss term.
    use_pois : bool, optional
        If True, adds a Poisson log-likelihood term.
    NB_probs : float, optional
        Probability parameter for the Negative Binomial likelihood.
    device : torch.device, optional
        Device for parameters and tensors.
    """
    def __init__(self,
                num_samples,
                num_genes,
                num_patterns, # num unsupervised
                fixed_patterns, # of shape genes x fixed patterns
                use_chisq = False,
                use_pois = False,
                #scale = 1,
                NB_probs = 0.5,
                device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):

        """Initialize the semi-supervised model with fixed gene patterns."""
        super().__init__(num_samples, num_genes, num_patterns, use_chisq, use_pois, NB_probs, device) 

        ## This is the same as unsupervised but with a set of fixed A, and P extended by this amount ##
        self.fixed_patterns = fixed_patterns # of shape genes x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]
        print(f"################# Running Exponential Model with fixed genes #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")

        #### Matrix P is samples x patterns (supervised+unsupervised) ####
        self.best_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        #### Matrix A total is expanded ###
        self.sum_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### Fixed patterns are samples x patterns ####
        self.fixed_A = torch.tensor(fixed_patterns, device=self.device,dtype=default_dtype) # tensor, not updatable

    def forward(self, D, U):
        """Run one stochastic forward pass with fixed gene patterns.

        Parameters
        ----------
        D : torch.Tensor
            Observed count matrix with shape ``(num_samples, num_genes)``.
        U : torch.Tensor
            Per-entry scale/uncertainty for chi-squared computation.
        """

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Exponential(self.scale_A))
        self.A = A

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_fixed_patterns + self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Exponential(self.scale_P)) # sample P from Exponential
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
            self.best_scaleA = self.scale_A
            self.best_scaleP = self.scale_P


        # Include chi squared loss in the model
        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            # Addition to Elbow Loss - should make this at least as large as Elbow
            pyro.factor("pois.loss",10.*poisL)
        
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
        """Placeholder guide (use AutoNormal externally)."""
        pass




class Exponential_SSFixedSamples(Exponential_base):
    """Semi-supervised Exponential model with fixed sample patterns.

    Extends ``Exponential_base`` by fixing a set of patterns over samples and
    learning additional patterns. The fixed patterns are concatenated to the
    learned ``P`` matrix during reconstruction.

    Parameters
    ----------
    num_samples : int
        Number of samples (rows in ``D``).
    num_genes : int
        Number of genes/features (columns in ``D``).
    num_patterns : int
        Number of additional patterns to learn.
    fixed_patterns : array-like
        Fixed patterns with shape ``(num_samples, num_fixed_patterns)``.
    use_chisq : bool, optional
        If True, adds a chi-squared loss term.
    use_pois : bool, optional
        If True, adds a Poisson log-likelihood term.
    NB_probs : float, optional
        Probability parameter for the Negative Binomial likelihood.
    device : torch.device, optional
        Device for parameters and tensors.
    """
    def __init__(self,
                num_samples,
                num_genes,
                num_patterns, # num unsupervised
                fixed_patterns, # of shape samples x fixed patterns
                use_chisq = False,
                use_pois = False,
                #scale = 1,
                NB_probs = 0.5,
                device=torch.device('cpu')
                 #init_method="mean", # Options: (["mean", "svd", None]): TODOS
            ):

        """Initialize the semi-supervised model with fixed sample patterns."""
        super().__init__(num_samples, num_genes, num_patterns, use_chisq, use_pois, NB_probs, device) 

        ## This is the same as unsupervised but with a set of fixed P and A extended by this amount ##

        self.fixed_patterns = fixed_patterns # of shape samples x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]
        print(f"################# Running Exponential Model with fixed samples #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")


        #### Matrix A is patterns (supervised+unsupervised) x genes ####
        self.best_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### Matrix P total is expanded ###
        self.sum_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        #### Fixed patterns are samples x patterns ####
        self.fixed_P = torch.tensor(fixed_patterns, device=self.device,dtype=default_dtype) # tensor, not updatable

    def forward(self, D, U):
        """Run one stochastic forward pass with fixed sample patterns.

        Parameters
        ----------
        D : torch.Tensor
            Observed count matrix with shape ``(num_samples, num_genes)``.
        U : torch.Tensor
            Per-entry scale/uncertainty for chi-squared computation.
        """

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_fixed_patterns + self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Exponential(self.scale_A))
        self.A = A

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Exponential(self.scale_P)) # sample P from Exponential
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
            self.best_scaleA = self.scale_A
            self.best_scaleP = self.scale_P


        # Include chi squared loss in the model
        if self.use_chisq:
            pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            # Addition to Elbow Loss - should make this at least as large as Elbow
            pyro.factor("pois.loss",10.*poisL)

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
    """Placeholder guide (use AutoNormal externally)."""
    pass
