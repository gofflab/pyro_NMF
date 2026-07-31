#%%
# Consolidate all the gamma NB models
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn.module import PyroParam
import torch
import matplotlib.pyplot as plt
import numpy as np
import gc
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
                device=torch.device('cpu'),
                init_method="random", # Options: (["mean", "svd", None]): TODOS
                debug=False
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

        self.markers_A = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_P = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Ascaled = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Pscaled = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Asoftmax = torch.zeros(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Psoftmax = torch.zeros(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)

        ## Set up the pyro parameters
        #### Matrix A is patterns x genes ####
        #### Matrix P is samples x patterns ####

        if init_method == "random":
            print("Initializing randomly with positive constraint")
            #### Initialize randomly, but with positive constraint ####
            self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)+10e-8, constraint=dist.constraints.positive)
            self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)+10e-8,constraint=dist.constraints.positive)

        elif init_method == "ones":
            print("Initializing to all ones with positive constraint")
            self.loc_A = PyroParam(torch.ones(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)
            self.loc_P = PyroParam(torch.ones(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype),constraint=dist.constraints.positive)

        else:
            print(f"Initialization method {init_method} not recognized, defaulting to random")
            self.loc_A = PyroParam(torch.rand(self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)+10e-8, constraint=dist.constraints.positive)
            self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_patterns, device=self.device, dtype=default_dtype)+10e-8,constraint=dist.constraints.positive)


        #### Single fixed scale parameter for gammas ####
        #self.scale = scale

        print("Initialized tensors")
        if debug:
            def debug_cuda_tensors(step, top_n=20):
                """Print the largest tensors currently allocated on CUDA."""
                tensors = []
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            tensors.append((obj.element_size() * obj.nelement(), obj.shape, obj.dtype, obj.requires_grad))
                    except:
                        pass
                tensors.sort(key=lambda x: x[0], reverse=True)
                print(f"\n--- Step {step}: {len(tensors)} CUDA tensors, top {top_n} ---")
                for size, shape, dtype, rg in tensors[:top_n]:
                    print(f"  {size/1e6:.1f} MB  {shape}  {dtype}  requires_grad={rg}")
                print(f"  Total allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")

            debug_cuda_tensors(0) # Call this at the start to check initial memory usage

    def forward(self, D, U=None, samp=False):
        
        #print(f'Uncertainty is {U}')

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale)) # sample A from Gamma
        self.A = A.detach().clone() # save A to model

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale)) # sample P from Gamma
        self.P = P.detach().clone() # save P to model

        # D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P, A)  # (samples x genes)

        #assert torch.isfinite(D_reconstructed).all(), "total_count has nan/inf"
        #assert (D_reconstructed >= 0).all(), "total_count has negative values"
        

        # All bookkeeping after the pyro.sample — detach everything
        self.D_reconstructed = D_reconstructed.detach() # save D_reconstructed to model

        # Calculate chi squared and track best chi squared if U is provided
        if U is not None: 
            chi2 = torch.sum((D_reconstructed-D)**2/U**2)
            self.chi2  = chi2.item()
            #theta = self.D_reconstructed
            #poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            #self.pois  = poisL.item()

            if chi2 < self.best_chisq: # if this is a better chi squared, save it
                self.best_chisq = chi2.item()
                self.best_chisq_iter = self.iter
                self.best_A = A.detach().clone()
                self.best_P = P.detach().clone()
                self.best_locA = self.loc_A.detach().clone()
                self.best_locP = self.loc_P.detach().clone()

            # Include chi squared loss in the model
            if self.use_chisq:
                pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            self.pois  = poisL.item()
            # Addition to Elbow Loss - should make this at least as large as Elbow
            pyro.factor("pois.loss",10.*poisL)

        if samp:
            with torch.no_grad():
                curr_P = self.P 
                curr_A = self.A

                #correction = P.max(axis=0).values
                correction = curr_P.sum(axis=0)
                Pn = curr_P / correction
                An = curr_A * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn) 

                max_pat_per_gene = curr_A.argmax(dim=0)  # shape: (Gene,)
                A_binary = torch.zeros_like(curr_A)
                A_binary[max_pat_per_gene, torch.arange(A.shape[1])] = 1
                self.markers_A += A_binary

                max_pat_per_samp = curr_P.argmax(dim=1)  # shape: (Samp,)
                P_binary = torch.zeros_like(curr_P)
                P_binary[torch.arange(curr_P.shape[0]), max_pat_per_samp] = 1
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
                device=torch.device('cpu'),
                init_method="random", # Options: (["mean", "svd", None]): TODOS
                debug=False
            ):

        super().__init__(num_samples, num_genes, num_patterns, use_chisq, use_pois, scale, NB_probs, device, init_method, debug=debug)

        ## This is the same as unsupervised but with a set of fixed A, and P extended by this amount ##

        self.fixed_patterns = fixed_patterns # of shape genes x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]

        print(f"################# Running Gamma-Negative Binomial Model with fixed genes #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")


        #### Matrix P is samples x patterns (supervised+unsupervised) ####
        #self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype), constraint=dist.constraints.nonnegative)        
        if init_method == "random":
            print("Initializing loc_P randomly with positive constraint")
            self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype) + 10e-8, constraint=dist.constraints.positive)
        elif init_method == "ones":
            print("Initializing loc_P to all ones with positive constraint")
            self.loc_P = PyroParam(torch.ones(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)
        else:
            print(f"Initialization method {init_method} not recognized, defaulting to random")
            self.loc_P = PyroParam(torch.rand(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype) + 10e-8, constraint=dist.constraints.positive)
            
        self.best_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.best_locP = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        #### Matrix A total is expanded ###
        self.sum_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### Fixed patterns are samples x patterns ####
        self.fixed_A = torch.tensor(fixed_patterns, device=self.device,dtype=default_dtype) # tensor, not updatable


        self.markers_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Ascaled = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Pscaled = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Asoftmax = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Psoftmax = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        print("Updated initialized tensors to reflect fixed genes")

        if debug:
            def debug_cuda_tensors(step, top_n=20):
                """Print the largest tensors currently allocated on CUDA."""
                tensors = []
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            tensors.append((obj.element_size() * obj.nelement(), obj.shape, obj.dtype, obj.requires_grad))
                    except:
                        pass
                tensors.sort(key=lambda x: x[0], reverse=True)
                print(f"\n--- Step {step}: {len(tensors)} CUDA tensors, top {top_n} ---")
                for size, shape, dtype, rg in tensors[:top_n]:
                    print(f"  {size/1e6:.1f} MB  {shape}  {dtype}  requires_grad={rg}")
                print(f"  Total allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")

            debug_cuda_tensors(0) # Call this at the start to check initial memory usage

    def forward(self, D, U=None, samp=False):

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale)) # sample A from Gamma
        self.A = A.detach().clone()

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_fixed_patterns + self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale)) # sample P from Gamma
        self.P = P.detach().clone()

        A_total = torch.cat((self.fixed_A.T, A), dim=0)
        self.A_total = A_total.detach().clone() # save P_total

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P, A_total)  # (samples x genes)
        self.D_reconstructed = D_reconstructed.detach() # save D_reconstructed
        
        if U is not None:
            chi2 = torch.sum((D_reconstructed - D)**2 / U**2)
            self.chi2 = chi2.item()
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D, torch.log(theta))) - torch.sum(theta) - torch.sum(torch.lgamma(D + 1))
            self.pois = poisL.item()

            if chi2 < self.best_chisq: # if this is a better chi squared, save it
                self.best_chisq = chi2.item()
                self.best_chisq_iter = self.iter
                self.best_A = A.detach().clone()
                self.best_P = P.detach().clone()
                self.best_locA = self.loc_A.detach().clone()
                self.best_locP = self.loc_P.detach().clone()

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
                curr_A = A_total.detach()
                curr_P = P.detach()


                #correction = P.max(axis=0).values
                correction = curr_P.sum(axis=0)
                Pn = curr_P / correction
                An = curr_A * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn) 

                max_pat_per_gene = curr_A.argmax(dim=0)  # shape: (Gene,)
                A_binary = torch.zeros_like(curr_A)
                A_binary[max_pat_per_gene, torch.arange(curr_A.shape[1])] = 1
                self.markers_A += A_binary

                max_pat_per_samp = curr_P.argmax(dim=1)  # shape: (Samp,)
                P_binary = torch.zeros_like(curr_P)
                P_binary[torch.arange(curr_P.shape[0]), max_pat_per_samp] = 1
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
                device=torch.device('cpu'),
                init_method="random", # Options: (["mean", "svd", None]): TODOS
                debug=False
            ):

        super().__init__(num_samples, num_genes, num_patterns, use_chisq, use_pois, scale, NB_probs, device, init_method, debug=debug)

        ## This is the same as unsupervised but with a set of fixed P and A extended by this amount ##

        self.fixed_patterns = fixed_patterns # of shape samples x fixed patterns
        self.num_fixed_patterns = fixed_patterns.shape[1]

        print(f"################# Running Gamma-Negative Binomial Model with fixed samples #################")
        print(f"Fixing {self.num_fixed_patterns} patterns")


        #### Matrix A is patterns (supervised+unsupervised) x genes ####
        #self.loc_A = PyroParam(torch.rand(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.nonnegative)        
        if init_method == "random":
            print("Initializing loc_A randomly with positive constraint")
            self.loc_A = PyroParam(torch.rand(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) + 10e-8, constraint=dist.constraints.positive)
        elif init_method == "ones":
            print("Initializing loc_A to all ones with positive constraint")
            self.loc_A = PyroParam(torch.ones(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype), constraint=dist.constraints.positive)
        else:
            print(f"Initialization method {init_method} not recognized, defaulting to random")
            self.loc_A = PyroParam(torch.rand(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) + 10e-8, constraint=dist.constraints.positive)

        self.best_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.best_locA = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)
        self.sum_A2 = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype)

        #### Matrix P total is expanded ###
        self.sum_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)
        self.sum_P2 = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        #### Fixed patterns are samples x patterns ####
        self.fixed_P = torch.tensor(fixed_patterns, device=self.device,dtype=default_dtype) # tensor, not updatable


        self.markers_A = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_P = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Ascaled = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Pscaled = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)

        self.markers_Asoftmax = torch.zeros(self.num_fixed_patterns + self.num_patterns, self.num_genes, device=self.device, dtype=default_dtype) 
        self.markers_Psoftmax = torch.zeros(self.num_samples, self.num_fixed_patterns + self.num_patterns, device=self.device, dtype=default_dtype)


        print("Updated initialized tensors to reflect fixed samples")
        if debug:
            def debug_cuda_tensors(step, top_n=20):
                """Print the largest tensors currently allocated on CUDA."""
                tensors = []
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            tensors.append((obj.element_size() * obj.nelement(), obj.shape, obj.dtype, obj.requires_grad))
                    except:
                        pass
                tensors.sort(key=lambda x: x[0], reverse=True)
                print(f"\n--- Step {step}: {len(tensors)} CUDA tensors, top {top_n} ---")
                for size, shape, dtype, rg in tensors[:top_n]:
                    print(f"  {size/1e6:.1f} MB  {shape}  {dtype}  requires_grad={rg}")
                print(f"  Total allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")

            debug_cuda_tensors(0) # Call this at the start to check initial memory usage


    def forward(self, D, U=None, samp=False):

        self.iter += 1 # keep a running total of iterations

        # Nested plates for pixel-wise independence
        with pyro.plate("patterns", self.num_fixed_patterns + self.num_patterns, dim = -2):
            with pyro.plate("genes", self.num_genes, dim = -1):
                A = pyro.sample("A", dist.Gamma(self.loc_A, self.scale)) # sample A from Gamma
        self.A = A.detach().clone()

        # Nested plates for pixel-wise independence
        with pyro.plate("samples", self.num_samples, dim=-2):
            with pyro.plate("patterns_P", self.num_patterns, dim = -1):
                P = pyro.sample("P", dist.Gamma(self.loc_P, self.scale)) # sample P from Gamma
        self.P = P.detach().clone()

        P_total = torch.cat((self.fixed_P, P), dim=1)
        self.P_total = P_total.detach().clone() # save P_total

        # Matrix D_reconstucted is samples x genes; calculated as the product of P and A
        D_reconstructed = torch.matmul(P_total, A)  # (samples x genes)
        self.D_reconstructed = D_reconstructed.detach() # save D_reconstructed
        
        if U is not None:
            # Calculate chi squared
            chi2 = torch.sum((D_reconstructed-D)**2/U**2)
            self.chi2  = chi2.item()
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            self.pois  = poisL.item()

            if chi2 < self.best_chisq:
                self.best_chisq = chi2.item()
                self.best_chisq_iter = self.iter
                self.best_A = A.detach().clone()
                self.best_P = P.detach().clone()
                self.best_locA = self.loc_A.detach().clone()
                self.best_locP = self.loc_P.detach().clone()

            # Include chi squared loss in the model
            if self.use_chisq:
                pyro.factor("chi2_loss", -chi2)  # Pyro's way of adding custom terms to the loss

        if self.use_pois:
            # Error Model Poisson
            theta = self.D_reconstructed
            poisL = torch.sum(torch.multiply(D,torch.log(theta)))-torch.sum(theta)-torch.sum(torch.lgamma(D+1))
            # Addition to Elbow Loss - should make this at least as large as Elbow
            self.pois  = poisL.item()
            pyro.factor("pois.loss",10.*poisL)

        if samp:
            with torch.no_grad():
                curr_A = A.detach()
                curr_P_total = P_total.detach()
                #correction = P_total.max(axis=0).values
                correction = curr_P_total.sum(axis=0)
                Pn = curr_P_total / correction
                An = curr_A * correction.unsqueeze(1)
                self.sum_A += An
                self.sum_P += Pn
                self.sum_A2 += torch.square(An)
                self.sum_P2 += torch.square(Pn)


                max_pat_per_gene = curr_A.argmax(dim=0)
                A_binary = torch.zeros_like(curr_A)
                A_binary[max_pat_per_gene, torch.arange(curr_A.shape[1])] = 1
                self.markers_A += A_binary

                max_pat_per_samp = curr_P_total.argmax(dim=1)
                P_binary = torch.zeros_like(curr_P_total)
                P_binary[torch.arange(curr_P_total.shape[0]), max_pat_per_samp] = 1
                self.markers_P += P_binary

                max_pat_per_gene_scaled = An.argmax(dim=0)
                A_binaryscaled = torch.zeros_like(An)
                A_binaryscaled[max_pat_per_gene_scaled, torch.arange(An.shape[1])] = 1
                self.markers_Ascaled += A_binaryscaled

                max_pat_per_samp_scaled = Pn.argmax(dim=1)
                P_binaryscaled = torch.zeros_like(Pn)
                P_binaryscaled[torch.arange(Pn.shape[0]), max_pat_per_samp_scaled] = 1
                self.markers_Pscaled += P_binaryscaled

                sumPerPat = Pn.sum(dim=1)
                self.markers_Psoftmax += (Pn / sumPerPat.unsqueeze(1))

                sumPerGene = An.sum(dim=0)
                self.markers_Asoftmax += (An / sumPerGene)
       
        #def check_tensor(name, x):
        #    print(f"\n{name}")
        #    print("shape:", x.shape)
        #    print("finite:", torch.isfinite(x).all().item())
        #    print("nan:", torch.isnan(x).any().item())
        #    print("inf:", torch.isinf(x).any().item())

        #    finite_x = x[torch.isfinite(x)]
        #    if finite_x.numel() > 0:
        #        print("min:", finite_x.min().item())
        #        print("max:", finite_x.max().item())
        #        print("mean:", finite_x.mean().item())
        #    else:
        #        print("No finite values")

        #check_tensor("loc_A", self.loc_A)
        #check_tensor("loc_P", self.loc_P)
        #check_tensor("fixed_P", self.fixed_P)
        #check_tensor("A", A)
        #check_tensor("P", P)

        #check_tensor("D_reconstructed", D_reconstructed)
        #assert torch.isfinite(D_reconstructed).all(), "total_count has nan/inf"
        #assert (D_reconstructed >= 0).all(), "total_count has negative values"
        


        pyro.sample("D", dist.NegativeBinomial(D_reconstructed, probs=self.NB_probs).to_event(2), obs=D) 


def guide(D):
    pass

