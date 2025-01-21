import torch
import pyro
import pyro.contrib.gp as gp
from pyro.contrib.gp.models import SparseGPRegression
from pyro.contrib.gp.kernels import Matern32
from pyro.nn import PyroModule, PyroSample, PyroParam
import pyro.distributions as dist

class NSFModel(PyroModule):
    def __init__(self, num_spatial_factors, num_nonspatial_factors, num_genes, num_features, spatial_coords, inducing_points):
        super().__init__()

        self.num_spatial_factors = num_spatial_factors
        self.num_nonspatial_factors = num_nonspatial_factors
        self.num_genes = num_genes
        self.num_features = num_features
        self.spatial_coords = spatial_coords
        self.inducing_points = inducing_points
        
        # Define the GP kernel
        self.kernel = Matern32(input_dim=spatial_coords.size(1))

        # Initialize inducing points
        self.inducing_points = inducing_points

        # Parameters for non-spatial factors and their loadings
        self.nonspatial_factors = PyroParam(torch.randn(self.num_features, self.num_nonspatial_factors))  # Transpose here
        self.loadings = PyroParam(torch.rand(num_genes, num_spatial_factors + num_nonspatial_factors), constraint=dist.constraints.positive)
        
    def model(self, D):
        # GP model outputs
        f_loc, f_var = self.gp_model(D)

        # Reshape f_loc and f_var for multiple spatial factors
        f_loc_expanded = f_loc.unsqueeze(1).expand(-1, self.num_spatial_factors)
        f_var_expanded = f_var.sqrt().unsqueeze(1).expand(-1, self.num_spatial_factors)

        # Sample spatial factors with the correct shape
        f_spatial = pyro.sample(
            "f_spatial", 
            dist.Normal(f_loc_expanded, f_var_expanded).to_event(1)
        )

        # Combine spatial and non-spatial factors
        # Ensure loadings are transposed if needed to match dimensionality for matrix multiplication
        combined_factors = torch.matmul(torch.cat((f_spatial,self.nonspatial_factors),1), self.loadings.T)
        lambda_ = torch.exp(combined_factors)  # Poisson rate parameter
        
        with pyro.plate("data", D.shape[0]):
            obs = pyro.sample("obs", dist.Poisson(lambda_).to_event(0), obs=D)
        return obs
    
    def forward(self, D):
        return self.model(D)
    
    def gp_model(self, D):
        # Initialize the GP model with spatial coordinates as the training data features
        gp_model = gp.models.SparseGPRegression(X=D, y=None, kernel=self.kernel, Xu=self.inducing_points, noise=torch.tensor(1.))
        
        # Condition the GP model on the observed data
        conditioned_gp_model = pyro.condition(gp_model.model, data={"y": gp_model.y})

        # Call the conditioned GP model
        return conditioned_gp_model()