#!/usr/bin/env -S python3 -u
#%%
import os
from datetime import datetime
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import pyro
import pyro.optim
import scanpy as sc
import seaborn as sns
import torch
from pyro.infer.autoguide import \
    AutoNormal  # , AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from pygam import LinearGAM, s
from pyroNMF.models.gamma_NB_base import Gamma_NegBinomial_base
import pygam
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn_extra.cluster import PrincipalCurve
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA
from scipy.interpolate import splprep, splev, splder
from scipy.spatial.distance import cdist


#from cogaps.utils import generate_structured_test_data, generate_test_data

import random

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



#%%

#######################################################
########### ALL USER DEFINED INPUTS HERE ##############
#######################################################

#### input data here, should be stored in anndata.X layer
# If you have a numpy array of data try : data = ad.AnnData(array_data) # Kyla untested
#data = ad.read_h5ad('/disk/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') # samples x genes
data = ad.read_h5ad('/home/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') # samples x genes
data = data[data.obsm['atlas']['Isocortex']]
coords = data.obs.loc[:,['x','y']] # samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset


# 1️⃣ **Apply PCA to find the main direction of the data (Principal Component)**
pca = PCA(n_components=2)
pca.fit(coords)  # Fit PCA to the 2D data

# The first principal component (line of maximum variance)
principal_component = pca.components_[0]  # First principal component (the direction of the curve)

# Project the data along the principal component (1D projection)
projected = coords.dot(principal_component)
coords['projected'] = projected


# 2️⃣ **Fit a B-Spline to the projected 1D data**
# Sorting data by the 1D projection
#sorted_indices = np.argsort(projected)
sorted_coords = coords.sort_values('projected')

# Fit a smooth B-spline to the 1D data (a cubic spline)
tck, u = splprep([sorted_coords.iloc[:, 0], sorted_coords.iloc[:, 1]], k=3, s=10000)  # Use smoothing factor `s`

# 3️⃣ **Evaluate the spline at a finer resolution**
curve_points = np.column_stack(splev(np.linspace(0, 1, 300), tck))  # 300 points for a smooth curve
#e_points = np.column_stack(splev(np.linspace(0, 1, 300), tck))  # Evaluate the curve at 300 points
plt.scatter(curve_points[:,0], curve_points[:,1])


# Step 4: Compute the distance from each point in the original data to the fitted curve
# Get the coordinates of the curve
curve_x, curve_y = curve_points[:, 0], curve_points[:, 1]


distances = cdist(coords[['x', 'y']], np.column_stack((curve_x, curve_y)))
min_distances = np.min(distances, axis=1)

u_eval = np.argmin(distances, axis=1) / len(curve_x)  # Corresponding parameter u values
tangent_vectors = np.array([splev(u_val, tck, der=1) for u_val in u_eval])

# Step 6: Compute the cross product to determine the sign of the distance (above/below the curve)
# Vector from the closest point on the curve to the original point
closest_points = curve_points[u_eval.astype(int)]

# Vectors from the curve points to the original points
vector_to_curve = coords[['x', 'y']].values - closest_points

# Compute the cross product of the tangent vector and the vector to the curve
cross_products = np.cross(tangent_vectors, vector_to_curve)

# Step 7: Assign signed distances based on the cross product (positive/negative)
signed_distances = np.sign(cross_products) * min_distances

# Add the signed distances to the original dataframe
coords['signed_distance_to_curve'] = signed_distances

# Compute pairwise distances between each original point and each point on the curve
#distances = cdist(coords[['x', 'y']], np.column_stack((curve_x, curve_y)))

# Find the closest point on the curve for each original point (minimum distance)
#min_distances = np.min(distances, axis=1)

# Add the distances to the original dataframe
#coords['distance_to_curve'] = min_distances


# 3️⃣ **Plot the Original Data and the 1D Curve**
plt.figure(figsize=(8, 6))
#plt.scatter(coords.iloc[:, 0], coords.iloc[:, 1],  alpha=0.5, c=coords['signed_distance_to_curve'], label="Original Points")  # Original data points
plt.scatter(coords['x'], coords['y'], c=coords['signed_distance_to_curve'], cmap='coolwarm', label='Original Points')

plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-', linewidth=2, label="1D Principal Curve")  # 1D Curve
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Original Coordinates and Fitted 1D Principal Curve")
plt.show()


#%%
D = torch.tensor(data.X) ## RAW COUNT DATA

# Calculate mean and variance of the observed data
#mean_obs = torch.mean(D).item()
#var_obs = torch.var(D, unbiased=False).item()

# Calculate `probs` from the observed data (using the formula above)
#probs = mean_obs / (var_obs + mean_obs)

#### coords should be two columns named x and y
coords = data.obs.loc[:,['x','y']] # samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset

num_patterns = 20 # select num patterns
num_steps = 10000 # Define the number of optimization steps

device = None # options ['cpu', 'cuda', 'mps', etc]; if None: auto detect cpu vs gpu vs mps
NB_probs = 0.99 # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D

#### output parameters
outputDir = '/home/kyla/projects/pyro_NMF/results/checkProbs/'
savename = 'NBprobs0.99_2' # output anndata will be saved in outputDir/savename.h5ad

useTensorboard = True
tensorboard_identifier = 'NBprobs0.99_2' # key added to tensorboard output name

plot_dims = [5, 4] # rows x columns should be > num patterns; this is for plotting


#### model parameters
optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # Use the Adam optimizer
loss_fn = pyro.infer.Trace_ELBO() # Define the loss function
draw_model = None # None or name for output file to create pdf diagram of pyro model



#%%
# Clear Pyro's parameter store
pyro.clear_param_store()

if device == None:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
#device = device

print(f"Using {device}")
D = D.to(device)

if NB_probs == None:
    percZeros = (D == 0).sum().sum() / (D.shape[0]*D.shape[1])
    NB_probs = 1-percZeros
    print(f"Data is {percZeros*100:.2f}% sparse")

if useTensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment = tensorboard_identifier)


# Instantiate the model
model = Gamma_NegBinomial_base(D.shape[1], D.shape[0], num_patterns, NB_probs = NB_probs, device=device)

# Draw model
if draw_model != None:
    pyro.render_model(model, model_args=(D,),
                     render_params=True,
                     render_distributions=True,
                     #render_deterministic=True,
                     filename=draw_model)


# Instantiate the guide
guide = AutoNormal(model)

# Define the inference algorithm
svi = pyro.infer.SVI(model=model,
                    guide=guide,
                    optim=optimizer,
                    loss=loss_fn)


# Start timer
startTime = datetime.now()

steps = []
losses = []

# Run inference
for step in range(1,num_steps+1):
    loss = svi.step(D)

    if step % 10 == 0:
        if useTensorboard:
            writer.add_scalar("Loss/train", loss, step)
            writer.flush()

        #for name, param in pyro.get_param_store().items():
        #    if param.grad is not None:
        #        print(name, param.grad.norm().item())
        #        #print(name, param.grad_fn)
        #        #print(name, param.requires_grad)
        #    else:
        #        print(name, ' None')
        #        #print(name, param.grad_fn)
        #        #print(name, param.requires_grad)

        losses.append(loss)
        steps.append(step)

    if step % 50 == 0:
        if useTensorboard:
            model.plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
            writer.add_figure("loc_P", plt.gcf(), step)

            plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("loc_P_hist", plt.gcf(), step)

            plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("loc_A_hist", plt.gcf(), step)

            plt.hist(pyro.param("scale_P").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("scale_P_hist", plt.gcf(), step)

            plt.hist(pyro.param("scale_A").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("scale_A_hist", plt.gcf(), step)

            plt.hist(model.A.detach().cpu().numpy().flatten(), bins=30)
            writer.add_figure("A_hist", plt.gcf(), step)

            plt.hist(model.P.detach().cpu().numpy().flatten(), bins=30)
            writer.add_figure("A_hist", plt.gcf(), step)
            
            meanA = model.A.detach().cpu().numpy().mean()
            writer.add_scalar("Mean A", meanA, step)

            meanP = model.P.detach().cpu().numpy().mean()
            writer.add_scalar("Mean P", meanP, step)


    if step % 100 == 0:

        print(f"Iteration {step}, ELBO loss: {loss}")

        if useTensorboard:
            D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
            plt.hist(D_reconstructed.flatten(), bins=30)
            writer.add_figure("D_reconstructed_his", plt.gcf(), step)

endTime = datetime.now()
print('Runtime: '+ str(round((endTime - startTime).total_seconds())) + ' seconds')

#%%
# Save the inferred parameters
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#savename = '/disk/kyla/projects/pyro_NMF/results/scaleD_constraint_comparison/' + identifier + '/' + 'ISO_n12'+ identifier
result_anndata = data.copy()

loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
loc_A.columns = ['Pattern_' + str(x+1) for x in loc_A.columns]
loc_A.index = result_anndata.var.index
result_anndata.varm['loc_A'] = loc_A
print("Saving loc_A in anndata.varm['loc_A']")

scale_A = pd.DataFrame(pyro.param("scale_A").detach().to('cpu').numpy()).T
scale_A.columns = ['Pattern_' + str(x+1) for x in scale_A.columns]
scale_A.index = result_anndata.var.index # need names to match anndata names
result_anndata.varm['scale_A'] = scale_A
print("Saving scale_A in anndata.varm['scale_A']")

loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
loc_P.index = result_anndata.obs.index
result_anndata.obsm['loc_P'] = loc_P
print("Saving loc_P in anndata.obsm['loc_P']")

scale_P = pd.DataFrame(pyro.param("scale_P").detach().to('cpu').numpy())
scale_P.columns = ['Pattern_' + str(x+1) for x in scale_P.columns]
scale_P.index = result_anndata.obs.index
result_anndata.obsm['scale_P'] = scale_P
print("Saving scale_P in anndata.obsm['scale_P']")

loc_D = pd.DataFrame(model.D_reconstructed.detach().cpu().numpy())
loc_D.index = result_anndata.obs.index
loc_D.columns = result_anndata.var.index # need names to match anndata names
result_anndata.layers['loc_D'] = loc_D
print("Saving loc_D in anndata.layers['loc_D']")

model.plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_loc_P.pdf")

result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])

result_anndata.write_h5ad(outputDir + savename + '.h5ad')

if useTensorboard:
    writer.flush()

#
# %%
