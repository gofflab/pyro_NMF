#!/usr/bin/env -S python3 -u
#%%
import os
from datetime import datetime

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import pyro
import pyro.optim
import pyro.poutine as poutine
import scanpy as sc
import seaborn as sns
import torch
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from sklearn.decomposition import PCA
from cogaps.utils import initialize_inducing_points_with_pca

#from nsf.guide import CoGAPSGuide
from cogaps.nsf import NSFModel
#from cogaps.utils import generate_structured_test_data, generate_test_data

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

#%% Plotting function
## designed in mind for the NSF toy data
## TODO move to utils -- tried but got error trying to import from utils
def plot_grid(patterns, coords, nrows, ncols, savename = None):
    fig, axes = plt.subplots(nrows,ncols)
    num_patterns = patterns.shape[1]
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                axes[r,c].scatter(coords['x'], coords['y'], c=patterns[:,i], cmap='viridis', s = 5)
                i += 1
    if savename != None:
        plt.savefig(savename)

def plot_correlations(true_vals, inferred_vals, savename):
    true_vals_df = pd.DataFrame(true_vals)
    true_vals_df.columns = ['True_' + str(x) for x in true_vals_df.columns]

    inferred_vals_df = pd.DataFrame(inferred_vals)
    inferred_vals_df.columns = ['Inferred_' + str(x) for x in inferred_vals_df.columns]

    correlations = true_vals_df.merge(inferred_vals_df, left_index=True, right_index=True).corr().round(2)

    plt.figure()
    sns.clustermap(correlations.iloc[:true_vals.shape[1], true_vals.shape[1]:], annot=False, cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
    if savename != None:
        plt.savefig(savename + '_correlations.png')


#%% Import tensorboard & setup writer
from torch.utils.tensorboard import SummaryWriter



#writer = SummaryWriter()

## Load in NSF toy data
S1 = sc.read_h5ad('S1.h5ad')
S1 = S1[:,:80] # based on NSF demo.ipynb
A_true = pd.DataFrame(S1.obsm['spfac'])
A_true.columns = ['True_' + str(i) for i in range(1, A_true.shape[1]+1)]
D = torch.tensor(pd.DataFrame(S1.layers['counts']).values)
print(D.shape)
spatial_coords = pd.DataFrame(S1.obsm['spatial'])
spatial_coords.columns = ['x', 'y']
spatial_coords = torch.tensor(spatial_coords.to_numpy(),dtype=torch.get_default_dtype())
print(spatial_coords.shape)
num_spatial_factors = 3
num_nonspatial_factors = 1

#plot_grid(A_true.values, coords, 2, 2, savename = 'patterns_heatmap.png')

# Set device
# if torch.backends.cuda.is_available():
#     device=torch.device('cuda')
if torch.backends.mps.is_available():
    device=torch.device('mps')
else:
    device=torch.device('cpu')
    
device = torch.device('cpu')

#%% Clear Pyro's parameter store
pyro.clear_param_store()

# Start timer
startTime = datetime.now()

# Define the number of optimization steps
num_steps = 1000

# Use the Adam optimizer
optimizer = pyro.optim.Adam({"lr": 0.05})

# Define the loss function
loss_fn = pyro.infer.Trace_ELBO()

# Initialize inducing points using PCA
num_inducing_points = 32
#inducing_points = initialize_inducing_points_with_pca(D, spatial_coords, num_spatial_factors+num_nonspatial_factors, num_inducing_points)
inducing_points = initialize_inducing_points_with_pca(D, spatial_coords, num_spatial_factors + num_nonspatial_factors, num_inducing_points)

print(inducing_points.shape)
# Move data to device
D = D.to(device)

#%% Instantiate the model
#model = CoGAPSModel(D, num_patterns, device=device)
model = NSFModel(num_spatial_factors, num_nonspatial_factors, D.size(1), D.size(0), spatial_coords, inducing_points)

#%%
# Inspect model
# pyro.render_model(model, model_args=(D,),
#                 render_params=True,
#                 render_distributions=True,
#                 #render_deterministic=True,
#                 filename="NSF_model.pdf")

#%% Logging model
#model.eval()
#writer.add_graph(model,D)
#model.train()

#%% Instantiate the guide
# guide = CoGAPSGuide(D, num_patterns, device=device)
guide = AutoDiagonalNormal(model)


#%% Define the inference algorithm
svi = pyro.infer.SVI(model=model,
                    guide=guide,
                    optim=optimizer,
                    loss=loss_fn)

# # #%% Trace
# # #trace = poutine.trace(model(D)).get_trace()
# # #trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
# # #print(trace.format_shapes())

#%% Run inference
for step in range(num_steps):
    loss = svi.step(D)
    #if step % 10 == 0:
        #writer.add_scalar("Loss/train", loss, step)
        #writer.flush()
    #if step % 50 == 0:
    #    plot_grid(pyro.param("A_mean").detach().to('cpu').numpy(), coords, 2, 2, savename = None)
    #    writer.add_figure("A_mean", plt.gcf(), step)
    if step % 100 == 0:
        print(f"Iteration {step}, ELBO loss: {loss}")

# #%% Retrieve the inferred parameters
# A_scale = pyro.param("A_scale").detach().to('cpu').numpy()
# P_scale = pyro.param("P_scale").detach().to('cpu').numpy()
# A_mean = pyro.param("A_mean").detach().to('cpu').numpy()
# P_mean = pyro.param("P_mean").detach().to('cpu').numpy()

# pd.DataFrame(A_scale).to_csv('A_scale.csv')
# pd.DataFrame(P_scale).to_csv('P_scale.csv')
# pd.DataFrame(A_mean).to_csv('A_mean.csv')
# pd.DataFrame(P_mean).to_csv('P_mean.csv')


# # Print the shapes of the inferred parameters
# print("Inferred A shape:", A_mean.shape)
# print("Inferred P shape:", P_mean.shape)

# # End timer
# print("Time taken:")
# print(datetime.now() - startTime)

# # Plot the inferred patterns and true patterns
# plot_grid(A_true.values, coords, 2, 2, savename = 'True_A.png')
# plot_grid(A_mean, coords, 2, 2, savename = 'Inferred_A.png')

# # Save outputs
# plt.figure()
# plt.hist(A_true.values.flatten(), bins=30)
# plt.savefig('A_true_hist.png')

# plt.figure()
# plt.hist(A_mean.flatten(), bins=30)
# plt.savefig('A_mean_hist.png')

# # Check inferred parameters against the true parameters
# plot_correlations(A_true.values, A_mean, 'A_mean')
# writer.add_figure("A_mean_correlations", plt.gcf(), step)

# # # %%
# # writer.flush()
# %%
