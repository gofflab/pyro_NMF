#!/usr/bin/env -S python3 -u
#%%
import os
from datetime import datetime

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.optim
import pyro.poutine as poutine
import scanpy as sc
import seaborn as sns
import torch
from pyro.infer.autoguide import \
    AutoNormal  # , AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

#from cogaps.guide import CoGAPSGuide
from models.model import ProbNMFModel
from models.utils import generate_structured_test_data, generate_test_data

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

#%% Plotting function
## designed in mind for the NSF toy data
## TODO move to utils -- tried but got error trying to import from utils
def plot_grid(patterns, coords, nrows, ncols, savename = None):
    fig, axes = plt.subplots(nrows,ncols)
    num_patterns = patterns.shape[1]
    #print(patterns.shape)
    #print(coords['x'].sort_values().head(40))
    x, y = coords['x'], coords['y']
    # Determine the extent of the grid
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    resolution = 0.114286
    x_grid = np.arange(x_min, x_max + resolution, resolution)
    y_grid = np.arange(y_min, y_max + resolution, resolution)

    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                # Initialize an empty grid (NaN-filled)
                grid = np.full((len(y_grid), len(x_grid)), np.nan)

                # Map the patterns to the grid; find nearest grid indices for each x, y pair
                ix = np.searchsorted(x_grid, x) - 1
                iy = np.searchsorted(y_grid, y) - 1

                for j in range(len(patterns[:,i])):
                    if 0 <= ix[j] < len(x_grid) and 0 <= iy[j] < len(y_grid):
                        grid[iy[j], ix[j]] = patterns[j,i]

                axes[r,c].imshow(grid, cmap='viridis')

                # or use scatter plot
                #axes[r,c].scatter(coords['x'], coords['y'], c=patterns[:,i], cmap='viridis', s = 5)
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

writer = SummaryWriter()

## Load in NSF toy data
S1 = sc.read_h5ad('S1.h5ad')
S1 = S1[:,:80] # based on NSF demo.ipynb
P_true = pd.DataFrame(S1.obsm['spfac'])
P_true.columns = ['True_' + str(i) for i in range(1, P_true.shape[1]+1)]
D = torch.tensor(pd.DataFrame(S1.layers['counts']).values) # D should be sample by genes
coords = pd.DataFrame(S1.obsm['spatial'])
coords.columns = ['x', 'y']

num_patterns = 4

#plot_grid(A_true.values, coords, 2, 2, savename = 'patterns_heatmap.png')

# Set device
# if torch.backends.cuda.is_available():
#     device=torch.device('cuda')
if torch.backends.mps.is_available():
    device=torch.device('mps')
    print('using mps')
elif torch.cuda.is_available():
    device=torch.device('cuda')
    print('using cuda')
else:
    device=torch.device('cpu')
    print('using cpu')
device=torch.device('cpu')

#device=torch.device('cpu')

# Move data to device
D = D.to(device)

# Create a DataLoader to handle batching and shuffling (not implemented yet)
dataset = TensorDataset(D)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

##### RUN BELOW HERE
#%% Clear Pyro's parameter store
pyro.clear_param_store()

# Start timer
startTime = datetime.now()

# Define the number of optimization steps
num_steps = 500

# Use the Adam optimizer
#optimizer = pyro.optim.Adam({"lr": 0.05})
optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # try default

# Define the loss function
loss_fn = pyro.infer.Trace_ELBO()

#%% Instantiate the model
#model = CoGAPSModel(D, num_patterns, device=device)
model = ProbNMFModel(D, num_patterns, device=device, init_method=None)


#%% Instantiate the guide
guide = AutoNormal(model)

#%% Define the inference algorithm
svi = pyro.infer.SVI(model=model,
                    guide=guide,
                    optim=optimizer,
                    loss=loss_fn)


#%% Run inference
for step in range(num_steps):
    loss = svi.step(D)
    if step % 10 == 0:
        writer.add_scalar("Loss/train", loss, step)
        writer.flush()
    if step % 50 == 0:
        plot_grid(pyro.param("P_mean").detach().to('cpu').numpy(), coords, 2, 2, savename = None)
        writer.add_figure("P_mean", plt.gcf(), step)

        plt.hist(pyro.param("P_mean").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("P_mean_hist", plt.gcf(), step)

        plt.hist(pyro.param("A_mean").std(dim=0).detach().to('cpu').numpy(), bins=30)
        writer.add_figure("A_mean std (gene stds)", plt.gcf(), step)

        plt.hist(pyro.param("P_mean").std(dim=1).detach().to('cpu').numpy(), bins=30)
        writer.add_figure("P_mean std (gene stds)", plt.gcf(), step)

    if step % 100 == 0:
        print(f"Iteration {step}, ELBO loss: {loss}")

#%% Retrieve the inferred parameters
#A_scale = pyro.param("A_scale").detach().to('cpu').numpy()
#P_scale = pyro.param("P_scale").detach().to('cpu').numpy()
A_mean = pyro.param("A_mean").detach().to('cpu').numpy()
P_mean = pyro.param("P_mean").detach().to('cpu').numpy()

#pd.DataFrame(A_scale).to_csv('A_scale.csv')
#pd.DataFrame(P_scale).to_csv('P_scale.csv')
pd.DataFrame(A_mean).to_csv('A_mean.csv')
pd.DataFrame(P_mean).to_csv('P_mean.csv')


# Print the shapes of the inferred parameters
print("Inferred A shape:", A_mean.shape)
print("Inferred P shape:", P_mean.shape)

# End timer
print("Time taken:")
print(datetime.now() - startTime)

# Plot the inferred patterns and true patterns
plot_grid(P_true.values, coords, 2, 2, savename = 'True_P.png')
plot_grid(P_mean, coords, 2, 2, savename = 'Inferred_P.png')

# Save outputs
plt.figure()
plt.hist(P_true.values.flatten(), bins=30)
plt.savefig('P_true_hist.png')

plt.figure()
plt.hist(P_mean.flatten(), bins=30)
plt.savefig('P_mean_hist.png')

# Check inferred parameters against the true parameters
plot_correlations(P_true.values, P_mean, 'P_mean')
writer.add_figure("P_mean_correlations", plt.gcf(), step)

# # %%
# writer.flush()