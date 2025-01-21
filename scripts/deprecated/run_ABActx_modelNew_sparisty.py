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
import pyro.poutine as poutine
import scanpy as sc
import seaborn as sns
import torch
from pyro.infer.autoguide import \
    AutoNormal  # , AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

#from cogaps.guide import CoGAPSGuide
from models.model_new import GammaMatrixFactorization
from models.utils import generate_structured_test_data, generate_test_data

import random

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

#%% Plotting function
## KW TODO -- generalize for any data; move out of model definition
def plot_grid(patterns, coords, nrows, ncols, savename = None):
    fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*4, nrows*4))
    num_patterns = patterns.shape[1]
    x, y = coords['x'], coords['y']
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                p5 = np.percentile(patterns[:,i], 5)
                p95 = np.percentile(patterns[:,i], 95)
                pattern_min = np.min(patterns[:, i])
                pattern_max = np.max(patterns[:, i])
                alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))

                axes[r,c].scatter(x, -1*y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                i += 1
    if savename != None:
        plt.savefig(savename)

def plot_correlations(true_vals, inferred_vals, savename = None):
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

ABA_data = ad.read_h5ad('/disk/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad')
ISO_data = ABA_data[ABA_data.obsm['atlas']['Isocortex']]

#print((pd.DataFrame(ISO_data.X) == 0).sum().sum() / (ISO_data.shape[0]*ISO_data.shape[1]) * 100)
#print(ISO_data.X.max())
#print(ISO_data.X.mean())
#print(ISO_data.X.std())
#plt.hist(ISO_data.X.flatten(),bins=50)

ISO_data_nonzero = ISO_data.X[ISO_data.X != 0]
#print(ISO_data_nonzero.max())
#print(ISO_data_nonzero.mean())
#print(ISO_data_nonzero.std())
#plt.hist(ISO_data_nonzero,bins=50)


#%%
random.seed(123)
# Reduce sparsity
#ISO_data_reduce_sparsity = ISO_data.X
zero_indices = np.where(ISO_data.X == 0)
num_zeros_to_replace = int(len(zero_indices[0]) * 0)  # For example, replace 10% of the zeros
# Randomly sample the zero positions
random_zero_indices = np.random.choice(len(zero_indices[0]), size=num_zeros_to_replace, replace=False)
# Replace the selected zeros with random values sampled from normal distribution
# Sample the values from a normal distribution
sampled_values = np.random.normal(loc=ISO_data_nonzero.mean(), scale=ISO_data_nonzero.std(), size=num_zeros_to_replace)
# Round the sampled values to integers
sampled_values = np.round(sampled_values).astype(int)
# Ensure non-zero values by replacing any zero values with a small positive integer (e.g., 1)
sampled_values[sampled_values <= 0] = 1
# Replace the zeros with the sampled non-zero integers

for i in range(num_zeros_to_replace):
    x = zero_indices[0][i]
    y = zero_indices[1][i]
    sample_i = sampled_values[i]
    ISO_data.X[x][y] = sample_i

#ISO_data.X = ISO_data_reduce_sparsity

#%%
print((pd.DataFrame(ISO_data.X) == 0).sum().sum() / (ISO_data.shape[0]*ISO_data.shape[1]) * 100)
print(ISO_data.X.max())
print(ISO_data.X.mean())
print(ISO_data.X.std())
plt.hist(ISO_data.X.flatten(),bins=50)

ISO_data_nonzero = ISO_data.X[ISO_data.X != 0]
print(ISO_data_nonzero.max())
print(ISO_data_nonzero.mean())
print(ISO_data_nonzero.std())
plt.hist(ISO_data_nonzero,bins=50)

#%%
#D = torch.tensor(np.log1p(ISO_data.X)) ## LOG TRANSFORM DATA
D = torch.tensor(ISO_data.X) ## RAW COUNT DATA
#D = (D/(D.max()/3)).round()# test smaller integers; note gammapoisson needs integers
coords = ISO_data.obs.loc[:,['x','y']]
num_patterns = 12

if torch.backends.mps.is_available():
    device=torch.device('mps')
    print('using mps')
elif torch.cuda.is_available():
    device=torch.device('cuda')
    print('using cuda')
else:
    device=torch.device('cpu')
    print('using cpu')

# Move data to device
D = D.to(device)

# Create a DataLoader to handle batching and shuffling (not implemented yet)
#dataset = TensorDataset(D)
#dataloader = DataLoader(dataset, batch_size=100, shuffle=True)



##### RUN BELOW HERE
#%% Clear Pyro's parameter store
pyro.clear_param_store()

# Start timer
startTime = datetime.now()

# Define the number of optimization steps
num_steps = 10000

# Use the Adam optimizer
optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # try default

# Define the loss function
loss_fn = pyro.infer.Trace_ELBO()

#%% Instantiate the model
model = GammaMatrixFactorization(D.shape[1], num_patterns, D.shape[0],device=device)

# Draw model
#pyro.render_model(model, model_args=(D,),
#                 render_params=True,
#                 render_distributions=True,
#                 #render_deterministic=True,
#                 filename="updated_model_102224.pdf")


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
        plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, 4, 3, savename = None)
        writer.add_figure("loc_P", plt.gcf(), step)

        plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("loc_P_hist", plt.gcf(), step)

        plt.hist(pyro.param("loc_A").std(dim=0).detach().to('cpu').numpy(), bins=30)
        writer.add_figure("loc_A std (gene stds)", plt.gcf(), step)

        plt.hist(pyro.param("scale_D").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("scale_D_hist", plt.gcf(), step)

    if step % 100 == 0:
        print(f"Iteration {step}, ELBO loss: {loss}")

        D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
        plt.hist(D_reconstructed.flatten(), bins=30)
        writer.add_figure("D_reconstructed_his", plt.gcf(), step)

        #plot_grid(D_reconstructed, coords, 2, 3, savename = None)
        #writer.add_figure("D_reconstructed", plt.gcf(), step)

writer.flush()

#%% Retrieve the inferred parameters
savename = 'results/SVI_MCMC/Oct29_14-59-19_SVI_10k_scaleD20max_run3'

loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
loc_A.columns = ['Pattern_' + str(x+1) for x in loc_A.columns]
loc_A.index = ISO_data.var['gene_symbol_x']
loc_A.to_csv(savename + "loc_A.csv")

scale_A = pd.DataFrame(pyro.param("scale_A").detach().to('cpu').numpy()).T
scale_A.columns = ['Pattern_' + str(x+1) for x in scale_A.columns]
scale_A.index = ISO_data.var['gene_symbol_x']
scale_A.to_csv(savename + "scale_A.csv")

loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
loc_P.index = ISO_data.obs.index
loc_P.to_csv(savename + "loc_P.csv")

scale_P = pd.DataFrame(pyro.param("scale_P").detach().to('cpu').numpy())
scale_P.columns = ['Pattern_' + str(x+1) for x in scale_P.columns]
scale_P.index = ISO_data.obs.index
scale_P.to_csv(savename + "scale_P.csv")

# %%
