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

from models.model_NB import GammaMatrixFactorization
from models.utils import generate_structured_test_data, generate_test_data

import random
from torch.utils.tensorboard import SummaryWriter

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


#%% Import data; set parameters

ABA_data = ad.read_h5ad('/disk/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad')
ISO_data = ABA_data[ABA_data.obsm['atlas']['Isocortex']]

percZeros = (pd.DataFrame(ISO_data.X) == 0).sum().sum() / (ISO_data.shape[0]*ISO_data.shape[1])
D_NegBinomial_probs = 1-percZeros

tensorboard_identifier = 'NB_probs' + str(round(D_NegBinomial_probs*100))
writer = SummaryWriter(comment = tensorboard_identifier)

#%%
D = torch.tensor(ISO_data.X) ## RAW COUNT DATA
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
model = GammaMatrixFactorization(D.shape[1], num_patterns, D.shape[0], D_NegBinomial_probs = D_NegBinomial_probs, device=device)

# Draw model
#pyro.render_model(model, model_args=(D,),
#                 render_params=True,
#                 render_distributions=True,
#                 #render_deterministic=True,
#                 filename="updated_model_120324_NB.pdf")


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

        plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("loc_A_hist", plt.gcf(), step)

        plt.hist(pyro.param("scale_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("scale_P_hist", plt.gcf(), step)

        plt.hist(pyro.param("scale_A").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("scale_A_hist", plt.gcf(), step)

    if step % 100 == 0:

        print(f"Iteration {step}, ELBO loss: {loss}")

        D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
        plt.hist(D_reconstructed.flatten(), bins=30)
        writer.add_figure("D_reconstructed_his", plt.gcf(), step)

writer.flush()

#