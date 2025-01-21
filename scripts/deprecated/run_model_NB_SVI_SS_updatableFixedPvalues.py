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
from pyro.optim import Adam
#from pyro.optim.multi import PyroMultiOptimizer
#from pyro.optim import Adam, SGD, MultiOptimizer
from pyro.infer.autoguide import \
    AutoNormal  # , AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from models.model_NB_cleaned_SS_updatableFixedPvalues import GammaMatrixFactorization, plot_grid, plot_correlations
from models.utils import generate_structured_test_data, generate_test_data

import random
from torch.utils.tensorboard import SummaryWriter

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



#%%

#######################################################
########### ALL USER DEFINED INPUTS HERE ##############
#######################################################

#data = ad.read_h5ad('/disk/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad')
data = ad.read_h5ad('/home/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') # samples x genes

data = data[data.obsm['atlas']['Isocortex']]
D = torch.tensor(data.X) ## RAW COUNT DATA

#atlas = data.obsm['atlas'].loc[:,['SS', 'MO', 'OLF','HIP','STR','layer 1','layer 2/3','layer 4','layer 5','layer 6']]*1
atlas = data.obsm['atlas'].loc[:,['layer 1','layer 2/3','layer 4','layer 5','layer 6']]*1
add_noise = False

coords = data.obs.loc[:,['x','y']]
coords['y'] = -1*coords['y']

num_patterns = 10 # Num EXTRA patterns
device = None # auto detect
NB_probs = None # use default of 1 - sparsity

#outputDir = '/disk/kyla/projects/pyro_NMF/results/20241218_SSlayers_updateFixed/'
outputDir = '/home/kyla/projects/pyro_NMF/results/SS_recovery_analysis/'

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

savename = 'SSlayers_all_learnFixed_lre-5_drop.3'

tensorboard_identifier = 'SSlayers_all_learnFixed_lre-5_drop.3'

num_steps = 10000 # Define the number of optimization steps

plot_dims = [3, 5]

# Define the fraction of values to drop (30%)
drop_ones = True
np.random.seed(42)  # For reproducibility
drop_fraction = 0.3

loss_fn = pyro.infer.Trace_ELBO() # Define the loss function
draw_model = None # None or name for output file

#%%
# ADD GAMMA NOISE
atlas_matrix = atlas.copy()

if add_noise:
    # Gamma distribution parameters for added noise
    alpha_0, beta_0 = 0.5, 0.1  # Small noise for 0s
    alpha_1, beta_1 = 5.0, 0.5  # Larger noise for 1s

    # Adding Gamma noise to each element
    gamma_noise = np.zeros_like(atlas_matrix, dtype=float)
    gamma_noise[atlas_matrix == 0] = np.random.gamma(alpha_0, beta_0, size=(atlas_matrix == 0).sum().sum())
    gamma_noise[atlas_matrix == 1] = np.random.gamma(alpha_1, beta_1, size=(atlas_matrix == 1).sum().sum())

    # Resultant matrix after adding noise
    transformed_matrix = atlas_matrix + gamma_noise
    transformed_matrix.to_csv(outputDir + 'atlas_noisy.csv')
    transformed_matrix = transformed_matrix.to_numpy()
elif drop_ones:
    # Process each column
    for col in range(atlas_matrix.shape[1]):
        # Find indices of 1s in the column
        ones_indices = np.where(atlas_matrix.iloc[:, col] == 1)[0]
        # Randomly select 30% of the 1s to drop
        drop_count = int(len(ones_indices) * drop_fraction)
        drop_indices = np.random.choice(ones_indices, size=drop_count, replace=False)
        # Set the selected indices to 0
        atlas_matrix.iloc[drop_indices, col] = 0
    transformed_matrix = atlas_matrix.to_numpy()
    print(atlas.sum())
    print(atlas_matrix.sum())
else:
    transformed_matrix = atlas_matrix.to_numpy()
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
    print(f"Data is {percZeros:.2f}% sparse")


writer = SummaryWriter(comment = tensorboard_identifier)


# Instantiate the model
model = GammaMatrixFactorization(D.shape[1], D.shape[0], num_patterns, fixed_patterns=transformed_matrix, NB_probs = NB_probs, device=device)


def per_param_callable(param_name):
    if param_name == 'fixed_P':
        return {"lr": 1e-5, "eps":1e-08}
    else:
        return {"lr": 0.1, "eps":1e-08}

optimizer = pyro.optim.Adam(per_param_callable)


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
        writer.add_scalar("Loss/train", loss, step)
        writer.flush()

        losses.append(loss)
        steps.append(step)

    if step % 50 == 0:
        plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
        writer.add_figure("loc_P", plt.gcf(), step)

        plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("loc_P_hist", plt.gcf(), step)

        plot_grid(model.P_total.detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
        writer.add_figure("P_total", plt.gcf(), step)

        plot_grid(pyro.param("fixed_P").detach().to('cpu').numpy(), coords, 2, 3, savename = None)
        writer.add_figure("fixed_P", plt.gcf(), step)

        plt.hist(pyro.param("fixed_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("fixed_P_hist", plt.gcf(), step)

        plt.hist(model.P_total.detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("P_total_hist", plt.gcf(), step)

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

endTime = datetime.now()

# Save the inferred parameters
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#savename = '/disk/kyla/projects/pyro_NMF/results/scaleD_constraint_comparison/' + identifier + '/' + 'ISO_n12'+ identifier
result_anndata = data.copy()

loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
#loc_P.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
loc_P.index = result_anndata.obs.index
result_anndata.obsm['loc_P'] = loc_P

scale_P = pd.DataFrame(pyro.param("scale_P").detach().to('cpu').numpy())
#scale_P.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
scale_P.columns = ['Pattern_' + str(x+1) for x in scale_P.columns]
scale_P.index = result_anndata.obs.index
result_anndata.obsm['scale_P'] = scale_P

total_P = pd.DataFrame(model.P_total.detach().to('cpu').numpy())
total_P.columns = list(atlas_matrix.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
total_P.index = result_anndata.obs.index
result_anndata.obsm['P_total'] = total_P

loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
loc_A.columns = list(atlas_matrix.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
loc_A.index = result_anndata.var.index
result_anndata.varm['loc_A'] = loc_A

scale_A = pd.DataFrame(pyro.param("scale_A").detach().to('cpu').numpy()).T
scale_A.columns = list(atlas_matrix.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
scale_A.index = result_anndata.var.index # need names to match anndata names
result_anndata.varm['scale_A'] = scale_A

loc_D = pd.DataFrame(model.D_reconstructed.detach().cpu().numpy())
loc_D.index = result_anndata.obs.index
loc_D.columns = result_anndata.var.index # need names to match anndata names
result_anndata.layers['loc_D'] = loc_D

#plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_loc_P.pdf")

result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])
result_anndata.obsm['atlas_used'] = atlas_matrix

fixed_P = pd.DataFrame(pyro.param("fixed_P").detach().to('cpu').numpy())
#loc_P.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
fixed_P.columns = atlas_matrix.columns
fixed_P.index = result_anndata.obs.index
result_anndata.obsm['fixed_P'] = fixed_P


result_anndata.write_h5ad(outputDir + savename + '.h5ad')

writer.flush()

#
# %%
