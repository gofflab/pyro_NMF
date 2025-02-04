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

from models.deprecated.model_NB_cleaned import GammaMatrixFactorization, plot_grid, plot_correlations
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
data_full = ad.read_h5ad('/home/kyla/data/human_sobj_transformed_projected.h5ad') # samples x genes
print(data_full.shape) #(74096, 36601)
data_full = data_full[data_full.obs.loc[:,'patient'] != '2099']
print(data_full.shape) #(67029, 36601)
patients = ['934', '1061', '1154', '1055', '4544', '4434', '4276', '1210',
       '4389', '1208', '21', '2099']
#i = 0
#data = data[data.obs['patient'] == patients[i]]
#data = data[data.obsm['atlas']['Isocortex']]
sc.pp.highly_variable_genes(data_full, flavor='seurat_v3', n_top_genes=2000)
data = data_full[:, data_full.var['highly_variable']].copy()

D = torch.tensor(data.X.todense()) ## RAW COUNT DATA

#### coords should be two columns named x and y
#coords = data.obs.loc[:,['x','y']] # samples x 2
#coords['y'] = -1*coords['y'] # specific for this dataset

num_patterns = 15 # select num patterns
num_steps = 10000 # Define the number of optimization steps

device = None # options ['cpu', 'cuda', 'mps', etc]; if None: auto detect cpu vs gpu vs mps
NB_probs = None # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D

#### output parameters
outputDir = '/home/kyla/projects/pyro_NMF/results/cerbellum/'
#savename = 'CB_SVI_n15_patient' + patients[i] # output anndata will be saved in outputDir/savename.h5ad
savename = 'CB_SVI_n15_HVG'

useTensorboard = True
#tensorboard_identifier = 'CB_SVI_n15_patient' + patients[i] # key added to tensorboard output name
tensorboard_identifier = 'CB_SVI_n15_HVG'

#plot_dims = [5, 4] # rows x columns should be > num patterns; this is for plotting


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
model = GammaMatrixFactorization(D.shape[1], num_patterns, D.shape[0], NB_probs = NB_probs, device=device)

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
                #print(name, param.grad_fn)
                #print(name, param.requires_grad)
        #    else:
        #        print(name, ' None')
                #print(name, param.grad_fn)
                #print(name, param.requires_grad)

        losses.append(loss)
        steps.append(step)

    if step % 50 == 0:
        if useTensorboard:
            #plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
            #writer.add_figure("loc_P", plt.gcf(), step)

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

#%%
#savename = '/disk/kyla/projects/pyro_NMF/results/scaleD_constraint_comparison/' + identifier + '/' + 'ISO_n12'+ identifier
result_anndata = data.copy()
result_anndata.X = data.X.todense()
result_anndata.raw = None

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

#plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_loc_P.pdf")

#result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])

result_anndata.write_h5ad(outputDir + savename + '.h5ad')

if useTensorboard:
    writer.flush()

#
# %%
