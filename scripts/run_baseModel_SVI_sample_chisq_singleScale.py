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

from pyroNMF.models.gamma_NB_base_singleScaleUpdate import Gamma_NegBinomial_base
import random

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


#%%

#######################################################
########### ALL USER DEFINED INPUTS HERE ##############
#######################################################

#### input data here, should be stored in anndata.X layer
data = ad.read_h5ad('/home/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') # samples x genes; must be called data and have .X attribute
D = torch.tensor(data.X) ## RAW COUNT DATA

#### user-specified model parameters
num_patterns = 25 # select num patterns
num_steps = 10000 # Define the number of optimization steps
device = None # options ['cpu', 'cuda', 'mps', etc]; if None: auto detect cpu vs gpu vs mps
NB_probs = 0.5 # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D
use_chisq = True

#### output parameters
outputDir = '/home/kyla/projects/pyro_NMF/results/test_singleScale'
savename = 'test_singleScale_wChisq_2' # output anndata will be saved in outputDir/savename.h5ad

useTensorboard = True
tensorboard_identifier = 'test_singleScale_wChisq_2' # key added to tensorboard output name

spatial = True # if True, will plot patterns on spatial coordinates
if spatial:
    #### coords should be two columns named x and y
    coords = data.obs.loc[:,['x','y']] # samples x 2
    coords['y'] = -1*coords['y'] # specific for this dataset
    plot_dims = [5, 5] # rows x columns should be > num patterns; this is for plotting
    if plot_dims[0]*plot_dims[1] < num_patterns:
        #raise ValueError("plot_dims should be greater than num_patterns")
        print("Warning: plot_dims less than num_patterns. Not all patterns will be plotted")


#### more model parameters
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
model = Gamma_NegBinomial_base(D.shape[0], D.shape[1], num_patterns, use_chisq=use_chisq, NB_probs = NB_probs, device=device)          

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
samp = False # start with no sampling
sampled = 0

# Run inference
for step in range(1,num_steps+1):
    if samp: # count number of sampled values
        sampled += 1

    loss = svi.step(D,samp)

    if step % 100 == 0:
        print(f"Iteration {step}, ELBO loss: {loss}")


    if useTensorboard:

        writer.add_scalar("Loss/train", loss, step)
        writer.flush()

        writer.add_scalar('Best chi-squared',  model.best_chisq, step)
        writer.flush()

        writer.add_scalar('Saved chi-squared iter',  model.best_chisq_iter, step)
        writer.flush()

        writer.add_scalar("Chi-squared", model.chi2, step)
        writer.flush()

        if step % 10 == 0:
            losses.append(loss)
            steps.append(step)

        if step % 50 == 0:
            if spatial:
                # plot loc P
                model.plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                writer.add_figure("loc_P", plt.gcf(), step)
            
                # plot this sampled P
                model.plot_grid(model.P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                writer.add_figure("current sampled P", plt.gcf(), step)

                # plot average of sampled P
                if samp:
                    model.plot_grid(model.sumP.detach().cpu().numpy() / sampled, coords, plot_dims[0], plot_dims[1], savename = None)
                    writer.add_figure("average sampled P", plt.gcf(), step)


            plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("loc_P_hist", plt.gcf(), step)

            plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("loc_A_hist", plt.gcf(), step)
        
            writer.add_scalar('scale_AP',  pyro.param("scale").detach().to('cpu'), step)
            writer.flush()

        if step % 100 == 0:

            D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
            plt.hist(D_reconstructed.flatten(), bins=30)
            writer.add_figure("D_reconstructed_his", plt.gcf(), step)

    if step == num_steps/2: # store second half of samples
        samp = True
        

endTime = datetime.now()
print('Runtime: '+ str(round((endTime - startTime).total_seconds())) + ' seconds')

#%%
# Save the inferred parameters
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

result_anndata = data.copy()

### Save loc_P
loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
loc_P.index = result_anndata.obs.index
result_anndata.obsm['loc_P'] = loc_P
print("Saving loc_P in anndata.obsm['loc_P']")

### Save loc_A
loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
loc_A.columns = ['Pattern_' + str(x+1) for x in loc_A.columns]
loc_A.index = result_anndata.var.index
result_anndata.varm['loc_A'] = loc_A
print("Saving loc_A in anndata.varm['loc_A']")

### Save last sampled P
last_P = pd.DataFrame(model.P.detach().cpu().numpy())
last_P.columns = ['Pattern_' + str(x+1) for x in last_P.columns]
last_P.index = result_anndata.obs.index
result_anndata.obsm['last_P'] = last_P
print("Saving final sampled P in anndata.obsm['last_P']")


### Save last sampled A
last_A = pd.DataFrame(model.A.detach().cpu().numpy()).T
last_A.columns = ['Pattern_' + str(x+1) for x in last_A.columns]
last_A.index = result_anndata.var.index
result_anndata.varm['last_A'] = last_A
print("Saving final sampled A in anndata.varm['last_A']")


### Save average sampled P
avg_samp_P = pd.DataFrame(model.sumP.detach().cpu().numpy()/sampled)
avg_samp_P.columns = ['Pattern_' + str(x+1) for x in avg_samp_P.columns]
avg_samp_P.index = result_anndata.obs.index
result_anndata.obsm['avg_P'] = avg_samp_P
print("Saving averaged of sampled P in anndata.obsm['avg_P']")

### Save average sampled A
avg_samp_A = pd.DataFrame(model.sumA.detach().cpu().numpy()/sampled).T
avg_samp_A.columns = ['Pattern_' + str(x+1) for x in avg_samp_A.columns]
avg_samp_A.index = result_anndata.var.index
result_anndata.varm['avg_A'] = avg_samp_A
print("Saving averaged of sampled A in anndata.varm['avg_A']")


### Save best P (via chi-squared)
best_P = pd.DataFrame(model.best_P.detach().cpu().numpy())
best_P.columns = ['Pattern_' + str(x+1) for x in best_P.columns]
best_P.index = result_anndata.obs.index
result_anndata.obsm['best_P'] = best_P
print("Saving best P via chi2 in anndata.obsm['best_P']")


### Save best A (via chi-squared)
best_A = pd.DataFrame(model.best_A.detach().cpu().numpy()).T
best_A.columns = ['Pattern_' + str(x+1) for x in best_A.columns]
best_A.index = result_anndata.var.index
result_anndata.varm['best_A'] = best_A
print("Saving best A via chi2 in anndata.varm['best_A']")


if spatial:
    model.plot_grid(model.best_P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_bestP.pdf")

result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])
result_anndata.uns['step_w_bestChisq'] = model.best_chisq_iter
result_anndata.uns['scale'] = pyro.param("scale").detach().to('cpu').numpy()

result_anndata.write_h5ad(outputDir + '/' + savename + '.h5ad')

if useTensorboard:
    writer.flush()

#
# %%
