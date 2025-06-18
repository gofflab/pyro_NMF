#!/usr/bin/env -S python3 -u
#%%
import os
from datetime import datetime
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import pyro
import pyro.optim
import torch
from pyro.infer.autoguide import AutoNormal
#from torch.utils.data import DataLoader, TensorDataset

from pyroNMF.models.gamma_NB_new_SSfixedP import Gamma_NegBinomial_SSFixed
import random

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


#%%

#######################################################
########### ALL USER DEFINED INPUTS HERE ##############
#######################################################

#### input data here, should be stored in anndata.X layer
#### input data here, should be stored in anndata.X layer
data = ad.read_h5ad('/raid/kyla/projects/pyro_NMF/analyses/Otero_tau/excitatory_subsetMerGenes.h5ad') # samples x genes
#data = data[data.obs.loc[:,'disease'] == 'Alzheimer disease']
counts = data.X

#%%
D = torch.tensor(counts, dtype=torch.float32) ## RAW COUNT DATA
U = (D*0.1).clip(min=0.3) # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D

#%%
fixed_patterns = torch.tensor(((data.obs.loc[:,['SORT']] == 'AT8')*1).to_numpy())
#fixed_patterns = torch.tensor((pd.get_dummies(data.obs.loc[:,['tau_intensity']])*1).to_numpy())
fixed_pattern_names = list(['AT8'])
#fixed_pattern_names = list((pd.get_dummies(data.obs.loc[:,['tau_intensity']])*1).columns)

percZeros = (D == 0).sum().sum() / (D.shape[0]*D.shape[1])
print(f"Data is {percZeros*100:.2f}% sparse")

#### user-specified model parameters
num_patterns = 10 # select num patterns
num_steps = 10000 # Define the number of optimization steps
device = None # options ['cpu', 'cuda', 'mps', etc]; if None: auto detect cpu vs gpu vs mps
NB_probs = 0.5 # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D
use_chisq = False
scale = (D.numpy().std())*2
useTensorboard = True

#### output parameters
outputDir = '/raid/kyla/projects/pyro_NMF/analyses/Otero_tau'
savename = 'excitatory_subsetMerGenes_fixAT8'
print(f"Outputting to {outputDir}/{savename}.h5ad")

tensorboard_identifier = savename # key added to tensorboard output name

spatial = False # if True, will plot patterns on spatial coordinates
if spatial:
    #### coords should be two columns named x and y
    coords =  pd.DataFrame(data.obsm['spatial'],columns=['x','y'])
    plot_dims = [4, 5] # rows x columns should be > num patterns; this is for plotting
    if plot_dims[0]*plot_dims[1] < num_patterns:
        #raise ValueError("plot_dims should be greater than num_patterns")
        print("Warning: plot_dims less than num_patterns. Not all patterns will be plotted")


#### more model parameters
optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # Use the Adam optimizer
loss_fn = pyro.infer.Trace_ELBO() # Define the loss function
draw_model = '/raid/kyla/projects/pyro_NMF/src/pyroNMF/models/gamma_NB_new_SSfixedP.pdf' # None or name for output file to create pdf diagram of pyro model


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

print(f"Selecting device {device}")
D = D.to(device)
U = U.to(device)
fixed_patterns = fixed_patterns.to(device)

if useTensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment = tensorboard_identifier)


# Instantiate the model
model = Gamma_NegBinomial_SSFixed(D.shape[0], D.shape[1], num_patterns, fixed_patterns=fixed_patterns, use_chisq=use_chisq, scale=scale, NB_probs = NB_probs, device=device)          

# Draw model
if draw_model != None:
    pyro.render_model(model, model_args=(D,U),
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

#%%
# Start timer
startTime = datetime.now()

steps = []
losses = []
#last_loss = None
#last_chi2 = None

# Run inference
for step in range(1,num_steps+1):

    try:
        loss = svi.step(D,U)
    except ValueError as e:
         print(f"ValueError during iteration {step}: {e}")
         break 
    
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

            plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("loc_P_hist", plt.gcf(), step)

            plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
            writer.add_figure("loc_A_hist", plt.gcf(), step)
        
        if step % 100 == 0:
            D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
            plt.hist(D_reconstructed.flatten(), bins=30)
            writer.add_figure("D_reconstructed_his", plt.gcf(), step)

        
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
loc_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
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
last_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
last_A.index = result_anndata.var.index
result_anndata.varm['last_A'] = last_A
print("Saving final sampled A in anndata.varm['last_A']")

### Save best P (via chi-squared)
best_P = pd.DataFrame(model.best_P.detach().cpu().numpy())
best_P.columns = ['Pattern_' + str(x+1) for x in best_P.columns]
best_P.index = result_anndata.obs.index
result_anndata.obsm['best_P'] = best_P
print("Saving best P via chi2 in anndata.obsm['best_P']")

### Save best P (via chi-squared)
fixed_P = pd.DataFrame(model.fixed_P.detach().cpu(), columns = fixed_pattern_names, index=result_anndata.obs.index)
#best_P.columns = ['Pattern_' + str(x+1) for x in best_P.columns]
#best_P.index = result_anndata.obs.index
result_anndata.obsm['fixed_P'] = fixed_P
print("Saving fixed P via chi2 in anndata.obsm['fixed_P']")

### Save best P (via chi-squared)
best_P_total = fixed_P.merge(best_P, left_index=True, right_index=True)
result_anndata.obsm['best_P_total'] = best_P_total
print("Saving best P via chi2 in anndata.obsm['best_P_total']")

### Save best A (via chi-squared)
best_A = pd.DataFrame(model.best_A.detach().cpu().numpy()).T
best_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
best_A.index = result_anndata.var.index
result_anndata.varm['best_A'] = best_A
print("Saving best A via chi2 in anndata.varm['best_A']")

### Save best locP (via chi-squared)
best_locP = pd.DataFrame(model.best_locP.detach().cpu().numpy())
best_locP.columns = ['Pattern_' + str(x+1) for x in best_locP.columns]
best_locP.index = result_anndata.obs.index
result_anndata.obsm['best_locP'] = best_locP
print("Saving best loc P via chi2 in anndata.obsm['best_locP']")

### Save best locA (via chi-squared)
best_locA = pd.DataFrame(model.best_locA.detach().cpu().numpy()).T
best_locA.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
best_locA.index = result_anndata.var.index
result_anndata.varm['best_locA'] = best_A
print("Saving best loc A via chi2 in anndata.varm['best_locA']")


if spatial:
    model.plot_grid(best_P_total.to_numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_bestP.pdf")

result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])
result_anndata.uns['step_w_bestChisq'] = model.best_chisq_iter
result_anndata.uns['scale'] = scale


settings = {
    'num_patterns': str(num_patterns),
    'num_steps': str(num_steps),
    'device': str(device),
    'NB_probs': str(NB_probs),
    'use_chisq': str(use_chisq),
    'scale': str(scale),
    'tensorboard_identifier': str(writer.log_dir)
}

result_anndata.uns['settings'] = pd.DataFrame(list(settings.values()),index=list(settings.keys()), columns=['settings'])
result_anndata.write_h5ad(outputDir + '/' + savename + '.h5ad')

if useTensorboard:
    writer.flush()

#
# %%
