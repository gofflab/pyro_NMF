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

from models.model_NB_cleaned import GammaMatrixFactorization, plot_grid, plot_correlations
from models.plotting import plot_multisample_scale
#from cogaps.utils import generate_structured_test_data, generate_test_data

import random
from torch.profiler import profile, record_function, ProfilerActivity

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



#%%

#######################################################
########### ALL USER DEFINED INPUTS HERE ##############
#######################################################

#### input data here, should be stored in anndata.X layer
# If you have a numpy array of data try : data = ad.AnnData(array_data) # Kyla untested
data = ad.read_h5ad('/home/kyla/data/Zhuang-ABCA-1-raw_wMeta_wAnnotations_wAtlas_sub20_KW.h5ad') # samples x genes
#data = data[data.obs['brain_section_label_x'].isin(['Zhuang-ABCA-1.082', 'Zhuang-ABCA-1.083', 'Zhuang-ABCA-1.084', 'Zhuang-ABCA-1.085', 'Zhuang-ABCA-1.086'])]
sections = data.obs['brain_section_label_x'].unique()
data = data[data.obs['brain_section_label_x'].isin(sections[:15])]
data = data[data.obs.sort_values('z').index,:]
sampleDf = data.obs['brain_section_label_x']

# Remove rows and columns that are all zeros
#data = data[~np.all(data.X == 0, axis=1), :]  # Remove rows with all zeros
#data = data[:, ~np.all(data.X == 0, axis=0)]  # Remove columns with all zeros


#(156722, 1122)
#data = data[data.obsm['atlas']['Isocortex']]

D = torch.tensor(data.X) ## RAW COUNT DATA

#### coords should be two columns named x and y
coords = data.obs.loc[:,['x','y']] # samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset

num_patterns = 20 # select num patterns
num_steps = 10000 # Define the number of optimization steps

device = None # options ['cpu', 'cuda', 'mps', etc]; if None: auto detect cpu vs gpu vs mps
NB_probs = None # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D

#### output parameters
outputDir = '/home/kyla/projects/pyro_NMF/results/'
savename = 'mem_trace_test_15slices' # output anndata will be saved in outputDir/savename.h5ad
mem_file = savename
useTensorboard = True
tensorboard_identifier = 'mem_trace_test_15slices' # key added to tensorboard output name

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

torch.cuda.memory._record_memory_history(enabled='all', device=device, max_entries=100000)

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

#%%
with profile(
    activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True) as prof:
    with torch.autograd.profiler.record_function("A"):
        with torch.autograd.profiler.record_function("P"):

        #with record_function("model_inference"):
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

                if step % 1000 == 0:

                    patterns = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
                    patterns.columns = ['Pattern_' + str(x+1) for x in patterns.columns]
                    patterns.index = data.obs.index

                    for pattern in patterns.columns:
                        plot_multisample_scale(patterns.loc[:,pattern], pattern, coords, sampleDf)
                        writer.add_figure(pattern, plt.gcf(), step)


            endTime = datetime.now()
            print('Runtime: '+ str(round((endTime - startTime).total_seconds())) + ' seconds')
            torch.cuda.memory._dump_snapshot(mem_file + '.pkl')

# Save the inferred parameters
if not os.path.exists(outputDir + savename + '/'):
    os.makedirs(outputDir + savename + '/')

#%%
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))

prof.export_chrome_trace("trace.json")



#%%
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

result_anndata.varm['A_mean'] = loc_A/scale_A


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

result_anndata.obsm['P_mean'] = loc_P/scale_P

loc_D = pd.DataFrame(model.D_reconstructed.detach().cpu().numpy())
loc_D.index = result_anndata.obs.index
loc_D.columns = result_anndata.var.index # need names to match anndata names
result_anndata.layers['loc_D'] = loc_D
print("Saving loc_D in anndata.layers['loc_D']")

#plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_loc_P.pdf")

result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])

result_anndata.write_h5ad(outputDir + savename + '/' + savename + '.h5ad')

patterns = loc_P
for pattern in patterns.columns:
    plot_multisample_scale(patterns.loc[:,pattern], pattern, coords, sampleDf)
    plt.savefig(outputDir + savename + '/' + pattern + '.png')

means = loc_P/scale_P
for mean in means.columns:
    plot_multisample_scale(means.loc[:,mean], mean, coords, sampleDf)
    plt.savefig(outputDir + savename + '/' + mean + '_mean.png')

if useTensorboard:
    writer.flush()

#
# %%
