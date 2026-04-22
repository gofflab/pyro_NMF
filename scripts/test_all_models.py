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
import random
from pyroNMF.run_inference import *
import gc

#%% LOAD DATA
data = ad.read_h5ad('/raid/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') 
data = data[data.obsm['atlas']['Isocortex']]
coords = data.obs.loc[:,['x','y']] # shape: samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset
data.obsm['spatial'] = coords.to_numpy() # expects coordinates in 'spatial' if using spatial NMF

### Additional parameters:
# fixed_patterns: DataFrame of fixed patterns to use, shape: samples x num_patterns
layers = data.obsm['atlas'].loc[:,['SS','MO']]*1 # pass this in as dataframe to preserve names

def mean_expression(adata, mask):
    X = adata.X[mask.values, :]
    return X.mean(axis=0)

gene_layers = pd.DataFrame(
    {
        'SS': mean_expression(data, layers['SS']),
        'MO': mean_expression(data, layers['MO']),
    },
    index=data.var_names
)

outputDir = "/raid/kyla/projects/pyro_NMF/analyses/test"
os.chdir(outputDir) # set working directory for tensorboard logging

num_steps = 500

###############################################################
####################### RUN ALL VERSIONS ######################
###############################################################

#%% RUN UNSUPERVISED VERSIONS

##### RUN 1, Unsupervised gamma #####    
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testGammaUns', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='gamma', optimizer=pyro.optim.AdamW({"lr": 0.1, "eps": 1e-08}))
nmf_res.write_h5ad(outputDir + '/testGammaUns.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##### RUN 2, Unsupervised exp #####    
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testExpUns', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='exponential', optimizer=pyro.optim.AdamW({"lr": 0.1, "eps": 1e-08}))
nmf_res.write_h5ad(outputDir + '/testExpUns.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##### RUN 3, Unsupervised expSingle #####    
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testExpSinUns', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='exponentialSingle', optimizer=pyro.optim.AdamW({"lr": 0.1, "eps": 1e-08}))
nmf_res.write_h5ad(outputDir + '/testExpSinUns.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

#%% RUN FIXED PATTERNS 
###### RUN 4, SSp gamma ##### 
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    fixed_patterns=layers,
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testGammaSSp', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='gamma', supervision_type='fixed_samples')
   
nmf_res.write_h5ad(outputDir + '/testGammaSSp.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

###### RUN 5, SSp exp ##### 
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    fixed_patterns=layers,
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testExpSSp', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='exponential', supervision_type='fixed_samples')
   
nmf_res.write_h5ad(outputDir + '/testExpSSp.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

###### RUN 6, SSp expSingle ##### 
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    fixed_patterns=layers,
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testExpSinSSp', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='exponentialSingle', supervision_type='fixed_samples')
   
nmf_res.write_h5ad(outputDir + '/testExpSinSSp.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


#%% RUN FIXED GENES 
###### RUN 7, SSg gamma ##### 
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    fixed_patterns=gene_layers,
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testGammaSSg', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='gamma', supervision_type='fixed_genes')
   
nmf_res.write_h5ad(outputDir + '/testGammaSSg.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

###### RUN 8, SSg exp ##### 
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    fixed_patterns=gene_layers,
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testExpSSg', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='exponential', supervision_type='fixed_genes')
   
nmf_res.write_h5ad(outputDir + '/testExpSSg.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

###### RUN 9, SSg expSingle ##### 
nmf_res = run_nmf(data, # data: Expects data in form of anndata 
                    20, # num_patterns: Number of patterns is the number of latent features to learn
                    num_burnin=num_steps,
                    num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                    fixed_patterns=gene_layers,
                    spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                    use_pois=False, use_chisq=False, # optional added loss terms
                    use_tensorboard_id='_testExpSinSSg', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                    model_family='exponentialSingle', supervision_type='fixed_genes')
   
nmf_res.write_h5ad(outputDir + '/testExpSinSSg.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %%
