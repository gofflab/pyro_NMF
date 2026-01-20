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
data = ad.read_h5ad('/raid/kyla/data/Zhuang-ABCA-1-raw_wMeta_wAnnotations_wAtlas_sub20_KW.h5ad') 
#data = data[data.obsm['atlas']['Isocortex']]
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
#%% RUN ALL VARIANTS

##### RUN 1, Unsupervised gamma #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/test') # set working directory for tensorboard logging
nmf_res = run_nmf_unsupervised(data, # data: Expects data in form of anndata 
                               20, # num_patterns: Number of patterns is the number of latent features to learn
                               num_steps=1000, # num_steps: Number of steps is the number of training iterations
                               device='cpu',
                               spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                               use_pois=False, use_chisq=False, # optional added loss terms
                               use_tensorboard_id='_test_exp_uns_batched', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                               model_family='exponential', batch_size=1000)
nmf_res.write_h5ad(outputDir + '/unsupervised_exponential_batched.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %%
