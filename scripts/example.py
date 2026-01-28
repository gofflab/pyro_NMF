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
num_steps = 5000


#%% RUN ALL VARIANTS

##### RUN 1, Unsupervised gamma #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/test') # set working directory for tensorboard logging
nmf_res = run_nmf_unsupervised(data, # data: Expects data in form of anndata 
                               20, # num_patterns: Number of patterns is the number of latent features to learn
                               num_burnin=5000,
                               num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                               spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                               use_pois=False, use_chisq=False, # optional added loss terms
                               use_tensorboard_id='_test_gamma_uns', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                               model_family='gamma')
nmf_res.write_h5ad(outputDir + '/unsupervised_gamma_comparebatch.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
#%%

##### RUN 2, Unsupervised exp #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/test') # set working directory for tensorboard logging
nmf_res = run_nmf_unsupervised(data, # data: Expects data in form of anndata 
                               20, # num_patterns: Number of patterns is the number of latent features to learn
                               num_burnin=5000,
                               num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                               spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                               use_pois=False, use_chisq=False, # optional added loss terms
                               use_tensorboard_id='_test_exponential_uns', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                               model_family='exponential')
nmf_res.write_h5ad(outputDir + '/unsupervised_exponential_comparebatch.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

#%%

##### RUN 3, SSg gamma #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/test') # set working directory for tensorboard logging
nmf_res = run_nmf_supervised(data, # data: Expects data in form of anndata 
                               20, # num_patterns: Number of patterns is the number of latent features to learn
                               fixed_patterns=layers,
                               num_burnin=5000,
                               num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                               spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                               use_pois=False, use_chisq=False, # optional added loss terms
                               use_tensorboard_id='_test_gamma_SSp', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                               model_family='gamma',
                               supervision_type='fixed_samples')

nmf_res.write_h5ad(outputDir + '/SSp_gamma.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##### RUN 4, SSp exp #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/test') # set working directory for tensorboard logging
nmf_res = run_nmf_supervised(data, # data: Expects data in form of anndata 
                               20, # num_patterns: Number of patterns is the number of latent features to learn
                               fixed_patterns=layers,
                               num_burnin=5000,
                               num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                               spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                               use_pois=False, use_chisq=False, # optional added loss terms
                               use_tensorboard_id='_test_exponential_SSp', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                               model_family='exponential',
                               supervision_type='fixed_samples')        
nmf_res.write_h5ad(outputDir + '/SSp_exponential.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



##### RUN 5, SSg gamma #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/test') # set working directory for tensorboard logging
nmf_res = run_nmf_supervised(data, # data: Expects data in form of anndata 
                               20, # num_patterns: Number of patterns is the number of latent features to learn
                               fixed_patterns=gene_layers,
                               num_burnin=5000,
                               num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                               spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                               use_pois=False, use_chisq=False, # optional added loss terms
                               use_tensorboard_id='_test_gamma_SSg', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                               model_family='gamma',
                               supervision_type='fixed_genes')

nmf_res.write_h5ad(outputDir + '/SSg_gamma.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##### RUN 6, SSg exp #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/test') # set working directory for tensorboard logging
nmf_res = run_nmf_supervised(data, # data: Expects data in form of anndata 
                               20, # num_patterns: Number of patterns is the number of latent features to learn
                               fixed_patterns=gene_layers,
                               num_burnin=5000,
                               num_sample_steps=num_steps, # num_steps: Number of steps is the number of training iterations
                               spatial=True, plot_dims=[5,4],  # spatial: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
                               use_pois=False, use_chisq=False, # optional added loss terms
                               use_tensorboard_id='_test_exponential_SSg', #: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
                               model_family='exponential',
                               supervision_type='fixed_genes')        
nmf_res.write_h5ad(outputDir + '/SSg_exponential.h5ad')

pyro.clear_param_store() 
del nmf_res
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()