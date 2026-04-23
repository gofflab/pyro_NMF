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

## RUN on 10/8/2025
## Using data from here: /mnt/morbo/Data/Users/ahally/white_body/pilot_data/combined/adata/251007_combined_all_filter.h5ad
## Data copied on 10/7/2025 ~6pm
# Ran NMF with 40 patterns, 10000 iterations, on cuda : /raid/kyla/projects/pyro_NMF/analyses/squid_20251007/squid_uns_n40.h5ad
# Ran NMF with 40 patterns, 25000 iterations, on cuda : /raid/kyla/projects/pyro_NMF/analyses/squid_20251007/squid_uns_n40_25kIter.h5ad


#%% LOAD DATA
data = ad.read_h5ad('/raid/kyla/data/251007_combined_all_filter.h5ad')

#%% Adjust coordinates
rois = data.obs['roi'].unique()
n_rois = len(rois)
    
n_cols = int(np.ceil(np.sqrt(n_rois)))
n_rows = int(np.ceil(n_rois / n_cols))
    
x_adj = np.zeros(len(data.obs))
y_adj = np.zeros(len(data.obs))

# find largest dimensions of any roi    
max_width,max_height = data.obs.loc[:,['roi', 'x', 'y']].groupby('roi').agg(lambda x: x.max()-x.min()).max(axis=0)

x_spacing = max_width * 1 # space them out a little
y_spacing = max_height * 1
    
for idx, roi in enumerate(rois):
    row = idx // n_cols
    col = idx % n_cols
        
    mask = data.obs['roi'] == roi
        
    roi_x = data.obs.loc[mask, 'x'].values
    roi_y = data.obs.loc[mask, 'y'].values
        
    roi_x_norm = roi_x - roi_x.min()
    roi_y_norm = roi_y - roi_y.min()
        
    x_adj[mask] = roi_x_norm + (col * x_spacing)
    y_adj[mask] = roi_y_norm + (row * y_spacing)
    
data.obs['x_adj'] = x_adj
data.obs['y_adj'] = y_adj
    
plt.scatter(data.obs.loc[:,'x_adj'], data.obs.loc[:,'y_adj'], s=1, alpha=0.3, c=data.obs.loc[:,'mean_intensity_edu'])

#%%
coords = data.obs.loc[:,['x_adj','y_adj']] # shape: samples x 2
data.obsm['spatial'] = coords.to_numpy() # expects coordinates in 'spatial' if using spatial NMF

#%% RUN UNSUPERVISED NMF

### Parameters:
# data: Expects data in form of anndata
# num_patterns: Number of patterns is the number of latent features to learn
# num_steps=20000: Number of steps is the number of training iterations
# device=None: Device to run the model on, e.g. 'cpu' or 'cuda', if None, will detect device automatically
# NB_probs=0.5: Probability of using negative binomial distribution for the data
# use_chisq=False: If True, will add chi-squared to loss
# scale=None: Scale parameter for the Negative Binomial distribution, if None, will use default of 2*std(data)

# The rest of the parameters are only important for tensorboard
    # use_tensorboard_id=None: Optional string to identify this run in tensorboard, if None, will not log to tensorboard
    # spatial=False: If True, will use spatial coordinates in obsm['spatial'] to plot patterns, if False, nothing will be plotted
    # plot_dims=None: Dimensions of the plot grid, e.g. [5,4] means 5 rows and 4 columns
nmf_res = run_nmf_unsupervised(data, 40, num_steps=25000, spatial=True, plot_dims=[6,7], device='cuda', use_tensorboard_id='_squid_uns_n40')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/squid_20251007/squid_uns_n40_25kIter.h5ad')
pyro.clear_param_store() 

# %%
