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
from pyroNMF.run_v2 import *


#%% LOAD DATA
data = ad.read_h5ad('/raid/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') 
data = data[data.obsm['atlas']['Isocortex']]
coords = data.obs.loc[:,['x','y']] # shape: samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset
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

nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_tensorboard_id='test_unsupervised')

#%% RUN SUPERVISED P NMF

### Additional parameters:
# fixed_patterns: DataFrame of fixed patterns to use, shape: samples x num_patterns
layers = data.obsm['atlas'].loc[:,['SS','MO']]*1 # pass this in as dataframe to preserve names
nmf_res_sup = run_nmf_supervisedP(data, 20, fixed_patterns=layers, num_steps=10000, spatial=True, plot_dims=[5,5], use_tensorboard_id='test_supervised')

