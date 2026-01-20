#%%
import os
from datetime import datetime
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
#import pyro
#import pyro.optim
#import torch
import random
from pyroNMF.run import run_nmf_unsupervised, run_nmf_supervisedP

#%%

#### input data here, should be stored in anndata.X layer
data = ad.read_h5ad('/raid/kyla/projects/pyro_NMF/analyses/nsf_spatialSimulated/S1.h5ad') # samples x genes
data.X = data.layers['counts']
nmf_res = run_nmf_unsupervised(data, 4, num_steps=200, use_tensorboard_id='test_simulation')

# %%
data = ad.read_h5ad('/raid/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') # samples x genes
data = data[data.obsm['atlas']['Isocortex']]
coords = data.obs.loc[:,['x','y']] # samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset
data.obsm['spatial'] = coords.to_numpy() # samples x 2

#%%
nmf_res = run_nmf_unsupervised(data, 20, num_steps=200)

# %%
layers = data.obsm['atlas'].loc[:,['SS','MO']]*1
#fixed_patterns = torch.tensor(layers.to_numpy())
nmf_res_sup = run_nmf_supervisedP(data, 20, layers, num_steps=200)



# %%
