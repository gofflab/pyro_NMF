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


##### RUN 1 #####    
#os.chdir('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models') # set working directory for tensorboard logging
#nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_pois=False, use_tensorboard_id='unsupervised_gamma_noerror', model_family='gamma')
#nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_noerror.h5ad')
#pyro.clear_param_store() 

##### RUN 2 #####    
#os.chdir('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models') # set working directory for tensorboard logging
#nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=True, use_pois=False, use_tensorboard_id='unsupervised_gamma_chisqError', model_family='gamma')
#nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_chisqError.h5ad')
#pyro.clear_param_store() 

##### RUN 3 #####    
#os.chdir('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models') # set working directory for tensorboard logging
#nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = True, use_tensorboard_id='unsupervised_gamma_poisError', model_family='gamma')
#nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_poisError.h5ad')
#pyro.clear_param_store() 


##### RUN 4 #####    
#os.chdir('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models') # set working directory for tensorboard logging
#nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = False, use_tensorboard_id='unsupervised_exponential_noerror', model_family='exponential')
#nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_noerror.h5ad')
#pyro.clear_param_store() 


##### RUN 5 #####    
#os.chdir('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models') # set working directory for tensorboard logging
#nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=True, use_pois = False, use_tensorboard_id='unsupervised_exponential_chisqError', model_family='exponential')
#nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_chisqError.h5ad')
#pyro.clear_param_store() 

##### RUN 6 #####    
#os.chdir('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models') # set working directory for tensorboard logging
#nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = True, use_tensorboard_id='unsupervised_exponential_poisError', model_family='exponential')
#nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_poisError.h5ad')
#pyro.clear_param_store() 








##### RUN 1 #####    
os.chdir('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models') # set working directory for tensorboard logging

nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_pois=False, use_tensorboard_id='unsupervised_gamma_noerror_2', model_family='gamma')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_noerror_2.h5ad')
pyro.clear_param_store() 
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_pois=False, use_tensorboard_id='unsupervised_gamma_noerror_3', model_family='gamma')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_noerror_3.h5ad')
pyro.clear_param_store() 

##### RUN 2 #####    
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=True, use_pois=False, use_tensorboard_id='unsupervised_gamma_chisqError_2', model_family='gamma')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_chisqError_2.h5ad')
pyro.clear_param_store() 
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=True, use_pois=False, use_tensorboard_id='unsupervised_gamma_chisqError_3', model_family='gamma')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_chisqError_3.h5ad')
pyro.clear_param_store() 

##### RUN 3 #####    
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = True, use_tensorboard_id='unsupervised_gamma_poisError_2', model_family='gamma')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_poisError_2.h5ad')
pyro.clear_param_store() 
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = True, use_tensorboard_id='unsupervised_gamma_poisError_3', model_family='gamma')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_gamma_poisError_3.h5ad')
pyro.clear_param_store() 


##### RUN 4 #####    
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = False, use_tensorboard_id='unsupervised_exponential_noerror_2', model_family='exponential')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_noerror_2.h5ad')
pyro.clear_param_store() 
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = False, use_tensorboard_id='unsupervised_exponential_noerror_3', model_family='exponential')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_noerror_3.h5ad')
pyro.clear_param_store()

##### RUN 5 #####    
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=True, use_pois = False, use_tensorboard_id='unsupervised_exponential_chisqError_2', model_family='exponential')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_chisqError_2.h5ad')
pyro.clear_param_store() 
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=True, use_pois = False, use_tensorboard_id='unsupervised_exponential_chisqError_3', model_family='exponential')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_chisqError_3.h5ad')
pyro.clear_param_store()

##### RUN 6 #####    
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = True, use_tensorboard_id='unsupervised_exponential_poisError_2', model_family='exponential')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_poisError_2.h5ad')
pyro.clear_param_store() 
nmf_res = run_nmf_unsupervised(data, 20, num_steps=10000, spatial=True, plot_dims=[5,4], use_chisq=False, use_pois = True, use_tensorboard_id='unsupervised_exponential_poisError_3', model_family='exponential')
nmf_res.write_h5ad('/raid/kyla/projects/pyro_NMF/analyses/20250108_compare_models/unsupervised_exponential_poisError_3.h5ad')
pyro.clear_param_store()







# %%
