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
from pyro.optim import Adam
#from pyro.optim.multi import PyroMultiOptimizer
#from pyro.optim import Adam, SGD, MultiOptimizer
from pyro.infer.autoguide import \
    AutoNormal  # , AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from cogaps.model_NB_cleaned_SS_updatableFixedPvalues import GammaMatrixFactorization, plot_grid, plot_correlations
from cogaps.utils import generate_structured_test_data, generate_test_data

import random
from torch.utils.tensorboard import SummaryWriter

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



#%%

#######################################################
########### ALL USER DEFINED INPUTS HERE ##############
#######################################################

data = ad.read_h5ad('/disk/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad')
#data = data[data.obsm['atlas']['Isocortex']]
D = torch.tensor(data.X) ## RAW COUNT DATA

#atlas = data.obsm['atlas'].loc[:,['SS', 'MO', 'OLF','HIP','STR','layer 1','layer 2/3','layer 4','layer 5','layer 6']]*1
atlas = data.obsm['atlas'].loc[:,['layer 1','layer 2/3','layer 4','layer 5','layer 6']]*1
add_noise = False

coords = data.obs.loc[:,['x','y']]
coords['y'] = -1*coords['y']

num_patterns = 20 # Num EXTRA patterns
device = None # auto detect
NB_probs = None # use default of 1 - sparsity

outputDir = '/disk/kyla/projects/pyro_NMF/results/20241218_SSlayers_updateFixed/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

savename = 'ABCA-1_wholeSlice_superviseLayers_updateFixed_Pfixed_LRe-3'

tensorboard_identifier = 'ABA_wholeSlice_SSLayers_updateFixed_Pfixed_LRe-3'

num_steps = 10000 # Define the number of optimization steps

plot_dims = [5, 5]



#optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # Use the Adam optimizer
# Define separate optimizers with different learning rates
#optimizer = MultiOptimizer([
#    {"optimizer": Adam, "optim_args": {"lr": 0.1, "eps": 1e-08}, "param_names": ["loc_A", "scale_A", "loc_P", "scale_P"]},
#    {"optimizer": Adam, "optim_args": {"lr": 0.01, "eps": 1e-08}, "param_names": ["fixed_P"]},
#])

#optimizer = Adam({
#    "lr": 0.1,  # Default learning rate
#    "eps": 1e-08,
#})

#optimizer.add_param_group({"params": ["fixed_P"], "lr": 0.01})
#optimizer_highlr = pyro.optim.Adam({"lr": 0.1, "eps":1e-08, "params":["loc_A", "scale_A", "loc_P", "scale_P"]}) # Use the Adam optimizer
#optimizer_lowlr = pyro.optim.Adam({"lr": 0.01, "eps":1e-08, "params":["fixed_P"]})
#optimizer = PyroMultiOptimizer([optimizer_highlr, optimizer_lowlr])
#optimizer = MixedMultiOptimizer([(["loc_A", "scale_A", "loc_P", "scale_P"], optimizer_highlr),(["fixed_P"], optimizer_lowlr)])

#optimizer = MultiOptimizer(
#    [
#        Adam({"lr": 0.01}),  # For fixed_P
#        Adam({"lr": 0.1})    # For other parameters
#    ],
#    {
#        "fixed_P": fixed_P_param,
#        "others": ["loc_A", "scale_A", "loc_P", "scale_P"]
#    }
#)

#optimizer = pyro.optim.ClippedAdam({
#    "lr": 0.1,
#    "eps": 1e-08,
#    "param_group": [
#        {"params": ["loc_A", "scale_A", "loc_P", "scale_P"], "lr": 0.1},
#        {"params": ["fixed_P"], "lr": 0.01},
#    ],
#})
'''
# Define the optimizer with parameter groups
class CustomClippedAdam:
    def __init__(self, params, lr=0.1, eps=1e-8, clip_norm=1.0):
        self.optimizer = pyro.optim.Adam(params, lr=lr, eps=eps)
        self.clip_norm = clip_norm

    def __call__(self, params):
        # Split parameters into groups with different learning rates
        param_groups = [
            {"params": [p for n, p in params if "fixed_P" not in n], "lr": 0.1},  # Higher learning rate
            {"params": [p for n, p in params if "fixed_P" in n], "lr": 0.01},   # Lower learning rate for fixed_P
        ]

        adam_optimizer = pyro.optim.Adam(param_groups, eps=1e-8)
        return pyro.optim.ClippedAdam(adam_optimizer, clip_norm=self.clip_norm)

optimizer = pyro.optim.PyroOptim(CustomClippedAdam, {"lr": 0.1, "eps": 1e-8})
'''
'''
def custom_clipped_adam(params):
    # Define parameter groups with specific learning rates
    param_groups = [
        {"params": [p for n, p in params if "fixed_P" not in n], "lr": 0.1},
        {"params": [p for n, p in params if "fixed_P" in n], "lr": 0.01},
    ]
    return pyro.optim.Adam(param_groups, eps=1e-8)

# Wrap the custom optimizer in Pyro's PyroOptim
optimizer = pyro.optim.PyroOptim(custom_clipped_adam, {})
'''
''''
# Define a custom ClippedAdam optimizer with parameter groups
def create_optimizer(params):
    # Manually create parameter groups with different learning rates
    param_groups = [
        {"params": [p for name, p in params if "fixed_P" not in name], "lr": 0.1},  # Default parameters
        {"params": [p for name, p in params if "fixed_P" in name], "lr": 0.01},    # Lower learning rate for "fixed_P"
    ]
    return torch.optim.Adam(param_groups, eps=1e-8)  # Base optimizer used with ClippedAdam

# Wrap the custom optimizer in Pyro's ClippedAdam
optimizer = ClippedAdam({"lr": 0.1, "clip_norm": 1.0})

# Create PyroOptim using a lambda to pass parameters to your custom optimizer
pyro_optimizer = pyro.optim.PyroOptim(
    lambda params: optimizer(get_param_groups(params)),
    {}
)

# Define helper to map parameter names and tensors
def get_param_groups(params):
    return [(name, param) for name, param in pyro.get_param_store().named_parameters()]

'''



loss_fn = pyro.infer.Trace_ELBO() # Define the loss function
draw_model = None # None or name for output file

#%%
# ADD GAMMA NOISE
if add_noise:
    # Gamma distribution parameters for added noise
    alpha_0, beta_0 = 0.5, 0.1  # Small noise for 0s
    alpha_1, beta_1 = 5.0, 0.5  # Larger noise for 1s

    # Adding Gamma noise to each element
    gamma_noise = np.zeros_like(atlas, dtype=float)
    gamma_noise[atlas == 0] = np.random.gamma(alpha_0, beta_0, size=(atlas == 0).sum().sum())
    gamma_noise[atlas == 1] = np.random.gamma(alpha_1, beta_1, size=(atlas == 1).sum().sum())

    # Resultant matrix after adding noise
    transformed_matrix = atlas + gamma_noise
    transformed_matrix.to_csv(outputDir + 'atlas_noisy.csv')

else:
    transformed_matrix = atlas
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

if NB_probs == None:
    percZeros = (D == 0).sum().sum() / (D.shape[0]*D.shape[1])
    NB_probs = 1-percZeros
    print(f"Data is {percZeros:.2f}% sparse")


writer = SummaryWriter(comment = tensorboard_identifier)


# Instantiate the model
model = GammaMatrixFactorization(D.shape[1], D.shape[0], num_patterns, fixed_patterns=atlas.to_numpy(), NB_probs = NB_probs, device=device)


# Define helper to group parameters
#def create_param_groups(params):
#    # Assign parameters to groups based on their names
#   param_groups = [
#        {"params": [p for name, p in params if "fixed_P" not in name], "lr": 0.1},  # Default learning rate
#        {"params": [p for name, p in params if "fixed_P" in name], "lr": 0.01},    # Lower learning rate for "fixed_P"
#    ]
#    print(param_groups)
#    return param_groups

# Custom optimizer function for PyroOptim
#def custom_optimizer(params):
#    param_groups = create_param_groups(params)
#    return torch.optim.Adam(param_groups, eps=1e-8)

# Wrap the custom optimizer in Pyro's PyroOptim
#optimizer = pyro.optim.PyroOptim(
#    lambda params: custom_optimizer([(name, param) for name, param in pyro.get_param_store().named_parameters()]),
#    {}
#)


def per_param_callable(param_name):
    if param_name == 'fixed_P':
        return {"lr": 0.001}
    else:
        return {"lr": 0.1}

optimizer = pyro.optim.Adam(per_param_callable)

#optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # Use the Adam optimizer


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

# Run inference
for step in range(1,num_steps+1):
    loss = svi.step(D)

    if step % 10 == 0:
        writer.add_scalar("Loss/train", loss, step)
        writer.flush()

        losses.append(loss)
        steps.append(step)

    if step % 50 == 0:
        plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
        writer.add_figure("loc_P", plt.gcf(), step)

        plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("loc_P_hist", plt.gcf(), step)

        plot_grid(model.P_total.detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
        writer.add_figure("P_total", plt.gcf(), step)

        #plot_grid(pyro.param("fixed_loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
        #writer.add_figure("fixed_loc_P", plt.gcf(), step)

        #plt.hist(pyro.param("fixed_loc_P").detach().to('cpu').numpy().flatten(), bins=30)
        #writer.add_figure("fixed_loc_P_hist", plt.gcf(), step)
        plot_grid(pyro.param("fixed_P").detach().to('cpu').numpy(), coords, 2, 3, savename = None)
        writer.add_figure("fixed_P", plt.gcf(), step)

        plt.hist(pyro.param("fixed_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("fixed_P_hist", plt.gcf(), step)


        plt.hist(model.P_total.detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("P_total_hist", plt.gcf(), step)

        plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("loc_A_hist", plt.gcf(), step)

        plt.hist(pyro.param("scale_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("scale_P_hist", plt.gcf(), step)

        plt.hist(pyro.param("scale_A").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("scale_A_hist", plt.gcf(), step)

    if step % 100 == 0:

        print(f"Iteration {step}, ELBO loss: {loss}")

        D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
        plt.hist(D_reconstructed.flatten(), bins=30)
        writer.add_figure("D_reconstructed_his", plt.gcf(), step)

endTime = datetime.now()

# Save the inferred parameters
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#savename = '/disk/kyla/projects/pyro_NMF/results/scaleD_constraint_comparison/' + identifier + '/' + 'ISO_n12'+ identifier
result_anndata = data.copy()

loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
#loc_P.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
loc_P.index = result_anndata.obs.index
result_anndata.obsm['loc_P'] = loc_P

scale_P = pd.DataFrame(pyro.param("scale_P").detach().to('cpu').numpy())
#scale_P.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
scale_P.columns = ['Pattern_' + str(x+1) for x in scale_P.columns]
scale_P.index = result_anndata.obs.index
result_anndata.obsm['scale_P'] = scale_P

total_P = pd.DataFrame(model.P_total.detach().to('cpu').numpy())
total_P.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
total_P.index = result_anndata.obs.index
result_anndata.obsm['P_total'] = total_P

loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
loc_A.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
loc_A.index = result_anndata.var.index
result_anndata.varm['loc_A'] = loc_A

scale_A = pd.DataFrame(pyro.param("scale_A").detach().to('cpu').numpy()).T
scale_A.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
scale_A.index = result_anndata.var.index # need names to match anndata names
result_anndata.varm['scale_A'] = scale_A

loc_D = pd.DataFrame(model.D_reconstructed.detach().cpu().numpy())
loc_D.index = result_anndata.obs.index
loc_D.columns = result_anndata.var.index # need names to match anndata names
result_anndata.layers['loc_D'] = loc_D

#plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_loc_P.pdf")

result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])
result_anndata.obsm['atlas_used'] = transformed_matrix

fixed_P = pd.DataFrame(pyro.param("fixed_P").detach().to('cpu').numpy())
#loc_P.columns = list(atlas.columns) + ['Pattern_' + str(x) for x in range(1,num_patterns+1)]
fixed_P.columns = transformed_matrix.columns
fixed_P.index = result_anndata.obs.index
result_anndata.obsm['fixed_P'] = fixed_P


result_anndata.write_h5ad(outputDir + savename + '.h5ad')

writer.flush()

#
# %%