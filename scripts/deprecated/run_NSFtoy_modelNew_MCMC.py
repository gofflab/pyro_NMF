#!/usr/bin/env -S python3 -u
#%%
import os
from datetime import datetime
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.optim
import pyro.poutine as poutine
import scanpy as sc
import seaborn as sns
import torch
from pyro.infer.autoguide import \
    AutoNormal  # , AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from pyro.infer import MCMC,NUTS,Predictive

#from cogaps.guide import CoGAPSGuide
from models.model_new import GammaMatrixFactorization
from models.utils import generate_structured_test_data, generate_test_data

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

#%% Plotting function
## designed in mind for the NSF toy data
## TODO move to utils -- tried but got error trying to import from utils
def plot_grid(patterns, coords, nrows, ncols, savename = None):
    fig, axes = plt.subplots(nrows,ncols)
    num_patterns = patterns.shape[1]
    x, y = coords['x'], coords['y']
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    resolution = 0.114286
    x_grid = np.arange(x_min, x_max + resolution, resolution)
    y_grid = np.arange(y_min, y_max + resolution, resolution)

    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                # Initialize an empty grid (NaN-filled)
                grid = np.full((len(y_grid), len(x_grid)), np.nan)
                # Map the patterns to the grid; find nearest grid indices for each x, y pair
                ix = np.searchsorted(x_grid, x) - 1
                iy = np.searchsorted(y_grid, y) - 1
                for j in range(len(patterns[:,i])):
                    if 0 <= ix[j] < len(x_grid) and 0 <= iy[j] < len(y_grid):
                        grid[iy[j], ix[j]] = patterns[j,i]
                axes[r,c].imshow(grid, cmap='viridis')
                i += 1
    if savename != None:
        plt.savefig(savename)

def plot_correlations(true_vals, inferred_vals, savename):
    true_vals_df = pd.DataFrame(true_vals)
    true_vals_df.columns = ['True_' + str(x) for x in true_vals_df.columns]
    inferred_vals_df = pd.DataFrame(inferred_vals)
    inferred_vals_df.columns = ['Inferred_' + str(x) for x in inferred_vals_df.columns]
    correlations = true_vals_df.merge(inferred_vals_df, left_index=True, right_index=True).corr().round(2)
    plt.figure()
    sns.clustermap(correlations.iloc[:true_vals.shape[1], true_vals.shape[1]:], annot=False, cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
    if savename != None:
        plt.savefig(savename + '_correlations.png')

#%% Import tensorboard & setup writer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

## Load in NSF toy data
S1 = sc.read_h5ad('S1.h5ad')
S1 = S1[:,:80] # based on NSF demo.ipynb
P_true = pd.DataFrame(S1.obsm['spfac'])
P_true.columns = ['True_' + str(i) for i in range(1, P_true.shape[1]+1)]
D = torch.tensor(pd.DataFrame(S1.layers['counts']).values) # D should be sample by genes
coords = pd.DataFrame(S1.obsm['spatial'])
coords.columns = ['x', 'y']

num_patterns = 4

if torch.backends.mps.is_available():
    device=torch.device('mps')
    print('using mps')
elif torch.cuda.is_available():
    device=torch.device('cuda')
    print('using cuda')
else:
    device=torch.device('cpu')
    print('using cpu')
#device=torch.device('cpu') ## NEED TO ADD BACK IN DEVICE IN THE MODEL

# Move data to device
D = D.to(device)

# Create a DataLoader to handle batching and shuffling (not implemented yet)
#dataset = TensorDataset(D)
#dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

##### RUN BELOW HERE
#%% Clear Pyro's parameter store
pyro.clear_param_store()

# Start timer
startTime = datetime.now()

# Define the number of optimization steps
num_steps = 1000

# Use the Adam optimizer
optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # try default

# Define the loss function
loss_fn = pyro.infer.Trace_ELBO()

#%% Instantiate the model
#model = CoGAPSModel(D, num_patterns, device=device)
model = GammaMatrixFactorization(D.shape[1], num_patterns, D.shape[0],device=device)
#(D, num_patterns, device=device, init_method=None)


#%% Instantiate the guide
guide = AutoNormal(model)

#%% Define the inference algorithm
#svi = pyro.infer.SVI(model=model,
#                    guide=guide,
#                    optim=optimizer,
#                    loss=loss_fn)

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples = num_steps, warmup_steps = 10)
mcmc.run(D)

#%% Run inference

### GET LOSS
samples = mcmc.get_samples()
mses = list()
for step in range(num_steps):
    A_i = samples['A'][step,:,:]
    P_i = samples['P'][step,:,:]
    AP_i = torch.matmul(P_i, A_i)
    mse = torch.mean((D - AP_i) ** 2)
    mses.append(mse.item())

    if step % 10 == 0:
        writer.add_scalar("MSE", mse, step)
        writer.flush()
    if step % 50 == 0:
        plot_grid(P_i.detach().to('cpu').numpy(), coords, 2, 2, savename = None)
        writer.add_figure("P", plt.gcf(), step)

        plt.hist(P_i.detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("P_hist", plt.gcf(), step)

        plt.hist(A_i.detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("A_hist", plt.gcf(), step)

        sns.heatmap(P_i.detach().to('cpu').numpy()[:10,:num_patterns].round(3), annot=True)
        writer.add_figure("P_i", plt.gcf(), step)
        sns.heatmap(A_i.detach().to('cpu').numpy()[:num_patterns,:10].round(3).T, annot=True)
        writer.add_figure("A_i", plt.gcf(), step)



#%% save
savename = 'NSFtoy_n4_MCMC_run3_'
last_Pi = pd.DataFrame(samples['P'][-1,:,:].detach().to('cpu').numpy())
#last_Pi.columns = ['Pattern_' + str(x + 1) for x in last_Pi.columns]
#last_Pi.index = ISO_data.obs.index
last_Pi.to_csv(savename + 'P.csv')

average_Pi = pd.DataFrame(torch.mean(samples['P'], dim=0).detach().to('cpu').numpy())
#average_Pi.columns = ['Pattern_' + str(x + 1) for x in average_Pi.columns]
#average_Pi.index = ISO_data.obs.index
average_Pi.to_csv(savename + 'P_avg.csv')

last_Ai = pd.DataFrame(samples['A'][-1,:,:].detach().to('cpu').numpy())
#last_Ai.index = ['Pattern_' + str(x + 1) for x in last_Ai.index]
#last_Ai.columns = ISO_data.var.loc[:,'gene_symbol_x']
last_Ai.T.to_csv(savename + 'A.csv')

average_Ai = pd.DataFrame(torch.mean(samples['A'], dim=0).detach().to('cpu').numpy())
#average_Ai.index = ['Pattern_' + str(x + 1) for x in average_Ai.index]
#average_Ai.columns = ISO_data.var.loc[:,'gene_symbol_x']
average_Ai.T.to_csv(savename + 'A_avg.csv')


#for step in range(num_steps):
    #print(step)
#    loss = svi.step(D)

#    if step % 10 == 0:
#        writer.add_scalar("Loss/train", loss, step)
#        writer.flush()
#    if step % 50 == 0:
#        plot_grid(pyro.param("scale_P").detach().to('cpu').numpy(), coords, 2, 2, savename = None)
#        writer.add_figure("scale_P", plt.gcf(), step)

#        plt.hist(pyro.param("scale_P").detach().to('cpu').numpy().flatten(), bins=30)
#        writer.add_figure("scale_P_hist", plt.gcf(), step)

#        plt.hist(pyro.param("scale_A").std(dim=0).detach().to('cpu').numpy(), bins=30)
#        writer.add_figure("scale_A std (gene stds)", plt.gcf(), step)

#    if step % 100 == 0:
#        print(f"Iteration {step}, ELBO loss: {loss}")

#%% Retrieve the inferred parameters
#A_scale = pyro.param("A_scale").detach().to('cpu').numpy()
#P_scale = pyro.param("P_scale").detach().to('cpu').numpy()
#A_mean = pyro.param("A_mean").detach().to('cpu').numpy()
#P_mean = pyro.param("P_mean").detach().to('cpu').numpy()

#pd.DataFrame(A_scale).to_csv('A_scale.csv')
#pd.DataFrame(P_scale).to_csv('P_scale.csv')
#pd.DataFrame(A_mean).to_csv('A_mean.csv')
#pd.DataFrame(P_mean).to_csv('P_mean.csv')


# Print the shapes of the inferred parameters
#print("Inferred A shape:", A_mean.shape)
#print("Inferred P shape:", P_mean.shape)

# End timer
#print("Time taken:")
#print(datetime.now() - startTime)

# Plot the inferred patterns and true patterns
#plot_grid(P_true.values, coords, 2, 2, savename = 'True_P.png')
#plot_grid(P_mean, coords, 2, 2, savename = 'Inferred_P.png')

# Save outputs
#plt.figure()
#plt.hist(P_true.values.flatten(), bins=30)
#plt.savefig('P_true_hist.png')

#plt.figure()
#plt.hist(P_mean.flatten(), bins=30)
#plt.savefig('P_mean_hist.png')

# Check inferred parameters against the true parameters
#plot_correlations(P_true.values, P_mean, 'P_mean')
#writer.add_figure("P_mean_correlations", plt.gcf(), step)

# # %%
# writer.flush()