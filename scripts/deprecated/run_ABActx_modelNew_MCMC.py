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
from pyro.infer import MCMC,NUTS,Predictive, RandomWalkKernel

#from cogaps.guide import CoGAPSGuide
from models.model_new import GammaMatrixFactorization
from models.utils import generate_structured_test_data, generate_test_data

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

#%% Plotting function
## KW TODO -- generalize for any data; move out of model definition
def plot_grid(patterns, coords, nrows, ncols, savename = None):
    fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*4, nrows*4))
    num_patterns = patterns.shape[1]
    x, y = coords['x'], coords['y']
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                p5 = np.percentile(patterns[:,i], 5)
                p95 = np.percentile(patterns[:,i], 95)
                pattern_min = np.min(patterns[:, i])
                pattern_max = np.max(patterns[:, i])
                alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))

                axes[r,c].scatter(x, -1*y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                i += 1
    if savename != None:
        plt.savefig(savename)

def plot_correlations(true_vals, inferred_vals, savename = None):
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

#CTX_data = pd.read_csv('ISO_data.csv',index_col=0)
ABA_data = ad.read_h5ad('/disk/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad')
ISO_data = ABA_data[ABA_data.obsm['atlas']['Isocortex']]
#D = torch.tensor(np.log1p(ISO_data.X)) ## LOG TRANSFORM DATA
D = torch.tensor(ISO_data.X) ## RAW COUNT DATA
coords = ISO_data.obs.loc[:,['x','y']]
num_patterns = 12

if torch.backends.mps.is_available():
    device=torch.device('mps')
    print('using mps')
elif torch.cuda.is_available():
    device=torch.device('cuda')
    print('using cuda')
else:
    device=torch.device('cpu')
    print('using cpu')


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
num_steps = 10

#%% Instantiate the model
model = GammaMatrixFactorization(D.shape[1], num_patterns, D.shape[0], device=device)

nuts_kernel = NUTS(model)
#kernel = RandomWalkKernel(model)
mcmc = MCMC(nuts_kernel, num_samples = num_steps, warmup_steps = 10)
mcmc.run(D)

#%%
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
        plot_grid(P_i.detach().to('cpu').numpy(), coords, 4, 3, savename = None)
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
savename = 'results/SVI_MCMC/Oct29_17-00-11_MCMC_1kwarm_1ksample_scaleD20max_run3'
last_Pi = pd.DataFrame(samples['P'][-1,:,:].detach().to('cpu').numpy())
last_Pi.columns = ['Pattern_' + str(x + 1) for x in last_Pi.columns]
last_Pi.index = ISO_data.obs.index
last_Pi.to_csv(savename + 'P.csv')

#average_Pi = pd.DataFrame(torch.mean(samples['P'], dim=0).detach().to('cpu').numpy())
#average_Pi.columns = ['Pattern_' + str(x + 1) for x in average_Pi.columns]
#average_Pi.index = ISO_data.obs.index
#average_Pi.to_csv(savename + 'P_avg.csv')

last_Ai = pd.DataFrame(samples['A'][-1,:,:].detach().to('cpu').numpy())
last_Ai.index = ['Pattern_' + str(x + 1) for x in last_Ai.index]
last_Ai.columns = ISO_data.var.loc[:,'gene_symbol_x']
last_Ai.T.to_csv(savename + 'A.csv')

#average_Ai = pd.DataFrame(torch.mean(samples['A'], dim=0).detach().to('cpu').numpy())
#average_Ai.index = ['Pattern_' + str(x + 1) for x in average_Ai.index]
#average_Ai.columns = ISO_data.var.loc[:,'gene_symbol_x']
#average_Ai.T.to_csv(savename + 'A_avg.csv')


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