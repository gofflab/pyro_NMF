
import torch
import matplotlib.pyplot as plt
import numpy as np

def detect_device():
    """
    Setup the device for PyTorch.
    
    Parameters:
    - device: Device to run the model on (e.g., 'cpu', 'cuda', 'mps').
    
    Returns:
    - device: The initialized device.
    """

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    return torch.device(device)



def plot_grid(patterns, coords, nrows, ncols, size=2, savename = None):
    fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*4, nrows*4))
    num_patterns = patterns.shape[1]
    x, y = coords[:,0], coords[:,1]
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                p5 = np.percentile(patterns[:,i], 5)
                p95 = np.percentile(patterns[:,i], 95)
                pattern_min = patterns[:,i].min()
                pattern_max = patterns[:,i].max()
                #pattern_min = np.min(patterns[:, i])
                #pattern_max = np.max(patterns[:, i])
                alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))
                #axes[r,c].scatter(x, y, c=patterns[:,i], s=8,alpha=np.minimum(alpha_values, 1), vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                p = axes[r,c].scatter(x, y, c=patterns[:,i], s=size,alpha=alpha_values, vmin=p5, vmax=p95, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                #axes[r,c].set_title(patterns.columns[i])
                i += 1

    if savename != None:
        plt.savefig(savename)


def plot_grid_noAlpha(patterns, coords, nrows, ncols, s=4, savename = None):
    fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*5, nrows*4))
    num_patterns = patterns.shape[1]
    x, y = coords['x'], coords['y']
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                pattern_min = patterns[:,i].min()
                pattern_max = patterns[:,i].max()
                p = axes[r,c].scatter(x, y, c=patterns[:,i], s=s, alpha=1, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                fig.colorbar(p, ax=axes[r,c])

                #axes[r,c].set_title(patterns.columns[i])
                i += 1
    if savename != None:
        plt.savefig(savename)



def plot_results(adata, nrows, ncols, which='best_P', s=4, a=1, scale_alpha = False, scale_values =False, savename = None):
    patterns = adata.obsm[which]
    coords = adata.obsm['spatial']
    fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*5, nrows*4))
    num_patterns = patterns.shape[1]
    x, y = coords[:,0], coords[:,1]
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                pattern_min = patterns.iloc[:,i].min()
                pattern_max = patterns.iloc[:,i].max()
                if scale_alpha:
                    a = 0.3 + (0.7 * (patterns.iloc[:, i] - pattern_min) / (pattern_max - pattern_min))
                if scale_values:
                    pattern_min = np.percentile(patterns.iloc[:,i], 5)
                    pattern_max = np.percentile(patterns.iloc[:,i], 95)
                p = axes[r,c].scatter(x, y, c=patterns.iloc[:,i], s=s, alpha=a, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                fig.colorbar(p, ax=axes[r,c])

                #axes[r,c].set_title(patterns.columns[i])
                i += 1
    if savename != None:
        plt.savefig(savename)
