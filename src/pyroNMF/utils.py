"""Utility functions for device selection and plotting."""

import torch
import matplotlib.pyplot as plt
import numpy as np

def detect_device():
    """Select an available PyTorch device.

    Returns
    -------
    torch.device
        ``cuda`` if available, otherwise ``mps`` if available, else ``cpu``.
    """

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    return torch.device(device)



def plot_grid(patterns, coords, nrows, ncols, size=2, savename = None):
    """Plot spatial patterns on a grid of scatter plots.

    Parameters
    ----------
    patterns : array-like
        Pattern matrix with shape ``(n_samples, n_patterns)``.
    coords : array-like
        Spatial coordinates with shape ``(n_samples, 2)``.
    nrows : int
        Number of rows in the plot grid.
    ncols : int
        Number of columns in the plot grid.
    size : float, optional
        Marker size for scatter points.
    savename : str or None, optional
        If provided, saves the figure to this path.
    """
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
    """Plot spatial patterns without alpha scaling.

    Parameters
    ----------
    patterns : array-like
        Pattern matrix with shape ``(n_samples, n_patterns)``.
    coords : Mapping or array-like
        Spatial coordinates. If a mapping, expects ``coords['x']`` and
        ``coords['y']``.
    nrows : int
        Number of rows in the plot grid.
    ncols : int
        Number of columns in the plot grid.
    s : float, optional
        Marker size for scatter points.
    savename : str or None, optional
        If provided, saves the figure to this path.
    """
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



def plot_results(adata, nrows, ncols, which='best_P', s=4, a=1, scale_alpha = False, scale_values =False, savename = None, title=None):
    """Plot patterns stored in an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with spatial coordinates in ``obsm['spatial']`` and patterns
        stored in ``obsm[which]``.
    nrows : int
        Number of rows in the plot grid.
    ncols : int
        Number of columns in the plot grid.
    which : str, optional
        Key in ``adata.obsm`` containing pattern matrix to plot.
    s : float, optional
        Marker size for scatter points.
    a : float, optional
        Base alpha for scatter points.
    scale_alpha : bool, optional
        If True, scale alpha by per-spot intensity.
    scale_values : bool, optional
        If True, clip color scale to 5th-95th percentile.
    savename : str or None, optional
        If provided, saves the figure to this path.
    title : str or None, optional
        Figure title.
    """
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
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    
    if savename != None:
        plt.savefig(savename)
