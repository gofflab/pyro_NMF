"""Utility functions for device selection and spatial/pattern plotting."""

import torch
import matplotlib.pyplot as plt
import numpy as np


def detect_device():
    """Select an available PyTorch device.

    Checks for CUDA, then MPS (Apple Silicon), then falls back to CPU.

    Returns
    -------
    torch.device
        ``cuda`` if available, ``mps`` if available, otherwise ``cpu``.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    return torch.device(device)


def plot_grid(patterns, coords, nrows, ncols, size=10, savename=None):
    """Plot spatial patterns on a grid of scatter plots with alpha scaling.

    Colour scale is clipped to the 5th–95th percentile of each pattern.
    Alpha transparency is linearly scaled from 0.3 (minimum) to 1.0
    (maximum) based on per-spot intensity.

    Parameters
    ----------
    patterns : array-like of shape (n_samples, n_patterns)
        Pattern matrix. Each column is one pattern.
    coords : array-like of shape (n_samples, 2)
        Spatial coordinates. Column 0 is x, column 1 is y.
    nrows : int
        Number of rows in the plot grid.
    ncols : int
        Number of columns in the plot grid.
    size : float, optional
        Marker size for scatter points.
    savename : str or None, optional
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    num_patterns = patterns.shape[1]
    x, y = coords[:, 0], coords[:, 1]
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                p5 = np.percentile(patterns[:, i], 5)
                p95 = np.percentile(patterns[:, i], 95)
                pattern_min = patterns[:, i].min()
                pattern_max = patterns[:, i].max()
                alpha_values = 0.3 + (0.7 * (patterns[:, i] - pattern_min) / (pattern_max - pattern_min))
                axes[r, c].scatter(x, y, c=patterns[:, i], s=size,
                                   alpha=alpha_values, vmin=p5, vmax=p95,
                                   cmap='viridis', edgecolors='none')
                axes[r, c].set_yticklabels([])
                axes[r, c].set_xticklabels([])
                i += 1

    if savename is not None:
        plt.savefig(savename)


def plot_grid_noAlpha(patterns, coords, nrows, ncols, s=10, savename=None):
    """Plot spatial patterns without alpha scaling.

    Full opacity scatter plots with per-panel colour bars. Coordinates
    are expected as a mapping with ``'x'`` and ``'y'`` keys.

    Parameters
    ----------
    patterns : array-like of shape (n_samples, n_patterns)
        Pattern matrix. Each column is one pattern.
    coords : Mapping
        Spatial coordinates with keys ``'x'`` and ``'y'``, each of
        length ``n_samples``.
    nrows : int
        Number of rows in the plot grid.
    ncols : int
        Number of columns in the plot grid.
    s : float, optional
        Marker size for scatter points.
    savename : str or None, optional
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    num_patterns = patterns.shape[1]
    x, y = coords['x'], coords['y']
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                pattern_min = patterns[:, i].min()
                pattern_max = patterns[:, i].max()
                p = axes[r, c].scatter(x, y, c=patterns[:, i], s=s, alpha=1,
                                       vmin=pattern_min, vmax=pattern_max,
                                       cmap='viridis', edgecolors='none')
                axes[r, c].set_yticklabels([])
                axes[r, c].set_xticklabels([])
                fig.colorbar(p, ax=axes[r, c])
                i += 1

    if savename is not None:
        plt.savefig(savename)


def plot_results(adata, nrows, ncols, which='best_P', s=4, a=1,
                 scale_alpha=False, scale_values=False, savename=None, title=None):
    """Plot patterns stored in an AnnData object on spatial coordinates.

    Requires ``adata.obsm['spatial']`` to be present. Any key in
    ``adata.obsm`` that holds a DataFrame of shape
    ``(n_samples, n_patterns)`` can be visualised.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with spatial coordinates in ``obsm['spatial']`` and a
        pattern DataFrame in ``obsm[which]``.
    nrows : int
        Number of rows in the plot grid.
    ncols : int
        Number of columns in the plot grid.
    which : str, optional
        Key in ``adata.obsm`` to plot. Default is ``'best_P'``.
        Common choices: ``'mean_P'``, ``'best_P'``, ``'last_P'``,
        ``'best_P_scaled'``, ``'markers_P'``.
    s : float, optional
        Marker size for scatter points.
    a : float, optional
        Base alpha (opacity) for scatter points.
    scale_alpha : bool, optional
        If True, scale alpha linearly with per-spot intensity
        (0.3 at minimum, 1.0 at maximum).
    scale_values : bool, optional
        If True, clip the colour scale to the 5th–95th percentile
        of each pattern.
    savename : str or None, optional
        If provided, save the figure to this path.
    title : str or None, optional
        Overall figure title.
    """
    patterns = adata.obsm[which]
    coords = adata.obsm['spatial']
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    num_patterns = patterns.shape[1]
    x, y = coords[:, 0], coords[:, 1]
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i < num_patterns:
                pattern_min = patterns.iloc[:, i].min()
                pattern_max = patterns.iloc[:, i].max()
                if scale_alpha:
                    a = 0.3 + (0.7 * (patterns.iloc[:, i] - pattern_min) / (pattern_max - pattern_min))
                if scale_values:
                    pattern_min = np.percentile(patterns.iloc[:, i], 5)
                    pattern_max = np.percentile(patterns.iloc[:, i], 95)
                p = axes[r, c].scatter(x, y, c=patterns.iloc[:, i], s=s,
                                       alpha=a, vmin=pattern_min, vmax=pattern_max,
                                       cmap='viridis', edgecolors='none')
                axes[r, c].set_yticklabels([])
                axes[r, c].set_xticklabels([])
                axes[r, c].set_title(patterns.columns[i])
                fig.colorbar(p, ax=axes[r, c])
                i += 1

    plt.suptitle(title, y=1.05)
    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename)
