"""Utility functions for device selection and spatial/pattern plotting."""

import pandas as pd

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
                    pattern_min = np.percentile(patterns.iloc[:,i], 1)
                    pattern_max = np.percentile(patterns.iloc[:,i], 99)
                p = axes[r,c].scatter(x, y, c=patterns.iloc[:,i], s=s, alpha=a, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                axes[r,c].set_title(patterns.columns[i])
                fig.colorbar(p, ax=axes[r,c])
                axes[r,c].set_box_aspect(1)
                axes[r,c].set_aspect('equal')  
                #axes[r,c].set_title(patterns.columns[i])
                i += 1
            else:
                axes[r,c].set_visible(False)
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    
    if savename != None:
        plt.savefig(savename)



def plot_results_sorted(adata, nrows, ncols, which='best_P', s=4, a=1, scale_alpha = False, scale_values =False, savename = None, title=None):
    # Adapt to sort each pattern from low to high before plotting
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
                    pattern_min = np.percentile(patterns.iloc[:,i], 1)
                    pattern_max = np.percentile(patterns.iloc[:,i], 99)
                order = np.argsort(patterns.iloc[:,i])
                p = axes[r,c].scatter(x[order], y[order], c=patterns.iloc[order,i], s=s, alpha=a, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                axes[r,c].set_title(patterns.columns[i])
                fig.colorbar(p, ax=axes[r,c])
                axes[r,c].set_box_aspect(1)
                axes[r,c].set_aspect('equal')  
                #axes[r,c].set_title(patterns.columns[i])
                i += 1
            else:
                axes[r,c].set_visible(False)
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    
    if savename != None:
        plt.savefig(savename)




def plot_results_multisample(adata, nrows, ncols, which='best_P', split_by='sample', sample_order=None, s=4, a=1, scale_alpha=False, scale_values=False, savename=None, title=None):
    patterns = adata.obsm[which]
    coords = adata.obsm['spatial']
    num_patterns = patterns.shape[1]

    samples = adata.obs[split_by].unique().tolist()
    if sample_order is not None:
        samples = [s for s in sample_order if s in samples]

    for i in range(num_patterns):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = np.array(axes).reshape(nrows, ncols)  # ensure 2D

        pattern_col = patterns.iloc[:, i]
        pattern_min = pattern_col.min()
        pattern_max = pattern_col.max()
        if scale_values:
            pattern_min = np.percentile(pattern_col, 1)
            pattern_max = np.percentile(pattern_col, 99)

        for j, (r, c) in enumerate([(r, c) for r in range(nrows) for c in range(ncols)]):
            if j < len(samples):
                mask = adata.obs[split_by] == samples[j]
                x, y = coords[mask, 0], coords[mask, 1]
                vals = pattern_col[mask]

                _a = a
                if scale_alpha:
                    _a = 0.3 + (0.7 * (vals - pattern_min) / (pattern_max - pattern_min))
                    _a = _a.clip(0, 1)

                p = axes[r, c].scatter(x, y, c=vals, s=s, alpha=_a, vmin=pattern_min, vmax=pattern_max, cmap='viridis', edgecolors='none')
                axes[r, c].set_xticklabels([])
                axes[r, c].set_yticklabels([])
                axes[r, c].set_title(samples[j])
                fig.colorbar(p, ax=axes[r, c])
                axes[r, c].set_box_aspect(1)
                axes[r, c].set_aspect('equal')
            else:
                axes[r, c].set_visible(False)

        pattern_title = f"{title} – {patterns.columns[i]}" if title else patterns.columns[i]
        plt.suptitle(pattern_title, y=1.05)
        plt.tight_layout()

        if savename is not None:
            # insert pattern index before extension so each figure saves separately
            base, ext = savename.rsplit('.', 1)
            plt.savefig(f"{base}_pattern{i}.{ext}")
        plt.show()


def plot_results_multisample_sorted(
    adata,
    nrows,
    ncols,
    which='best_P',
    split_by='sample',
    sample_order=None,
    s=4,
    a=1,
    scale_alpha=False,
    scale_values=False,
    savename=None,
    title=None
):
    patterns = adata.obsm[which]
    coords = adata.obsm['spatial']
    num_patterns = patterns.shape[1]

    samples = adata.obs[split_by].unique().tolist()
    if sample_order is not None:
        samples = [samp for samp in sample_order if samp in samples]

    for i in range(num_patterns):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = np.array(axes).reshape(nrows, ncols)  # ensure 2D

        pattern_col = patterns.iloc[:, i]

        pattern_min = pattern_col.min()
        pattern_max = pattern_col.max()

        if scale_values:
            pattern_min = np.percentile(pattern_col, 1)
            pattern_max = np.percentile(pattern_col, 99)

        for j, (r, c) in enumerate([(r, c) for r in range(nrows) for c in range(ncols)]):
            if j < len(samples):
                mask = adata.obs[split_by] == samples[j]

                x = coords[mask, 0]
                y = coords[mask, 1]
                vals = pattern_col[mask]

                # Sort points from low to high values so high values are plotted on top
                order = np.argsort(vals)

                x_sorted = x[order]
                y_sorted = y[order]
                vals_sorted = vals.iloc[order] if hasattr(vals, "iloc") else vals[order]

                _a = a
                if scale_alpha:
                    _a = 0.3 + (0.7 * (vals_sorted - pattern_min) / (pattern_max - pattern_min))
                    _a = _a.clip(0, 1)

                p = axes[r, c].scatter(
                    x_sorted,
                    y_sorted,
                    c=vals_sorted,
                    s=s,
                    alpha=_a,
                    vmin=pattern_min,
                    vmax=pattern_max,
                    cmap='viridis',
                    edgecolors='none'
                )

                axes[r, c].set_xticklabels([])
                axes[r, c].set_yticklabels([])
                axes[r, c].set_title(samples[j])
                fig.colorbar(p, ax=axes[r, c])
                axes[r, c].set_box_aspect(1)
                axes[r, c].set_aspect('equal')

            else:
                axes[r, c].set_visible(False)

        pattern_title = f"{title} – {patterns.columns[i]}" if title else patterns.columns[i]
        #plt.suptitle(pattern_title, y=1.05)
        #plt.tight_layout()
        fig.suptitle(pattern_title, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if savename is not None:
            base, ext = savename.rsplit('.', 1)
            plt.savefig(f"{base}_{pattern_title}.{ext}")

        plt.show()



def plot_genes(adata, nrows, ncols, which_genes, s=4, a=1, scale_alpha = False, scale_values =False, savename = None, title=None):
    gene_sub = adata[:, adata.var_names.isin(which_genes)]
    patterns = pd.DataFrame(gene_sub.X)
    patterns.columns = gene_sub.var_names
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
                    pattern_min = np.percentile(patterns.iloc[:,i], 1)
                    pattern_max = np.percentile(patterns.iloc[:,i], 99)
                p = axes[r,c].scatter(x, y, c=patterns.iloc[:,i], s=s, alpha=a, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                axes[r,c].set_title(patterns.columns[i])
                fig.colorbar(p, ax=axes[r,c])
                axes[r,c].set_box_aspect(1)
                axes[r,c].set_aspect('equal')  
                #axes[r,c].set_title(patterns.columns[i])
                i += 1
            else:
                axes[r,c].set_visible(False)
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    
    if savename != None:
        plt.savefig(savename)




def plot_genes(adata, nrows, ncols, which_genes, s=4, a=1, scale_alpha = False, scale_values =False, savename = None, title=None):
    gene_sub = adata[:, adata.var_names.isin(which_genes)]
    patterns = pd.DataFrame(gene_sub.X)
    patterns.columns = gene_sub.var_names
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
                    pattern_min = np.percentile(patterns.iloc[:,i], 1)
                    pattern_max = np.percentile(patterns.iloc[:,i], 99)
                p = axes[r,c].scatter(x, y, c=patterns.iloc[:,i], s=s, alpha=a, vmin=pattern_min, vmax=pattern_max, cmap='viridis',edgecolors='none')
                axes[r,c].set_yticklabels([])
                axes[r,c].set_xticklabels([])
                axes[r,c].set_title(patterns.columns[i])
                fig.colorbar(p, ax=axes[r,c])
                axes[r,c].set_box_aspect(1)
                axes[r,c].set_aspect('equal')  
                #axes[r,c].set_title(patterns.columns[i])
                i += 1
            else:
                axes[r,c].set_visible(False)
    plt.suptitle(title, y=1.05)
    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename)



import gc

def debug_cuda_tensors(step, top_n=20):
    """Print the largest tensors currently allocated on CUDA."""
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((obj.element_size() * obj.nelement(), obj.shape, obj.dtype, obj.requires_grad))
        except Exception:
            pass
    tensors.sort(key=lambda x: x[0], reverse=True)
    print(f"\n--- Step {step}: {len(tensors)} CUDA tensors, top {top_n} ---")
    for size, shape, dtype, rg in tensors[:top_n]:
        print(f"  {size/1e6:.1f} MB  {shape}  {dtype}  requires_grad={rg}")
    print(f"  Total allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")