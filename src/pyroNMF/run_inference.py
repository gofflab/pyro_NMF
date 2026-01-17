"""High-level inference wrappers for pyroNMF models."""

import torch

from pyroNMF.models.gamma_NB_models import (
    Gamma_NegBinomial_base,
    Gamma_NegBinomial_SSFixedGenes,
    Gamma_NegBinomial_SSFixedSamples
)
from pyroNMF.models.exp_pois_models import (
    Exponential_base,
    Exponential_SSFixedGenes,
    Exponential_SSFixedSamples
)
from pyroNMF.utils import detect_device, plot_grid, plot_grid_noAlpha
from torch.utils.tensorboard import SummaryWriter
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
from pyro.infer.autoguide import AutoNormal

default_dtype = torch.float32

def validate_data(data, spatial=False, plot_dims=None, num_patterns=None):
    """Validate AnnData inputs and optional spatial coordinates.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object containing raw counts in ``.X``.
    spatial : bool, optional
        If True, validate that ``data.obsm['spatial']`` exists and is 2D.
    plot_dims : sequence of int or None, optional
        Plotting grid dimensions ``[rows, cols]`` for spatial plots.
    num_patterns : int or None, optional
        Number of patterns to compare against ``plot_dims``.

    Returns
    -------
    numpy.ndarray or None
        Spatial coordinates if ``spatial=True``, otherwise None.

    Raises
    ------
    ValueError
        If spatial coordinates are missing or malformed.
    """
    print("Running validate_data")
    D = torch.tensor(data.X) if not hasattr(data.X, 'toarray') else torch.tensor(data.X.toarray())
    coords = None
    
    perc_zeros = (D == 0).sum().sum() / (D.shape[0] * D.shape[1])
    print(f"Data contains {D.shape[0]} cells and {D.shape[1]} genes")
    print(f"Data is {perc_zeros*100:.2f}% sparse")
    
    if spatial:
        if data.obsm.get('spatial') is None:
            raise ValueError("Spatial coordinates are not present in the data. Please provide spatial coordinates in obsm['spatial']")
        
        coords = data.obsm['spatial']
        if coords.shape[1] != 2:
            raise ValueError("Spatial coordinates should have two columns named 'x' and 'y'")
        
        if plot_dims is not None and num_patterns is not None:
            if plot_dims[0] * plot_dims[1] < num_patterns:
                print("Warning: plot_dims less than num_patterns. Not all patterns will be plotted")
    
    return coords


def prepare_tensors(data, device=None):
    """Prepare tensors for inference and move them to a device.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object with raw counts in ``.X`` (dense or sparse).
    device : str or torch.device or None, optional
        Target device (``'cpu'``, ``'cuda'``, ``'mps'``). If None, autodetect.

    Returns
    -------
    tuple
        ``(D, U, scale, device)`` where:

        - ``D`` is the data tensor with shape ``(n_samples, n_genes)``.
        - ``U`` is a per-entry scale tensor used in chi-squared computation.
        - ``scale`` is a scalar derived from the data standard deviation.
        - ``device`` is the selected torch device.
    """
    if device is None:
        device = detect_device()
    print(f"Selecting device {device}")
    print(f"preparing tensors")
    
    if hasattr(data.X, 'toarray'):
        print('Setting sparse D')
        D = torch.tensor(data.X.toarray(), dtype=default_dtype).to(device)
    else:
        print('Setting D')
        D = torch.tensor(data.X, dtype=default_dtype).to(device)
    U = (D * 0.1).clip(min=0.3).to(device)
    scale = torch.tensor((D.cpu().numpy().std()) * 2, dtype=default_dtype, device=device)
    
    return D, U, scale, device


def setup_model_and_optimizer(D, num_patterns, scale=1, NB_probs=0.5, use_chisq=False, use_pois=False, device=None,
                             fixed_patterns=None, model_type='gamma_unsupervised',
                             supervision_type=None):
    """Construct the model, guide, and SVI optimizer.

    Parameters
    ----------
    D : torch.Tensor
        Data tensor with shape ``(n_samples, n_genes)``.
    num_patterns : int
        Number of patterns to learn.
    scale : float, optional
        Scale parameter used for Gamma-based models.
    NB_probs : float, optional
        Probability parameter for the Negative Binomial likelihood.
    use_chisq : bool, optional
        If True, include chi-squared loss via ``pyro.factor``.
    use_pois : bool, optional
        If True, include Poisson log-likelihood via ``pyro.factor``.
    device : torch.device or None, optional
        Device for parameters and tensors.
    fixed_patterns : torch.Tensor or None, optional
        Fixed pattern matrix for semi-supervised models.
    model_type : str, optional
        One of ``'gamma_unsupervised'``, ``'gamma_supervised'``,
        ``'exponential_unsupervised'``, or ``'exponential_supervised'``.
    supervision_type : str or None, optional
        ``'fixed_genes'`` or ``'fixed_samples'`` for supervised models.

    Returns
    -------
    tuple
        ``(model, guide, svi)`` where ``guide`` is an ``AutoNormal`` guide
        and ``svi`` is a ``pyro.infer.SVI`` instance.
    """
    # Instantiate the model
    if model_type == 'gamma_unsupervised':
        model = Gamma_NegBinomial_base(
            D.shape[0], D.shape[1], num_patterns, 
            use_chisq=use_chisq, use_pois=use_pois,scale=scale, 
            NB_probs=NB_probs, device=device
        )
    elif model_type == 'gamma_supervised':
        if supervision_type == 'fixed_genes':
            model = Gamma_NegBinomial_SSFixedGenes(
                D.shape[0], D.shape[1], num_patterns, 
                fixed_patterns=fixed_patterns, use_chisq=use_chisq, use_pois=use_pois,
                scale=scale, NB_probs=NB_probs, device=device
            )
        elif supervision_type == 'fixed_samples':
            model = Gamma_NegBinomial_SSFixedSamples(
                D.shape[0], D.shape[1], num_patterns, 
                fixed_patterns=fixed_patterns, use_chisq=use_chisq, use_pois=use_pois,
                scale=scale, NB_probs=NB_probs, device=device
            )
        else:
            raise ValueError("supervision_type must be 'fixed_genes' or 'fixed_samples'")
        
    elif model_type == 'exponential_unsupervised':
        model = Exponential_base(
            D.shape[0], D.shape[1], num_patterns, 
            use_chisq=use_chisq, use_pois=use_pois, NB_probs=NB_probs, device=device
        )
    elif model_type == 'exponential_supervised':
        if supervision_type == 'fixed_genes':
            model = Exponential_SSFixedGenes(
                D.shape[0], D.shape[1], num_patterns, 
                fixed_patterns=fixed_patterns, use_chisq=use_chisq, use_pois=use_pois,
                NB_probs=NB_probs, device=device
            )
        elif supervision_type == 'fixed_samples':
            model = Exponential_SSFixedSamples(
                D.shape[0], D.shape[1], num_patterns, 
                fixed_patterns=fixed_patterns, use_chisq=use_chisq, use_pois=use_pois,
                NB_probs=NB_probs, device=device
            )
        else:
            raise ValueError("supervision_type must be 'fixed_genes' or 'fixed_samples'")
    else:
        raise ValueError("model_type must be 'gamma_unsupervised', 'gamma_supervised', 'exponential_unsupervised', or 'exponential_supervised'")
    
    # Setup guide and optimizer
    guide = AutoNormal(model)
    optimizer = pyro.optim.Adam({"lr": 0.1, "eps": 1e-08})
    loss_fn = pyro.infer.Trace_ELBO()
    
    svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=loss_fn)
    
    return model, guide, svi


def run_inference_loop(svi, model, D, U, num_steps, use_tensorboard_id=None, 
                      spatial=False, coords=None, plot_dims=None):
    """Run the SVI optimization loop with optional TensorBoard logging.

    Parameters
    ----------
    svi : pyro.infer.SVI
        Configured SVI object.
    model : pyro.nn.PyroModule
        Model instance being optimized.
    D : torch.Tensor
        Data tensor with shape ``(n_samples, n_genes)``.
    U : torch.Tensor
        Per-entry scale/uncertainty tensor used in chi-squared computation.
    num_steps : int
        Number of optimization steps.
    use_tensorboard_id : str or None, optional
        If provided, enable TensorBoard logging with this identifier.
    spatial : bool, optional
        If True, attempt to log spatial pattern plots.
    coords : array-like or None, optional
        Spatial coordinates for plotting.
    plot_dims : sequence of int or None, optional
        Grid dimensions ``[rows, cols]`` for spatial plots.

    Returns
    -------
    tuple
        ``(losses, steps, runtime, writer)`` where ``writer`` is a
        ``SummaryWriter`` or None.
    """
    start_time = datetime.now()
    steps = []
    losses = []
    writer = None
    
    if use_tensorboard_id is not None:
        print('Logging to Tensorboard')
        writer = SummaryWriter(comment=use_tensorboard_id)
    
    for step in range(1, num_steps + 1):
        try:
            loss = svi.step(D, U)
        except ValueError as e:
            print(f"ValueError during iteration {step}: {e}")
            break
        
        if step % 10 == 0:
            losses.append(loss)
            steps.append(step)
        
        if step % 100 == 0:
            print(f"Iteration {step}, ELBO loss: {loss}")
        
        if writer is not None:
            _log_tensorboard_metrics(writer, model, step, loss, spatial, coords, plot_dims)
    
    end_time = datetime.now()
    runtime = round((end_time - start_time).total_seconds())
    print(f'Runtime: {runtime} seconds')
    
    return losses, steps, runtime, writer


def _log_tensorboard_metrics(writer, model, step, loss, spatial=False, coords=None, plot_dims=None):
    """Log scalar metrics and optional plots to TensorBoard.

    Parameters
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard summary writer.
    model : pyro.nn.PyroModule
        Model instance.
    step : int
        Current optimization step.
    loss : float
        Current loss value.
    spatial : bool, optional
        If True, log spatial plots when possible.
    coords : array-like or None, optional
        Spatial coordinates for plotting.
    plot_dims : sequence of int or None, optional
        Grid dimensions ``[rows, cols]`` for spatial plots.
    """
    writer.add_scalar("Loss/train", loss, step)
    if hasattr(model, "best_chisq"):
        writer.add_scalar("Best chi-squared", float(getattr(model, "best_chisq", np.inf)), step)
        writer.add_scalar("Saved chi-squared iter", int(getattr(model, "best_chisq_iter", 0)), step)
    if hasattr(model, "chi2"):
        writer.add_scalar("Chi-squared", float(getattr(model, "chi2")), step)
    if hasattr(model, "pois"):
        writer.add_scalar("Poisson loss", float(getattr(model, "pois")), step)
    writer.flush()
    
    if step % 50 == 0:
        if spatial and coords is not None and plot_dims is not None:
            store = pyro.get_param_store()
            # Plot loc_P if available
            if 'loc_P' in store:
                try:
                    locP = pyro.param("loc_P").detach().cpu().numpy()
                    plot_grid(locP, coords, plot_dims[0], plot_dims[1], savename=None)
                    writer.add_figure("loc_P", plt.gcf(), step)
                except Exception:
                    pass

            # Plot current sampled P if available
            if hasattr(model, "P"):
                try:
                    plot_grid(model.P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename=None)
                    writer.add_figure("current sampled P", plt.gcf(), step)
                except Exception:
                    pass

        # Histograms of parameters
        store = pyro.get_param_store()
        if 'loc_P' in store:
            plt.figure()
            plt.hist(pyro.param("loc_P").detach().cpu().numpy().flatten(), bins=30)
            writer.add_figure("loc_P_hist", plt.gcf(), step)
        if 'loc_A' in store:
            plt.figure()
            plt.hist(pyro.param("loc_A").detach().cpu().numpy().flatten(), bins=30)
            writer.add_figure("loc_A_hist", plt.gcf(), step)
        if 'scale_A' in store:
            plt.figure()
            plt.hist(pyro.param("scale_A").detach().cpu().numpy().flatten(), bins=30)
            writer.add_figure("scale_A_hist", plt.gcf(), step)
        if 'scale_P' in store:
            plt.figure()
            plt.hist(pyro.param("scale_P").detach().cpu().numpy().flatten(), bins=30)
            writer.add_figure("scale_P_hist", plt.gcf(), step)

    if step % 100 == 0:
        if hasattr(model, "D_reconstructed"):
            plt.figure()
            D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
            plt.hist(D_reconstructed.flatten(), bins=30)
            writer.add_figure("D_reconstructed_hist", plt.gcf(), step)

def _detect_and_save_parameters(result_anndata, model, fixed_pattern_names=None, num_learned_patterns=None):
    """Auto-detect and persist model parameters into AnnData slots.

    Parameters
    ----------
    result_anndata : anndata.AnnData
        AnnData object to populate with results.
    model : pyro.nn.PyroModule
        Trained model instance.
    fixed_pattern_names : sequence of str or None, optional
        Names for fixed patterns (semi-supervised models).
    num_learned_patterns : int or None, optional
        Number of learned patterns used to construct column names.
    """
    store = pyro.get_param_store()
    learned_pattern_names = ["Pattern_" + str(x + 1) for x in range(num_learned_patterns)]
    if fixed_pattern_names is not None:
        full_pattern_names = [str(x) for x in fixed_pattern_names] + learned_pattern_names
   

    ### Save loc_A/P if present (Gamma models)
    if "loc_P" in store:
        loc_P = pd.DataFrame(pyro.param("loc_P").detach().cpu().numpy())
        if loc_P.shape[1] != len(learned_pattern_names): # this was semisupervised then
            loc_P.columns = full_pattern_names
        else:
            loc_P.columns = learned_pattern_names
        loc_P.index = result_anndata.obs.index
        result_anndata.obsm["loc_P"] = loc_P
        print("Saving loc_P in anndata.obsm['loc_P']")
    if "loc_A" in store:
        loc_A = pd.DataFrame(pyro.param("loc_A").detach().cpu().numpy().T)
        if loc_A.shape[1] != len(learned_pattern_names): # this was semisupervised then
            loc_A.columns = full_pattern_names
        else:
            loc_A.columns = learned_pattern_names
        loc_A.index = result_anndata.var.index
        result_anndata.varm["loc_A"] = loc_A
        print("Saving loc_A in anndata.varm['loc_A']")
    

    ### Save scale_A/P (scalar) if present (Exponential models)
    if "scale_P" in store:
        scale_P_val = pyro.param("scale_P").detach().cpu().item()
        result_anndata.uns["scale_P"] = scale_P_val
        print(f"Saving scale_P = {scale_P_val} in anndata.uns['scale_P']")
    if "scale_A" in store:
        scale_A_val = pyro.param("scale_A").detach().cpu().item()
        result_anndata.uns["scale_A"] = scale_A_val
        print(f"Saving scale_A = {scale_A_val} in anndata.uns['scale_A']")

    
    ### Save last sampled A/P
    if hasattr(model, "P"):
        last_P = pd.DataFrame(model.P.detach().cpu().numpy())
        if last_P.shape[1] != len(learned_pattern_names): # this was semisupervised then
            last_P.columns = full_pattern_names
        else:
            last_P.columns = learned_pattern_names
        last_P.index = result_anndata.obs.index
        result_anndata.obsm["last_P"] = last_P
        print("Saving final sampled P in anndata.obsm['last_P']")
    if hasattr(model, "A"):
        last_A = pd.DataFrame(model.A.detach().cpu().numpy().T)
        if last_A.shape[1] != len(learned_pattern_names): # this was semisupervised then
            last_A.columns = full_pattern_names
        else:
            last_A.columns = learned_pattern_names
        last_A.index = result_anndata.var.index
        result_anndata.varm["last_A"] = last_A
        print("Saving final sampled A in anndata.varm['last_A']")


    ### Save A/P_total for supervised models 
    if hasattr(model, "P_total"):
        P_total = pd.DataFrame(model.P_total.detach().cpu().numpy())
        P_total.columns = full_pattern_names
        P_total.index = result_anndata.obs.index
        result_anndata.obsm["P_total"] = P_total
        print("Saving P_total in anndata.obsm['P_total']")
    if hasattr(model, "A_total"):
        A_total = pd.DataFrame(model.A_total.detach().cpu().numpy().T)
        A_total.columns = full_pattern_names
        A_total.index = result_anndata.var.index
        result_anndata.varm["A_total"] = A_total
        print("Saving A_total in anndata.varm['A_total']")


    ### Save fixed A/P for supervised models
    if hasattr(model, "fixed_P"):
        fixed_P = pd.DataFrame(
            model.fixed_P.detach().cpu().numpy(), 
            columns=[str(p) for p in fixed_pattern_names], # make sure they are strings
            index=result_anndata.obs.index
        )
        result_anndata.obsm["fixed_P"] = fixed_P
        print("Saving fixed P in anndata.obsm['fixed_P']")
    if hasattr(model, "fixed_A"):
        fixed_A = pd.DataFrame(
            model.fixed_A.detach().cpu().numpy(), 
            columns=[str(p) for p in fixed_pattern_names], # make sure they are strings
            index=result_anndata.var.index
        )
        result_anndata.varm["fixed_A"] = fixed_A
        print("Saving fixed A in anndata.varm['fixed_A']")

    ##### Save 'best' parameters based on chi2 ####

    ### Save loc_A/P if present (Gamma models)
    if hasattr(model, "best_locP"):
        best_locP = pd.DataFrame(model.best_locP.detach().cpu().numpy())
        if best_locP.shape[1] != len(learned_pattern_names): # this was semisupervised then
            best_locP.columns = full_pattern_names
        else:
            best_locP.columns = learned_pattern_names
        best_locP.index = result_anndata.obs.index
        result_anndata.obsm["best_locP"] = best_locP
        print("Saving best_locP in anndata.obsm['best_locP']")
    if hasattr(model, "best_locA"):
        best_locA = pd.DataFrame(model.best_locA.detach().cpu().numpy().T)
        if best_locA.shape[1] != len(learned_pattern_names): # this was semisupervised then
            best_locA.columns = full_pattern_names
        else:
            best_locA.columns = learned_pattern_names
        best_locA.index = result_anndata.var.index
        result_anndata.varm["best_locA"] = best_locA
        print("Saving best_locA in anndata.varm['best_locA']")    

    ### Save scale_A/P (scalar) if present (Exponential models)
    if hasattr(model, "best_scaleP"):
        scale_P_val = model.best_scaleP.detach().cpu().item()
        result_anndata.uns["best_scale_P"] = scale_P_val
        print(f"Saving best_scale_P = {scale_P_val} in anndata.uns['best_scale_P']")
    if hasattr(model, "best_scaleA"):
        scale_A_val = model.best_scaleA.detach().cpu().item()
        result_anndata.uns["best_scale_A"] = scale_A_val
        print(f"Saving best_scale_A = {scale_A_val} in anndata.uns['best_scale_A']")

    ### Save best sampled A/P
    if hasattr(model, "best_P"):
        best_P = pd.DataFrame(model.best_P.detach().cpu().numpy())
        if best_P.shape[1] != len(learned_pattern_names): # this was semisupervised then
            best_P.columns = full_pattern_names
        else:
            best_P.columns = learned_pattern_names
        best_P.index = result_anndata.obs.index
        result_anndata.obsm["best_P"] = best_P
        print("Saving final sampled P in anndata.obsm['best_P']")
    if hasattr(model, "best_A"):
        best_A = pd.DataFrame(model.best_A.detach().cpu().numpy().T)
        if best_A.shape[1] != len(learned_pattern_names): # this was semisupervised then
            best_A.columns = full_pattern_names
        else:
            best_A.columns = learned_pattern_names
        best_A.index = result_anndata.var.index
        result_anndata.varm["best_A"] = best_A
        print("Saving final sampled A in anndata.varm['best_A']")

    ### Save best sampled A/P with fixed patterns for supervised models
    if "fixed_P" in result_anndata.obsm and "best_P" in result_anndata.obsm:
        best_P_total = result_anndata.obsm["fixed_P"].merge(result_anndata.obsm["best_P"], left_index=True, right_index=True)
        result_anndata.obsm["best_P_total"] = best_P_total
        print("Saving best_P_total in anndata.obsm['best_P_total']")
    if "fixed_A" in result_anndata.varm and "best_A" in result_anndata.varm:
        best_A_total = result_anndata.varm["fixed_A"].merge(result_anndata.varm["best_A"], left_index=True, right_index=True)
        result_anndata.varm["best_A_total"] = best_A_total
        print("Saving best_A_total in anndata.varm['best_A_total']")


def save_results_to_anndata(result_anndata, model, losses, steps, runtime, scale, settings, 
                           fixed_pattern_names=None, num_learned_patterns=None, supervised=None):
    """Save inference outputs into AnnData ``obsm``, ``varm``, and ``uns``.

    Parameters
    ----------
    result_anndata : anndata.AnnData
        AnnData object to populate with results (typically a copy of input).
    model : pyro.nn.PyroModule
        Trained model instance.
    losses : list[float]
        Loss values recorded during training.
    steps : list[int]
        Optimization steps corresponding to ``losses``.
    runtime : int
        Runtime in seconds.
    scale : torch.Tensor or float
        Scale factor used for Gamma-based models.
    settings : dict
        Training settings metadata.
    fixed_pattern_names : sequence of str or None, optional
        Names for fixed patterns in semi-supervised runs.
    num_learned_patterns : int or None, optional
        Number of learned patterns used to generate column names.
    supervised : str or None, optional
        Included for compatibility; saved results are auto-detected.

    Returns
    -------
    anndata.AnnData
        The AnnData object with results stored under:

        - ``obsm``: ``loc_P``, ``last_P``, ``best_P``, and variants.
        - ``varm``: ``loc_A``, ``last_A``, ``best_A``, and variants.
        - ``uns``: training metadata, losses, settings, and scales.
    """
    # Save P parameters
    _detect_and_save_parameters(result_anndata, model, fixed_pattern_names, num_learned_patterns)
    
    # Save metadata
    result_anndata.uns["runtime (seconds)"] = runtime
    result_anndata.uns["loss"] = pd.DataFrame(losses, index=steps, columns=["loss"])
    if hasattr(model, "best_chisq_iter"):
        result_anndata.uns["step_w_bestChisq"] = model.best_chisq_iter
    if hasattr(model, "best_chisq"):
        result_anndata.uns["best_chisq"] = float(model.best_chisq)
    result_anndata.uns["scale"] = scale.detach().cpu().item()
    result_anndata.uns["settings"] = pd.DataFrame(
        list(settings.values()), 
        index=list(settings.keys()), 
        columns=["settings"]
    )
    
    return result_anndata


def create_settings_dict(num_patterns, num_steps, device, NB_probs, use_chisq, scale, 
                        model_type, use_tensorboard_id=None, writer=None):
    """Assemble a settings dictionary for persistence in ``AnnData.uns``.

    Parameters
    ----------
    num_patterns : int
        Number of patterns learned.
    num_steps : int
        Number of training steps.
    device : torch.device
        Device used for training.
    NB_probs : float
        Negative Binomial probability parameter.
    use_chisq : bool
        Whether chi-squared loss was used.
    scale : float
        Scale factor used by Gamma-based models.
    model_type : str
        Model type string (e.g., ``gamma_unsupervised``).
    use_tensorboard_id : str or None, optional
        TensorBoard identifier string.
    writer : SummaryWriter or None, optional
        TensorBoard writer (used to record log dir).

    Returns
    -------
    dict
        Settings dictionary suitable for ``AnnData.uns``.
    """
    settings = {
        'num_patterns': str(num_patterns),
        'num_steps': str(num_steps),
        'device': str(device),
        'NB_probs': str(NB_probs),
        'use_chisq': str(use_chisq),
        'scale': str(scale),
        'model_type': str(model_type)
    }
    
    if use_tensorboard_id is not None and writer is not None:
        settings['tensorboard_identifier'] = str(writer.log_dir)
        writer.flush()
    
    return settings


def run_nmf_unsupervised(data, num_patterns, num_steps=20000, device=None, NB_probs=0.5, 
                        use_chisq=False, use_pois=False, use_tensorboard_id=None, spatial=False, 
                        plot_dims=None, scale=None, model_family='gamma'):
    """Run unsupervised NMF analysis on an AnnData object.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object with raw counts in ``.X``.
    num_patterns : int
        Number of latent patterns to learn.
    num_steps : int, optional
        Number of optimization steps.
    device : str or torch.device or None, optional
        Target device (``'cpu'``, ``'cuda'``, ``'mps'``). If None, autodetect.
    NB_probs : float, optional
        Negative Binomial probability parameter.
    use_chisq : bool, optional
        If True, include chi-squared loss term.
    use_pois : bool, optional
        If True, include Poisson log-likelihood term.
    use_tensorboard_id : str or None, optional
        TensorBoard logging identifier. If None, logging is disabled.
    spatial : bool, optional
        If True, use ``obsm['spatial']`` for plotting and logging.
    plot_dims : sequence of int or None, optional
        Grid dimensions ``[rows, cols]`` for spatial plots.
    scale : float or None, optional
        Scale factor for Gamma models. Currently computed from data
        regardless of this value.
    model_family : {'gamma', 'exponential'}, optional
        Model family to use.

    Returns
    -------
    anndata.AnnData
        Copy of the input AnnData with results saved into ``obsm``,
        ``varm``, and ``uns``.
    """
    coords = validate_data(data, spatial, plot_dims, num_patterns)
    D, U, scale, device = prepare_tensors(data, device)
    print(f"D shape: {D.shape}, U shape: {U.shape}")
    
    model_type = f'{model_family}_unsupervised'
    model, guide, svi = setup_model_and_optimizer(
        D, num_patterns, scale, NB_probs, use_chisq, use_pois, device, model_type=model_type
    )
    print(f'Plotting with spatial coordinates: {spatial}')
    
    losses, steps, runtime, writer = run_inference_loop(
        svi, model, D, U, num_steps, use_tensorboard_id, spatial, coords, plot_dims
    )
    
    settings = create_settings_dict(
        num_patterns, num_steps, device, NB_probs, use_chisq, scale, 
        model_type, use_tensorboard_id, writer
    )
    
    result_anndata = data.copy()
    result_anndata = save_results_to_anndata(
        result_anndata, model, losses, steps, runtime, scale, settings, num_learned_patterns=num_patterns, supervised=None
    )
    
    pyro.clear_param_store()
    return result_anndata


def run_nmf_supervised(data, num_patterns, fixed_patterns, num_steps=20000, device=None, 
                      NB_probs=0.5, use_chisq=False, use_pois=False, use_tensorboard_id=None, 
                      spatial=False, plot_dims=None, scale=None, model_family='gamma',
                      supervision_type='fixed_genes'):
    """Run semi-supervised NMF analysis with fixed patterns.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object with raw counts in ``.X``.
    num_patterns : int
        Number of additional patterns to learn.
    fixed_patterns : pandas.DataFrame
        Fixed patterns. Shape depends on ``supervision_type``:

        - ``fixed_genes``: ``(n_genes, n_fixed_patterns)``
        - ``fixed_samples``: ``(n_samples, n_fixed_patterns)``
    num_steps : int, optional
        Number of optimization steps.
    device : str or torch.device or None, optional
        Target device (``'cpu'``, ``'cuda'``, ``'mps'``). If None, autodetect.
    NB_probs : float, optional
        Negative Binomial probability parameter.
    use_chisq : bool, optional
        If True, include chi-squared loss term.
    use_pois : bool, optional
        If True, include Poisson log-likelihood term.
    use_tensorboard_id : str or None, optional
        TensorBoard logging identifier. If None, logging is disabled.
    spatial : bool, optional
        If True, use ``obsm['spatial']`` for plotting and logging.
    plot_dims : sequence of int or None, optional
        Grid dimensions ``[rows, cols]`` for spatial plots.
    scale : float or None, optional
        Scale factor for Gamma models. Currently computed from data
        regardless of this value.
    model_family : {'gamma', 'exponential'}, optional
        Model family to use.
    supervision_type : {'fixed_genes', 'fixed_samples'}, optional
        Whether fixed patterns are provided across genes or samples.

    Returns
    -------
    anndata.AnnData
        Copy of the input AnnData with results saved into ``obsm``,
        ``varm``, and ``uns``.
    """
    coords = validate_data(data, spatial, plot_dims, num_patterns)
    D, U, scale, device = prepare_tensors(data, device)
    
    fixed_pattern_names = list(fixed_patterns.columns)
    fixed_patterns_tensor = torch.tensor(fixed_patterns.to_numpy(), dtype=torch.float32).to(device)
    
    model_type = f'{model_family}_supervised'
    model, guide, svi = setup_model_and_optimizer(
        D, num_patterns, scale, NB_probs, use_chisq, use_pois, device, 
        fixed_patterns_tensor, model_type, supervision_type
    )
    
    losses, steps, runtime, writer = run_inference_loop(
        svi, model, D, U, num_steps, use_tensorboard_id, spatial, coords, plot_dims
    )
    
    settings = create_settings_dict(
        num_patterns, num_steps, device, NB_probs, use_chisq, scale, 
        model_type, use_tensorboard_id, writer
    )
    
    result_anndata = data.copy()
    result_anndata = save_results_to_anndata(
        result_anndata, model, losses, steps, runtime, scale, settings, fixed_pattern_names=fixed_pattern_names, num_learned_patterns=num_patterns, supervised=supervision_type
    )
    
    pyro.clear_param_store()
    return result_anndata
