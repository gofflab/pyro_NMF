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
    """
    Validate input data and spatial coordinates.
    
    Parameters:
    - data: AnnData object containing the data
    - spatial: Whether spatial analysis is requested
    - plot_dims: Plotting dimensions [rows, cols]
    - num_patterns: Number of patterns for validation
    
    Returns:
    - coords: Spatial coordinates if spatial=True, else None
    
    Raises:
    - ValueError: If validation fails
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
    """
    Prepare and move tensors to the specified device.
    
    Parameters:
    - data: AnnData object containing the data
    - device: Target device ('cpu', 'cuda', 'mps', or None for auto-detection)
    
    Returns:
    - D: Data tensor
    - U: Probability tensor
    - scale: Scale factor
    - device: Selected device
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
    """
    Setup the NMF model and optimizer.
    
    Parameters:
    - D: Data tensor
    - num_patterns: Number of patterns
    - scale: Scale factor
    - NB_probs: Negative binomial probability
    - use_chisq: Whether to use chi-squared loss
    - device: Device to run on
    - fixed_patterns: Fixed patterns for supervised learning
    - model_type: 'gamma_unsupervised', 'gamma_supervised', 'exponential_unsupervised', 'exponential_supervised'
    - supervision_type: 'fixed_genes' or 'fixed_samples' (for supervised models)
    
    Returns:
    - model: Initialized model
    - guide: AutoNormal guide
    - svi: SVI optimizer
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
    """
    Run the inference loop with optional tensorboard logging.
    
    Parameters:
    - svi: SVI optimizer
    - model: The model
    - D: Data tensor
    - U: Probability tensor
    - num_steps: Number of optimization steps
    - use_tensorboard_id: Tensorboard identifier
    - spatial: Whether to include spatial plots
    - coords: Spatial coordinates
    - plot_dims: Plotting dimensions
    
    Returns:
    - losses: List of loss values
    - steps: List of step numbers
    - runtime: Runtime in seconds
    - writer: Tensorboard writer (if used)
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
    """
    Log metrics to tensorboard.
    
    Parameters:
    - writer: Tensorboard writer
    - model: The model
    - step: Current step
    - loss: Current loss
    - spatial: Whether to include spatial plots
    - coords: Spatial coordinates
    - plot_dims: Plotting dimensions
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
    
    if step % 50 == 0 or step==1: # I want to see first sample too
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
    """
    Auto-detect and save all model parameters from model and param store.
    
    Parameters:
    - result_anndata: AnnData object to save to
    - model: The trained model
    - fixed_pattern_names: Names of fixed patterns (for supervised)
    - num_learned_patterns: Number of learned patterns (for supervised)
    - supervised: 'fixed_genes' or 'fixed_samples' or None

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


    ### Scale patterns to 1 and adjust gene weights accordingly
    # Scale both best and last sampled pattern, but need to use total patterns for supervised models since these are the same size
    if "best_P_total" in result_anndata.obsm:
        P_to_scale = result_anndata.obsm["best_P_total"]
    elif "best_P" in result_anndata.obsm:
        P_to_scale = result_anndata.obsm["best_P"]
    P_scaled = P_to_scale.div(P_to_scale.sum(axis=0), axis=1)
    result_anndata.obsm["best_P_scaled"] = P_scaled
    print("Saving best_P_scaled in anndata.obsm['best_P_scaled']")
        
    # Adjust A accordingly
    if "best_A_total" in result_anndata.varm:
        A_to_adjust = result_anndata.varm["best_A_total"]
    elif "best_A" in result_anndata.varm:
        A_to_adjust = result_anndata.varm["best_A"]
    A_adjusted = A_to_adjust.multiply(P_to_scale.sum(axis=0), axis=1)
    result_anndata.varm["best_A_scaled"] = A_adjusted
    print("Saving best_A_scaled in anndata.varm['best_A_scaled']")

    # Scale both best and last sampled pattern, but need to use total patterns for supervised models since these are the same size
    if "P_total" in result_anndata.obsm:
        P_to_scale = result_anndata.obsm["P_total"]
    elif "last_P" in result_anndata.obsm:
        P_to_scale = result_anndata.obsm["last_P"]
    P_scaled = P_to_scale.div(P_to_scale.sum(axis=0), axis=1)
    result_anndata.obsm["last_P_scaled"] = P_scaled
    print("Saving last_P_scaled in anndata.obsm['last_P_scaled']")
        
    # Adjust A accordingly
    if "A_total" in result_anndata.varm:
        A_to_adjust = result_anndata.varm["A_total"]
    elif "last_A" in result_anndata.varm:
        A_to_adjust = result_anndata.varm["last_A"]
    A_adjusted = A_to_adjust.multiply(P_to_scale.sum(axis=0), axis=1)
    result_anndata.varm["last_A_scaled"] = A_adjusted
    print("Saving last_A_scaled in anndata.varm['last_A_scaled']")



def save_results_to_anndata(result_anndata, model, losses, steps, runtime, scale, settings, 
                           fixed_pattern_names=None, num_learned_patterns=None, supervised=None):
    """
    Save results to AnnData object with auto-detection of parameters.
    
    Parameters:
    - result_anndata: AnnData object to save to
    - model: Trained model
    - losses: Training losses
    - steps: Training steps
    - runtime: Training runtime
    - scale: Scale factor used
    - settings: Training settings
    - fixed_pattern_names: Names of fixed patterns (for supervised)
    - num_learned_patterns: Number of learned patterns (for supervised)
    
    Returns:
    - result_anndata: AnnData object with results
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
    """
    Create settings dictionary for saving.
    
    Parameters:
    - num_patterns: Number of patterns
    - num_steps: Number of training steps
    - device: Device used
    - NB_probs: Negative binomial probability
    - use_chisq: Whether chi-squared loss was used
    - scale: Scale factor
    - model_type: Type of model used
    - use_tensorboard_id: Tensorboard identifier
    - writer: Tensorboard writer
    
    Returns:
    - settings: Dictionary of settings
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
    """
    Run unsupervised NMF analysis.
    
    Parameters:
    - data: AnnData object with raw counts in X
    - num_patterns: Number of patterns to extract
    - num_steps: Number of optimization steps
    - device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
    - NB_probs: Negative binomial probability
    - use_chisq: Whether to use chi-squared loss
    - use_tensorboard_id: Tensorboard logging identifier
    - spatial: Whether to include spatial analysis
    - plot_dims: Plotting dimensions [rows, cols]
    - scale: Scale factor (computed automatically if None)
    - model_family: 'gamma' or 'exponential'
    
    Returns:
    - result_anndata: AnnData object with results
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
    """
    Run supervised NMF analysis with fixed patterns.
    
    Parameters:
    - data: AnnData object with raw counts in X
    - num_patterns: Number of additional patterns to learn
    - fixed_patterns: DataFrame with fixed patterns
    - num_steps: Number of optimization steps
    - device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
    - NB_probs: Negative binomial probability
    - use_chisq: Whether to use chi-squared loss
    - use_tensorboard_id: Tensorboard logging identifier
    - spatial: Whether to include spatial analysis
    - plot_dims: Plotting dimensions [rows, cols]
    - scale: Scale factor (computed automatically if None)
    - model_family: 'gamma' or 'exponential'
    - supervision_type: 'fixed_genes' or 'fixed_samples'
    
    Returns:
    - result_anndata: AnnData object with results
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
