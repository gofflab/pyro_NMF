import torch
from pyroNMF.models.gamma_NB_newBase import Gamma_NegBinomial_base
from pyroNMF.models.gamma_NB_new_SSfixedP import Gamma_NegBinomial_SSFixed
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
    D = torch.tensor(data.X)
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
    
    D = torch.tensor(data.X).to(device)
    U = (D * 0.1).clip(min=0.3).to(device)  # Probability for NegativeBinomial
    scale = (D.cpu().numpy().std()) * 2
    
    return D, U, scale, device


def setup_model_and_optimizer(D, num_patterns, scale, NB_probs, use_chisq, device, 
                             fixed_patterns=None, model_type='unsupervised'):
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
    - model_type: 'unsupervised' or 'supervised'
    
    Returns:
    - model: Initialized model
    - guide: AutoNormal guide
    - svi: SVI optimizer
    """
    # Instantiate the model
    if model_type == 'unsupervised':
        model = Gamma_NegBinomial_base(
            D.shape[0], D.shape[1], num_patterns, 
            use_chisq=use_chisq, scale=scale, 
            NB_probs=NB_probs, device=device
        )
    elif model_type == 'supervised':
        model = Gamma_NegBinomial_SSFixed(
            D.shape[0], D.shape[1], num_patterns, 
            fixed_patterns=fixed_patterns, use_chisq=use_chisq, 
            scale=scale, NB_probs=NB_probs, device=device
        )
    else:
        raise ValueError("model_type must be 'unsupervised' or 'supervised'")
    
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
    writer.add_scalar('Best chi-squared', model.best_chisq, step)
    writer.add_scalar('Saved chi-squared iter', model.best_chisq_iter, step)
    writer.add_scalar("Chi-squared", model.chi2, step)
    writer.flush()
    
    if step % 50 == 0:
        if spatial and coords is not None and plot_dims is not None:
            # Plot loc P
            plot_grid(
                pyro.param("loc_P").detach().to('cpu').numpy(), 
                coords, plot_dims[0], plot_dims[1], savename=None
            )
            writer.add_figure("loc_P", plt.gcf(), step)
            
            # Plot current sampled P
            plot_grid(
                model.P.detach().cpu().numpy(), 
                coords, plot_dims[0], plot_dims[1], savename=None
            )
            writer.add_figure("current sampled P", plt.gcf(), step)
        
        # Histograms
        plt.figure()
        plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("loc_P_hist", plt.gcf(), step)
        
        plt.figure()
        plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
        writer.add_figure("loc_A_hist", plt.gcf(), step)
    
    if step % 100 == 0:
        plt.figure()
        D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
        plt.hist(D_reconstructed.flatten(), bins=30)
        writer.add_figure("D_reconstructed_hist", plt.gcf(), step)


def save_results_to_anndata(data, model, losses, steps, runtime, scale, settings, 
                           model_type='unsupervised', fixed_pattern_names=None):
    """
    Save results to AnnData object.
    
    Parameters:
    - data: Original AnnData object
    - model: Trained model
    - losses: Training losses
    - steps: Training steps
    - runtime: Training runtime
    - scale: Scale factor used
    - settings: Training settings
    - model_type: 'unsupervised' or 'supervised'
    - fixed_pattern_names: Names of fixed patterns (for supervised)
    
    Returns:
    - result_anndata: AnnData object with results
    """
    result_anndata = data.copy()
    
    # Common results
    _save_common_results(result_anndata, model, losses, steps, runtime, scale, settings)
    
    # Model-specific results
    if model_type == 'unsupervised':
        _save_unsupervised_results(result_anndata, model)
    elif model_type == 'supervised':
        _save_supervised_results(result_anndata, model, fixed_pattern_names)
    #pyro.clear_param_store()
    return result_anndata


def _save_common_results(result_anndata, model, losses, steps, runtime, scale, settings):
    """Save common results for both supervised and unsupervised models."""
    # Save loc_P
    loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
    loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
    loc_P.index = result_anndata.obs.index
    result_anndata.obsm['loc_P'] = loc_P
    print("Saving loc_P in anndata.obsm['loc_P']")
    
    # Save last sampled P
    last_P = pd.DataFrame(model.P.detach().cpu().numpy())
    last_P.columns = ['Pattern_' + str(x+1) for x in last_P.columns]
    last_P.index = result_anndata.obs.index
    result_anndata.obsm['last_P'] = last_P
    print("Saving final sampled P in anndata.obsm['last_P']")
    
    # Save best P
    best_P = pd.DataFrame(model.best_P.detach().cpu().numpy())
    best_P.columns = ['Pattern_' + str(x+1) for x in best_P.columns]
    best_P.index = result_anndata.obs.index
    result_anndata.obsm['best_P'] = best_P
    print("Saving best P via chi2 in anndata.obsm['best_P']")
    
    # Save best locP
    best_locP = pd.DataFrame(model.best_locP.detach().cpu().numpy())
    best_locP.columns = ['Pattern_' + str(x+1) for x in best_locP.columns]
    best_locP.index = result_anndata.obs.index
    result_anndata.obsm['best_locP'] = best_locP
    print("Saving best loc P via chi2 in anndata.obsm['best_locP']")
    
    # Save metadata
    result_anndata.uns['runtime (seconds)'] = runtime
    result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])
    result_anndata.uns['step_w_bestChisq'] = model.best_chisq_iter
    result_anndata.uns['scale'] = scale
    result_anndata.uns['settings'] = pd.DataFrame(
        list(settings.values()), 
        index=list(settings.keys()), 
        columns=['settings']
    )


def _save_unsupervised_results(result_anndata, model):
    """Save results specific to unsupervised model."""
    # Save loc_A
    loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
    loc_A.columns = ['Pattern_' + str(x+1) for x in loc_A.columns]
    loc_A.index = result_anndata.var.index
    result_anndata.varm['loc_A'] = loc_A
    print("Saving loc_A in anndata.varm['loc_A']")
    
    # Save last sampled A
    last_A = pd.DataFrame(model.A.detach().cpu().numpy()).T
    last_A.columns = ['Pattern_' + str(x+1) for x in last_A.columns]
    last_A.index = result_anndata.var.index
    result_anndata.varm['last_A'] = last_A
    print("Saving final sampled A in anndata.varm['last_A']")
    
    # Save best A
    best_A = pd.DataFrame(model.best_A.detach().cpu().numpy()).T
    best_A.columns = ['Pattern_' + str(x+1) for x in best_A.columns]
    best_A.index = result_anndata.var.index
    result_anndata.varm['best_A'] = best_A
    print("Saving best A via chi2 in anndata.varm['best_A']")
    
    # Save best locA
    best_locA = pd.DataFrame(model.best_locA.detach().cpu().numpy()).T
    best_locA.columns = ['Pattern_' + str(x+1) for x in best_locA.columns]
    best_locA.index = result_anndata.var.index
    result_anndata.varm['best_locA'] = best_A  # Note: This looks like a bug in original code
    print("Saving best loc A via chi2 in anndata.varm['best_locA']")


def _save_supervised_results(result_anndata, model, fixed_pattern_names):
    """Save results specific to supervised model."""
    num_patterns = model.P.shape[1]
    
    # Save loc_A
    loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
    loc_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    loc_A.index = result_anndata.var.index
    result_anndata.varm['loc_A'] = loc_A
    print("Saving loc_A in anndata.varm['loc_A']")
    
    # Save last sampled A
    last_A = pd.DataFrame(model.A.detach().cpu().numpy()).T
    last_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    last_A.index = result_anndata.var.index
    result_anndata.varm['last_A'] = last_A
    print("Saving final sampled A in anndata.varm['last_A']")
    
    # Save fixed P
    fixed_P = pd.DataFrame(
        model.fixed_P.detach().cpu().numpy(), 
        columns=fixed_pattern_names, 
        index=result_anndata.obs.index
    )
    result_anndata.obsm['fixed_P'] = fixed_P
    print("Saving fixed P in anndata.obsm['fixed_P']")
    
    # Save best P total (fixed + learned)
    best_P = result_anndata.obsm['best_P']
    best_P_total = fixed_P.merge(best_P, left_index=True, right_index=True)
    result_anndata.obsm['best_P_total'] = best_P_total
    print("Saving best P total in anndata.obsm['best_P_total']")
    
    # Save best A
    best_A = pd.DataFrame(model.best_A.detach().cpu().numpy()).T
    best_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    best_A.index = result_anndata.var.index
    result_anndata.varm['best_A'] = best_A
    print("Saving best A via chi2 in anndata.varm['best_A']")
    
    # Save best locA
    best_locA = pd.DataFrame(model.best_locA.detach().cpu().numpy()).T
    best_locA.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    best_locA.index = result_anndata.var.index
    result_anndata.varm['best_locA'] = best_A  # Note: This looks like a bug in original code
    print("Saving best loc A via chi2 in anndata.varm['best_locA']")


def create_settings_dict(num_patterns, num_steps, device, NB_probs, use_chisq, scale, 
                        use_tensorboard_id=None, writer=None):
    """
    Create settings dictionary for saving.
    
    Parameters:
    - num_patterns: Number of patterns
    - num_steps: Number of training steps
    - device: Device used
    - NB_probs: Negative binomial probability
    - use_chisq: Whether chi-squared loss was used
    - scale: Scale factor
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
        'scale': str(scale)
    }
    
    if use_tensorboard_id is not None and writer is not None:
        settings['tensorboard_identifier'] = str(writer.log_dir)
        writer.flush()
    
    return settings


def run_nmf_unsupervised(data, num_patterns, num_steps=20000, device=None, NB_probs=0.5, 
                        use_chisq=False, use_tensorboard_id=None, spatial=False, 
                        plot_dims=None, scale=None):
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
    
    Returns:
    - result_anndata: AnnData object with results
    """
    # Validate data and setup
    coords = validate_data(data, spatial, plot_dims, num_patterns)
    D, U, scale, device = prepare_tensors(data, device)
    
    # Setup model and optimizer
    model, guide, svi = setup_model_and_optimizer(
        D, num_patterns, scale, NB_probs, use_chisq, device
    )
    print(f'Plotting with spatial coordinates: {spatial}')
    # Run inference
    losses, steps, runtime, writer = run_inference_loop(
        svi, model, D, U, num_steps, use_tensorboard_id, spatial, coords, plot_dims
    )
    
    # Create settings and save results
    settings = create_settings_dict(
        num_patterns, num_steps, device, NB_probs, use_chisq, scale, 
        use_tensorboard_id, writer
    )
    
    result_anndata = save_results_to_anndata(
        data, model, losses, steps, runtime, scale, settings, 'unsupervised'
    )
    pyro.clear_param_store()
    return result_anndata


def run_nmf_supervised(data, num_patterns, fixed_patterns, num_steps=20000, device=None, 
                      NB_probs=0.5, use_chisq=False, use_tensorboard_id=None, 
                      spatial=False, plot_dims=None, scale=None):
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
    
    Returns:
    - result_anndata: AnnData object with results
    """
    # Validate data and setup
    coords = validate_data(data, spatial, plot_dims, num_patterns)
    D, U, scale, device = prepare_tensors(data, device)
    
    # Prepare fixed patterns
    fixed_pattern_names = list(fixed_patterns.columns)
    fixed_patterns_tensor = torch.tensor(fixed_patterns.to_numpy()).to(device)
    
    # Setup model and optimizer
    model, guide, svi = setup_model_and_optimizer(
        D, num_patterns, scale, NB_probs, use_chisq, device, 
        fixed_patterns_tensor, 'supervised'
    )
    
    # Run inference
    losses, steps, runtime, writer = run_inference_loop(
        svi, model, D, U, num_steps, use_tensorboard_id, spatial, coords, plot_dims
    )
    
    # Create settings and save results
    settings = create_settings_dict(
        num_patterns, num_steps, device, NB_probs, use_chisq, scale, 
        use_tensorboard_id, writer
    )
    
    result_anndata = save_results_to_anndata(
        data, model, losses, steps, runtime, scale, settings, 
        'supervised', fixed_pattern_names
    )
    pyro.clear_param_store()
    return result_anndata


# Convenience alias for backward compatibility
def run_nmf_supervisedP(data, num_patterns, fixed_patterns, num_steps=20000, device=None, 
                       NB_probs=0.5, use_chisq=False, use_tensorboard_id=None, 
                       spatial=False, plot_dims=None, scale=None):
    """Alias for run_nmf_supervised for backward compatibility."""
    return run_nmf_supervised(
        data, num_patterns, fixed_patterns, num_steps, device, 
        NB_probs, use_chisq, use_tensorboard_id, spatial, plot_dims, scale
    )