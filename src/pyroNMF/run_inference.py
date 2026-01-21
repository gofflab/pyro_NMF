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
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoNormal
import math

default_dtype = torch.float32


def _param_AP_guide(model):
    def guide(D, U):
        num_samples = model.num_samples
        if hasattr(model, "fixed_P"):
            num_A = model.num_fixed_patterns + model.num_patterns
        else:
            num_A = model.num_patterns
        if hasattr(model, "fixed_A"):
            num_P = model.num_fixed_patterns + model.num_patterns
        else:
            num_P = model.num_patterns

        loc_A = pyro.param(
            "q_loc_A",
            torch.zeros(num_A, model.num_genes, device=model.device, dtype=default_dtype),
        )
        scale_A = pyro.param(
            "q_scale_A",
            torch.full((num_A, model.num_genes), 0.1, device=model.device, dtype=default_dtype),
            constraint=dist.constraints.positive,
        )
        with pyro.plate("patterns", num_A, dim=-2):
            with pyro.plate("genes", model.num_genes, dim=-1):
                pyro.sample("A", dist.LogNormal(loc_A, scale_A))

        storage_device = getattr(model, "storage_device", model.device)
        loc_full = pyro.param(
            "q_loc_P",
            torch.zeros(num_samples, num_P, device=storage_device, dtype=default_dtype),
        )
        scale_full = pyro.param(
            "q_scale_P",
            torch.full((num_samples, num_P), 0.1, device=storage_device, dtype=default_dtype),
            constraint=dist.constraints.positive,
        )

        if model.batch_size is None or model.batch_size >= num_samples:
            sample_plate = pyro.plate("samples", num_samples, dim=-2)
        else:
            sample_plate = pyro.plate("samples", num_samples, dim=-2, subsample_size=model.batch_size)

        with sample_plate as batch_idx:
            idx_store = batch_idx
            if storage_device != batch_idx.device:
                idx_store = batch_idx.to(storage_device)
            loc_b = loc_full.index_select(0, idx_store)
            scale_b = scale_full.index_select(0, idx_store)
            if loc_b.device != model.device:
                loc_b = loc_b.to(model.device)
                scale_b = scale_b.to(model.device)
            with pyro.plate("patterns_P", num_P, dim=-1):
                pyro.sample("P", dist.LogNormal(loc_b, scale_b))

    return guide

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


def prepare_tensors(data, device=None, keep_on_cpu=False):
    """
    Prepare and move tensors to the specified device.
    
    Parameters:
    - data: AnnData object containing the data
    - device: Target device ('cpu', 'cuda', 'mps', or None for auto-detection)
    - keep_on_cpu: Keep data tensors on CPU even if a GPU device is selected
    
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
        D = torch.tensor(data.X.toarray(), dtype=default_dtype)
    else:
        print('Setting D')
        D = torch.tensor(data.X, dtype=default_dtype)
    if not keep_on_cpu:
        D = D.to(device)
    U = (D * 0.1).clip(min=0.3)
    if not keep_on_cpu:
        U = U.to(device)
    scale_device = torch.device('cpu') if keep_on_cpu else device
    scale = torch.tensor((D.cpu().numpy().std()) * 2, dtype=default_dtype, device=scale_device)
    
    return D, U, scale, device


def setup_model_and_optimizer(D, num_patterns, scale=1, NB_probs=0.5, use_chisq=False, use_pois=False, device=None,
                             fixed_patterns=None, model_type='gamma_unsupervised',
                             supervision_type=None, batch_size=None, lr=0.1, clip_norm=None, param_P=False):
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
    - batch_size: Optional minibatch size for sample-wise subsampling
    
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
            use_chisq=use_chisq,
            use_pois=use_pois,
            NB_probs=NB_probs,
            device=device,
            batch_size=batch_size,
            param_P=param_P,
        )
    elif model_type == 'exponential_supervised':
        if supervision_type == 'fixed_genes':
            model = Exponential_SSFixedGenes(
                D.shape[0], D.shape[1], num_patterns, 
                fixed_patterns=fixed_patterns, use_chisq=use_chisq, use_pois=use_pois,
                NB_probs=NB_probs,
                device=device,
                batch_size=batch_size,
                param_P=param_P,
            )
        elif supervision_type == 'fixed_samples':
            model = Exponential_SSFixedSamples(
                D.shape[0], D.shape[1], num_patterns, 
                fixed_patterns=fixed_patterns, use_chisq=use_chisq, use_pois=use_pois,
                NB_probs=NB_probs,
                device=device,
                batch_size=batch_size,
                param_P=param_P,
            )
        else:
            raise ValueError("supervision_type must be 'fixed_genes' or 'fixed_samples'")
    else:
        raise ValueError("model_type must be 'gamma_unsupervised', 'gamma_supervised', 'exponential_unsupervised', or 'exponential_supervised'")
    
    # Setup guide and optimizer
    if param_P and model_type.startswith("exponential"):
        guide = _param_AP_guide(model)
    else:
        guide = AutoNormal(model)
    if clip_norm is not None:
        optimizer = pyro.optim.ClippedAdam({"lr": lr, "eps": 1e-08, "clip_norm": clip_norm})
    else:
        optimizer = pyro.optim.Adam({"lr": lr, "eps": 1e-08})
    loss_fn = pyro.infer.Trace_ELBO()
    
    svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=loss_fn)
    
    return model, guide, svi


def run_inference_loop(svi, model, D, U, num_steps, use_tensorboard_id=None, 
                      spatial=False, coords=None, plot_dims=None, tb_max_points=5000):
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
            _log_tensorboard_metrics(writer, model, step, loss, spatial, coords, plot_dims, tb_max_points)
    
    end_time = datetime.now()
    runtime = round((end_time - start_time).total_seconds())
    print(f'Runtime: {runtime} seconds')
    
    return losses, steps, runtime, writer


def _log_tensorboard_metrics(writer, model, step, loss, spatial=False, coords=None, plot_dims=None, tb_max_points=5000):
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
    
    if step % 50 == 0:
        if spatial and coords is not None:
            store = pyro.get_param_store()
            if plot_dims is None:
                plot_dims = _infer_plot_dims(model, store)
            if plot_dims is None:
                return

            max_points = tb_max_points
            idx = None
            if max_points is not None and max_points > 0 and coords.shape[0] > max_points:
                rng = np.random.default_rng(0)
                idx = rng.choice(coords.shape[0], size=max_points, replace=False)
            coords_plot = coords if idx is None else coords[idx]
            point_size = _auto_point_size(coords_plot.shape[0])

            # Plot loc_P if available (gamma) or q_loc_P (exponential param_P)
            if 'loc_P' in store or 'q_loc_P' in store:
                try:
                    if 'loc_P' in store:
                        locP = pyro.param("loc_P").detach()
                    else:
                        q_loc = pyro.param("q_loc_P").detach()
                        q_scale = pyro.param("q_scale_P").detach()
                        locP = torch.exp(q_loc + 0.5 * torch.square(q_scale))
                    if idx is not None:
                        idx_t = torch.as_tensor(idx, device=locP.device)
                        locP = locP.index_select(0, idx_t)
                    locP = locP.cpu().numpy()
                    plot_grid(locP, coords_plot, plot_dims[0], plot_dims[1], size=point_size, savename=None)
                    writer.add_figure("loc_P", plt.gcf(), step)
                except Exception:
                    pass

            # Plot current sampled P if available
            if hasattr(model, "P"):
                try:
                    P = model.P.detach()
                    if idx is not None:
                        idx_t = torch.as_tensor(idx, device=P.device)
                        P = P.index_select(0, idx_t)
                    P = P.cpu().numpy()
                    plot_grid(P, coords_plot, plot_dims[0], plot_dims[1], size=point_size, savename=None)
                    writer.add_figure("current sampled P", plt.gcf(), step)
                except Exception:
                    pass

        # Histograms of parameters
        store = pyro.get_param_store()
        if 'loc_P' in store or 'q_loc_P' in store:
            plt.figure()
            if 'loc_P' in store:
                vals = pyro.param("loc_P").detach().cpu().numpy().flatten()
            else:
                q_loc = pyro.param("q_loc_P").detach()
                q_scale = pyro.param("q_scale_P").detach()
                vals = torch.exp(q_loc + 0.5 * torch.square(q_scale)).cpu().numpy().flatten()
            plt.hist(vals, bins=30)
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


def _auto_point_size(num_points, min_size=0.2, max_size=4.0):
    if num_points <= 0:
        return 1.0
    size = 4000.0 / float(num_points)
    return float(min(max_size, max(min_size, size)))


def _infer_plot_dims(model, store):
    num_patterns = None
    if hasattr(model, "P") and model.P is not None:
        num_patterns = int(model.P.shape[1])
    elif "loc_P" in store:
        num_patterns = int(pyro.param("loc_P").shape[1])
    elif hasattr(model, "A") and model.A is not None:
        num_patterns = int(model.A.shape[0])
    if not num_patterns or num_patterns <= 0:
        return None
    cols = math.ceil(math.sqrt(num_patterns))
    rows = math.ceil(num_patterns / cols)
    return (rows, cols)

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
                        model_type, use_tensorboard_id=None, writer=None, batch_size=None,
                        lr=None, clip_norm=None, tb_max_points=None, post_full_P_steps=None,
                        param_P=None):
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
    - batch_size: Optional minibatch size
    
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
    if batch_size is not None:
        settings['batch_size'] = str(batch_size)
    if lr is not None:
        settings['lr'] = str(lr)
    if clip_norm is not None:
        settings['clip_norm'] = str(clip_norm)
    if tb_max_points is not None:
        settings['tb_max_points'] = str(tb_max_points)
    if post_full_P_steps is not None:
        settings['post_full_P_steps'] = str(post_full_P_steps)
    if param_P is not None:
        settings['param_P'] = str(param_P)
    
    if use_tensorboard_id is not None and writer is not None:
        settings['tensorboard_identifier'] = str(writer.log_dir)
        writer.flush()
    
    return settings


def run_nmf_unsupervised(data, num_patterns, num_steps=20000, device=None, NB_probs=0.5, 
                        use_chisq=False, use_pois=False, use_tensorboard_id=None, spatial=False, 
                        plot_dims=None, scale=None, model_family='gamma', batch_size=None,
                        lr=0.1, clip_norm=None, tb_max_points=5000, post_full_P_steps=0,
                        param_P=False):
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
    - batch_size: Optional minibatch size for sample-wise subsampling
    
    Returns:
    - result_anndata: AnnData object with results
    """
    coords = validate_data(data, spatial, plot_dims, num_patterns)
    keep_on_cpu = (
        batch_size is not None
        and model_family == 'exponential'
        and batch_size < data.shape[0]
    )
    D, U, scale, device = prepare_tensors(data, device, keep_on_cpu=keep_on_cpu)
    print(f"D shape: {D.shape}, U shape: {U.shape}")
    
    model_type = f'{model_family}_unsupervised'
    model, guide, svi = setup_model_and_optimizer(
        D,
        num_patterns,
        scale,
        NB_probs,
        use_chisq,
        use_pois,
        device,
        model_type=model_type,
        batch_size=batch_size,
        lr=lr,
        clip_norm=clip_norm,
        param_P=param_P,
    )
    print(f'Plotting with spatial coordinates: {spatial}')
    
    losses, steps, runtime, writer = run_inference_loop(
        svi, model, D, U, num_steps, use_tensorboard_id, spatial, coords, plot_dims, tb_max_points
    )

    if param_P and model_family == "exponential":
        store = pyro.get_param_store()
        if "q_loc_P" in store and "q_scale_P" in store:
            q_loc = pyro.param("q_loc_P").detach()
            q_scale = pyro.param("q_scale_P").detach()
            mean_P = torch.exp(q_loc + 0.5 * torch.square(q_scale))
            storage_device = getattr(model, "storage_device", mean_P.device)
            if mean_P.device != storage_device:
                mean_P = mean_P.to(storage_device)
            model.P = mean_P
            model.best_P = mean_P

    settings = create_settings_dict(
        num_patterns,
        num_steps,
        device,
        NB_probs,
        use_chisq,
        scale,
        model_type,
        use_tensorboard_id,
        writer,
        batch_size,
        lr=lr,
        clip_norm=clip_norm,
        tb_max_points=tb_max_points,
        post_full_P_steps=post_full_P_steps,
        param_P=param_P,
    )
    
    result_anndata = data.copy()
    result_anndata = save_results_to_anndata(
        result_anndata, model, losses, steps, runtime, scale, settings, num_learned_patterns=num_patterns, supervised=None
    )

    if (
        post_full_P_steps
        and not param_P
        and model_family == "exponential"
        and batch_size is not None
        and batch_size < data.shape[0]
    ):
        pattern_names = None
        if "best_P" in result_anndata.obsm:
            pattern_names = list(result_anndata.obsm["best_P"].columns)
        elif "last_P" in result_anndata.obsm:
            pattern_names = list(result_anndata.obsm["last_P"].columns)
        if pattern_names is None:
            pattern_names = [f"Pattern_{i+1}" for i in range(num_patterns)]

        best_A = model.best_A.detach().cpu().numpy()
        pyro.clear_param_store()
        post_model = _post_infer_full_P(
            data=data,
            fixed_A=best_A,
            num_steps=post_full_P_steps,
            device=device,
            NB_probs=NB_probs,
            use_chisq=use_chisq,
            use_pois=use_pois,
            lr=lr,
            clip_norm=clip_norm,
        )
        if post_model is not None and hasattr(post_model, "best_P"):
            post_best_P = pd.DataFrame(
                post_model.best_P.detach().cpu().numpy(),
                index=result_anndata.obs.index,
                columns=pattern_names,
            )
            post_last_P = pd.DataFrame(
                post_model.P.detach().cpu().numpy(),
                index=result_anndata.obs.index,
                columns=pattern_names,
            )
            result_anndata.obsm["post_best_P"] = post_best_P
            result_anndata.obsm["post_last_P"] = post_last_P
            result_anndata.obsm["best_P"] = post_best_P
            result_anndata.obsm["last_P"] = post_last_P
            result_anndata.uns["post_full_P"] = True
            result_anndata.uns["post_full_P_steps"] = int(post_full_P_steps)

    pyro.clear_param_store()
    return result_anndata


def _post_infer_full_P(data, fixed_A, num_steps, device=None, NB_probs=0.5,
                      use_chisq=False, use_pois=False, lr=0.1, clip_norm=None):
    if num_steps is None or num_steps <= 0:
        return None
    fixed_patterns = torch.tensor(fixed_A.T, dtype=default_dtype)
    D, U, _, device = prepare_tensors(data, device, keep_on_cpu=False)
    fixed_patterns = fixed_patterns.to(device)
    model = Exponential_SSFixedGenes(
        D.shape[0],
        D.shape[1],
        num_patterns=0,
        fixed_patterns=fixed_patterns,
        use_chisq=use_chisq,
        use_pois=use_pois,
        NB_probs=NB_probs,
        device=device,
        batch_size=None,
    )
    guide = AutoNormal(model)
    if clip_norm is not None:
        optimizer = pyro.optim.ClippedAdam({"lr": lr, "eps": 1e-08, "clip_norm": clip_norm})
    else:
        optimizer = pyro.optim.Adam({"lr": lr, "eps": 1e-08})
    loss_fn = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=loss_fn)
    run_inference_loop(
        svi,
        model,
        D,
        U,
        num_steps=num_steps,
        use_tensorboard_id=None,
        spatial=False,
        coords=None,
        plot_dims=None,
        tb_max_points=None,
    )
    return model


def run_nmf_supervised(data, num_patterns, fixed_patterns, num_steps=20000, device=None, 
                      NB_probs=0.5, use_chisq=False, use_pois=False, use_tensorboard_id=None, 
                      spatial=False, plot_dims=None, scale=None, model_family='gamma',
                      supervision_type='fixed_genes', batch_size=None,
                      lr=0.1, clip_norm=None, tb_max_points=5000, param_P=False):
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
    - batch_size: Optional minibatch size for sample-wise subsampling
    
    Returns:
    - result_anndata: AnnData object with results
    """
    coords = validate_data(data, spatial, plot_dims, num_patterns)
    keep_on_cpu = (
        batch_size is not None
        and model_family == 'exponential'
        and batch_size < data.shape[0]
    )
    D, U, scale, device = prepare_tensors(data, device, keep_on_cpu=keep_on_cpu)
    
    fixed_pattern_names = list(fixed_patterns.columns)
    fixed_patterns_tensor = torch.tensor(fixed_patterns.to_numpy(), dtype=torch.float32).to(device)
    
    model_type = f'{model_family}_supervised'
    model, guide, svi = setup_model_and_optimizer(
        D,
        num_patterns,
        scale,
        NB_probs,
        use_chisq,
        use_pois,
        device,
        fixed_patterns_tensor,
        model_type,
        supervision_type,
        batch_size,
        lr=lr,
        clip_norm=clip_norm,
        param_P=param_P,
    )
    
    losses, steps, runtime, writer = run_inference_loop(
        svi, model, D, U, num_steps, use_tensorboard_id, spatial, coords, plot_dims, tb_max_points
    )

    if param_P and model_family == "exponential":
        store = pyro.get_param_store()
        if "q_loc_P" in store and "q_scale_P" in store:
            q_loc = pyro.param("q_loc_P").detach()
            q_scale = pyro.param("q_scale_P").detach()
            mean_P = torch.exp(q_loc + 0.5 * torch.square(q_scale))
            storage_device = getattr(model, "storage_device", mean_P.device)
            if mean_P.device != storage_device:
                mean_P = mean_P.to(storage_device)
            model.P = mean_P
            model.best_P = mean_P
            if hasattr(model, "fixed_P"):
                fixed_P = model.fixed_P
                if fixed_P.device != storage_device:
                    fixed_P = fixed_P.to(storage_device)
                model.P_total = torch.cat((fixed_P, mean_P), dim=1)
    
    settings = create_settings_dict(
        num_patterns,
        num_steps,
        device,
        NB_probs,
        use_chisq,
        scale,
        model_type,
        use_tensorboard_id,
        writer,
        batch_size,
        lr=lr,
        clip_norm=clip_norm,
        tb_max_points=tb_max_points,
        param_P=param_P,
    )
    
    result_anndata = data.copy()
    result_anndata = save_results_to_anndata(
        result_anndata, model, losses, steps, runtime, scale, settings, fixed_pattern_names=fixed_pattern_names, num_learned_patterns=num_patterns, supervised=supervision_type
    )
    
    pyro.clear_param_store()
    return result_anndata
