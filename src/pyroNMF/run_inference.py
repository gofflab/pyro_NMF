#from pyroNMF.models.deprecated.exp_pois_models_noNB import ExponentialPoisson_base
import torch

from pyroNMF.models.gamma_NB_models import (
    Gamma_NegBinomial_base,
    Gamma_NegBinomial_SSFixedGenes,
    Gamma_NegBinomial_SSFixedSamples
)
from pyroNMF.models.expSingle_NB_models import (
    ExponentialSingle_base,
    ExponentialSingle_SSFixedGenes,
    ExponentialSingle_SSFixedSamples
)
from pyroNMF.models.exp_NB_models import (
    Exponential_NegBinomial_base,
    Exponential_NegBinomial_SSFixedGenes,
    Exponential_NegBinomial_SSFixedSamples
)

#from pyroNMF.models.extras.gamma_NB_fixedAlphas import GammaFixAlpha_NegBinomial_base
#from pyroNMF.models.gamma_NB_alphaPerGene import GammaAlphaPerGene_NegBinomial_base
#from pyroNMF.models.gamma_NB_alphaPerPattern import GammaAlphaPerPattern_NegBinomial_base
#from pyroNMF.models.gamma_NB_hierarchicalAlpha import GammaHierarchicalAlpha_NegBinomial_base
#from pyroNMF.models.gamma_NB_hierarchicalAlpha import GammaHierarchicalAlpha_NegBinomial_base
#from pyroNMF.models.gamma_NB_alphaPerPattern_fixedScale import GammaAlphaPerPattern_FixScale_NegBinomial_base

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

#def validate_data(data, spatial=False, plot_dims=None, num_patterns=None):
#    """
#    Validate input data and spatial coordinates.
#    
#    Parameters:
#    - data: AnnData object containing the data
#    - spatial: Whether spatial analysis is requested
#    - plot_dims: Plotting dimensions [rows, cols]
#    - num_patterns: Number of patterns for validation
#    
#    Returns:
#    - coords: Spatial coordinates if spatial=True, else None
    
#    Raises:
#    - ValueError: If validation fails
#    """
#    print("Running validate_data")
#    D = torch.tensor(data.X) if not hasattr(data.X, 'toarray') else torch.tensor(data.X.toarray())
#    coords = None
    
#    perc_zeros = (D == 0).sum().sum() / (D.shape[0] * D.shape[1])
#    print(f"Data contains {D.shape[0]} cells and {D.shape[1]} genes")
#    print(f"Data is {perc_zeros*100:.2f}% sparse")
    
#    if spatial:
#        if data.obsm.get('spatial') is None:
#            raise ValueError("Spatial coordinates are not present in the data. Please provide spatial coordinates in obsm['spatial']")
        
#        coords = data.obsm['spatial']
#        if coords.shape[1] != 2:
#            raise ValueError("Spatial coordinates should have two columns named 'x' and 'y'")
        
#        if plot_dims is not None and num_patterns is not None:
#            if plot_dims[0] * plot_dims[1] < num_patterns:
#                print("Warning: plot_dims less than num_patterns. Not all patterns will be plotted")
    
#    return coords


def prepare_tensors(data, layer=None, uncertainty=None, scale=None, device=None, spatial=False):

    """
    Prepare and move tensors to the specified device.
 
    Parameters:
    - data: AnnData object containing the data
    - layer: Layer key to use instead of data.X (optional)
    - uncertainty: Pre-computed uncertainty tensor (optional)
    - scale: Scale factor (optional; defaults to 2*std of data)
    - device: Target device ('cpu', 'cuda', 'mps', or None for auto-detection)
    - spatial: Whether to extract spatial coordinates from data.obsm['spatial']
 
    Returns:
    - D: Data tensor
    - U: Uncertainty tensor
    - scale: Scale factor tensor
    - device: Selected device
    - coords: Spatial coordinates array if spatial=True, else None
 
    """
    
    print(f"################## Preparing tensors ##################")

    # Use specified device unless None, then autodetect
    if device is None:
        device = detect_device() 
    print(f"Selecting device {device}")

    # Use specified layer unless None, then use .X    
    if layer is None:
        if hasattr(data.X, 'toarray'):
            print('Setting sparse D from data.X')
            D = torch.tensor(data.X.toarray(), dtype=default_dtype).to(device)
        else:
            print('Setting D from data.X')
            D = torch.tensor(data.X, dtype=default_dtype).to(device)
    else:
        # ADD TRY CATCH HERE TO CHECK IF LAYER EXISTS
        D = torch.tensor(data.layers[layer], dtype=default_dtype).to(device)
        print(f'Setting D from data.layers[{layer}]')

    # Look at sparsity
    perc_zeros = (D == 0).sum().sum() / (D.shape[0] * D.shape[1])
    print(f"Data contains {D.shape[0]} cells and {D.shape[1]} genes")
    print(f"Data is {perc_zeros*100:.2f}% sparse")

    # Use 10% of D (clipped at 0.3) as default uncertainty
    if uncertainty is None:
        U = (D * 0.1).clip(min=0.3).to(device)
        print("Using default uncertainty, 10% expression (clipped at 0.3)")
    else:
        U = torch.tensor(uncertainty, dtype=default_dtype).to(device)
        print("Using user-specified uncertainty")

    # Use 2*std of data as default scale unless otherwise specified
    if scale is None:
        scale = torch.tensor((D.cpu().numpy().std()) * 2, dtype=default_dtype, device=device)
        print(f"Using default scale as 2*std(data) = {scale}")
    else:
        scale=torch.tensor(scale, dtype=default_dtype, device=device)
        print(f"Using user-specified scale = {scale}")

    if spatial:
        if data.obsm.get('spatial') is None:
            raise ValueError("Spatial coordinates are not present in the data. Please provide spatial coordinates in obsm['spatial']")
        
        coords = data.obsm['spatial']
        if coords.shape[1] != 2:
            raise ValueError("Spatial coordinates should have two columns named 'x' and 'y'")
    else:
        coords = None
         
    return D, U, scale, device, coords


def setup_model_and_optimizer(D, num_patterns, scale=1, NB_probs=0.5, use_chisq=False, use_pois=False, device=None,
                             fixed_patterns=None, model_type='gamma_unsupervised',
                             supervision_type=None, optimizer=pyro.optim.Adam({"lr": 0.1, "eps": 1e-08})):
    """
    Setup the NMF model and optimizer.
    
    Parameters:
    - D: Data tensor
    - num_patterns: Number of patterns
    - scale: Scale for gamma
    - NB_probs: Negative binomial probability
    - use_chisq: Whether to use chi-squared loss
    - device: Device to run on
    - fixed_patterns: Fixed patterns for supervised learning
    - model_type: 'gamma_unsupervised', 'gamma_supervised', 'exponential_unsupervised', 'exponential_supervised', , 'exponentialSingle_unsupervised', 'exponentialSingle_supervised'
    - supervision_type: 'fixed_genes' or 'fixed_samples' (for supervised models)
    - supervision_type: 'fixed_genes' or 'fixed_samples' (supervised models only)
    - optimizer: Pyro optimizer instance
    
    Returns:
    - model: Initialized model
    - guide: AutoNormal guide
    - svi: SVI optimizer
    """
    print(f'model_type is {model_type}')

    n_cells, n_genes = D.shape[0], D.shape[1]
    shared_kwargs = dict(
        use_chisq=use_chisq, use_pois=use_pois, NB_probs=NB_probs, device=device
    )
    gamma_kwargs = dict(**shared_kwargs, scale=scale)


    # Instantiate the model
    if model_type == 'gamma_unsupervised':
        model = Gamma_NegBinomial_base(n_cells, n_genes, num_patterns, **gamma_kwargs)
 
    elif model_type == 'gamma_supervised':
        cls_map = {
            'fixed_genes':    Gamma_NegBinomial_SSFixedGenes,
            'fixed_samples':  Gamma_NegBinomial_SSFixedSamples,
        }
        _assert_supervision_type(supervision_type, cls_map)
        model = cls_map[supervision_type](
            n_cells, n_genes, num_patterns, fixed_patterns=fixed_patterns, **gamma_kwargs
        )

    elif model_type == 'exponential_unsupervised':
        model = Exponential_NegBinomial_base(n_cells, n_genes, num_patterns, **shared_kwargs)
 
    elif model_type == 'exponential_supervised':
        cls_map = {
            'fixed_genes':    Exponential_NegBinomial_SSFixedGenes,
            'fixed_samples':  Exponential_NegBinomial_SSFixedSamples,
        }
        _assert_supervision_type(supervision_type, cls_map)
        model = cls_map[supervision_type](
            n_cells, n_genes, num_patterns, fixed_patterns=fixed_patterns, **shared_kwargs
        )
 
    #elif model_type == 'gammaFixAlpha_unsupervised':
    #    model = GammaFixAlpha_NegBinomial_base(
    #        D.shape[0], D.shape[1], num_patterns, 
    #        use_chisq=use_chisq, use_pois=use_pois,scale=scale, 
    #        NB_probs=NB_probs, device=device
    #    )

    #elif model_type == 'gammaAlphaPerGene_unsupervised':
    #    model = GammaAlphaPerGene_NegBinomial_base(
    #        D.shape[0], D.shape[1], num_patterns, 
    #        use_chisq=use_chisq, use_pois=use_pois,scale=scale, 
    #        NB_probs=NB_probs, device=device
    #    )
    
    #elif model_type == 'gammaAlphaPerPattern_unsupervised':
    #    model = GammaAlphaPerPattern_NegBinomial_base(
    #        D.shape[0], D.shape[1], num_patterns, 
    #        use_chisq=use_chisq, use_pois=use_pois,scale=scale, 
    #        NB_probs=NB_probs, device=device
    #    )

    #elif model_type == 'gammaHierarchicalAlpha_unsupervised':
    #    model = GammaHierarchicalAlpha_NegBinomial_base(
    #        D.shape[0], D.shape[1], num_patterns, 
    #        use_chisq=use_chisq, use_pois=use_pois,scale=scale, 
    #        NB_probs=NB_probs, device=device
    #    )
    #elif model_type == 'gammaAlphaPerPattern_fixedScale_unsupervised':
    #    model = GammaAlphaPerPattern_FixScale_NegBinomial_base(
    #        D.shape[0], D.shape[1], num_patterns, 
    #        use_chisq=use_chisq, use_pois=use_pois,scale=scale, 
    #        NB_probs=NB_probs, device=device
    #    )

    elif model_type == 'exponentialSingle_unsupervised':
        model = ExponentialSingle_base(n_cells, n_genes, num_patterns, **shared_kwargs)
 
    elif model_type == 'exponentialSingle_supervised':
        cls_map = {
            'fixed_genes':    ExponentialSingle_SSFixedGenes,
            'fixed_samples':  ExponentialSingle_SSFixedSamples,
        }
        _assert_supervision_type(supervision_type, cls_map)
        model = cls_map[supervision_type](
            n_cells, n_genes, num_patterns, fixed_patterns=fixed_patterns, **shared_kwargs
        )
 
    else:
        raise ValueError(
            "model_type must be one of: 'gamma_unsupervised', 'gamma_supervised', "
            "'exponential_unsupervised', 'exponential_supervised', "
            "'exponentialSingle_unsupervised', 'exponentialSingle_supervised'"
        )
    
    #pyro.render_model(model, model_args=(D,D), 
    #                render_params=True,
    #                render_distributions=True,
    #                #render_deterministic=True,
    #                filename=f"{model_type}.pdf")

    # Setup guide and optimizer
    guide = AutoNormal(model)
    #optimizer = pyro.optim.Adam({"lr": 0.1, "eps": 1e-08})
    loss_fn = pyro.infer.Trace_ELBO()
    
    svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=loss_fn)
    
    return model, guide, svi

def _assert_supervision_type(supervision_type, cls_map):
    """Raise a clear error when supervision_type is not one of the expected values."""
    if supervision_type not in cls_map:
        raise ValueError("supervision_type must be 'fixed_genes' or 'fixed_samples'")
 

def run_inference_loop(svi, model, D, U, num_burnin, num_sample_steps, use_tensorboard_id=None, 
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
        print(f'Logging to Tensorboard {os.getcwd()}{writer.get_logdir()}')
    
    for step in range(1, num_burnin + num_sample_steps + 1):
        try:
            if step <= num_burnin+1:
                loss = svi.step(D, U, samp=False)
            else:
                loss = svi.step(D, U, samp=True)
        except ValueError as e:
            print(f"ValueError during iteration {step}: {e}")
            break
        
        if step % 10 == 0:
            losses.append(loss)
            steps.append(step)
            #print(svi.optim.get_state())
            log_adam_state_all(writer, svi, step, log_every=10)
        
        if step % 100 == 0:
            print(f"Iteration {step}, ELBO loss: {loss}")
        
        if writer is not None:
            _log_tensorboard_metrics(writer, model, D, step, loss, spatial, coords, plot_dims)
    
    end_time = datetime.now()
    runtime = round((end_time - start_time).total_seconds())
    print(f'Runtime: {runtime} seconds')
    
    return losses, steps, runtime, writer

def log_adam_state_all(writer, svi, step, tag_prefix="OptimAdam", log_every=1):
    if writer is None:
        return
    if (step % log_every) != 0:
        return
 
    state = svi.optim.get_state()
 
    for name, d in state.items():
        if "param_groups" not in d or "state" not in d:
            continue
        if len(d["param_groups"]) == 0 or len(d["state"]) == 0:
            continue
 
        g0 = d["param_groups"][0]
        if "lr" not in g0 or "betas" not in g0 or "eps" not in g0:
            continue
 
        lr = float(g0["lr"])
        beta1, beta2 = g0["betas"]
        eps = float(g0["eps"])
 
        safe_name = str(name).replace(":", "_").replace(" ", "_").replace("/", "_")
        base = f"{tag_prefix}/{safe_name}"
 
        for param_id, st in d["state"].items():
            if "step" not in st:
                continue
            t = int(st["step"].item())
 
            bc1 = 1.0 - (beta1 ** t)
            bc2 = 1.0 - (beta2 ** t)
 
            sub = base if len(d["state"]) == 1 else f"{base}/param{param_id}"
 
            m = st.get("exp_avg", None)
            v = st.get("exp_avg_sq", None)
 
            if m is not None:
                m2_mean = (m * m).mean()
                writer.add_scalar(f"{sub}/m_rms",  float(torch.sqrt(m2_mean).item()), step)
                writer.add_scalar(f"{sub}/m_norm", float(torch.linalg.vector_norm(m).item()), step)
 
            if v is not None:
                v2_mean = (v * v).mean()
                writer.add_scalar(f"{sub}/v_rms",  float(torch.sqrt(v2_mean).item()), step)
                writer.add_scalar(f"{sub}/v_norm", float(torch.linalg.vector_norm(v).item()), step)
 
                eff = lr * (bc2 ** 0.5) / bc1 / (v.sqrt() + eps)
                writer.add_scalar(f"{sub}/efflr_mean", float(eff.mean().item()), step)
                writer.add_scalar(f"{sub}/efflr_min",  float(eff.min().item()), step)
                writer.add_scalar(f"{sub}/efflr_max",  float(eff.max().item()), step)
 
                def _prep(x):
                    x = x.detach().flatten()
                    if x.numel() > 200000:
                        idx = torch.randperm(x.numel(), device=x.device)[:200000]
                        x = x[idx]
                    return x
 
                eff_log10 = torch.log10(eff.clamp_min(1e-30))
                eff_log10_1 = _prep(eff_log10)  # noqa: F841  (kept for parity)
                plt.hist(eff_log10.detach().cpu().numpy().flatten(), bins=30)
                writer.add_figure("efflr_log10_hist", plt.gcf(), step)
 
            writer.add_scalar(f"{sub}/adam_step", float(t), step)
 
 
def _log_tensorboard_metrics(writer, model, D, step, loss, spatial=False, coords=None, plot_dims=None):
    """
    Log metrics to tensorboard.
 
    Parameters:
    - writer: Tensorboard writer
    - model: The model
    - D: Observed data tensor (used for residual plots)
    - step: Current step
    - loss: Current loss
    - spatial: Whether to include spatial plots
    - coords: Spatial coordinates
    - plot_dims: Plotting dimensions [rows, cols]
    """
    writer.add_scalar("Loss/train", loss, step)
    if hasattr(model, "best_chisq"):
        writer.add_scalar("Best chi-squared",      float(getattr(model, "best_chisq", np.inf)), step)
        writer.add_scalar("Saved chi-squared iter", int(getattr(model, "best_chisq_iter", 0)),   step)
    if hasattr(model, "chi2"):
        writer.add_scalar("Chi-squared", float(getattr(model, "chi2")), step)
    if hasattr(model, "pois"):
        writer.add_scalar("Poisson loss",      float(getattr(model, "pois")),          step)
        writer.add_scalar("Pois / ELBO ratio", abs(float(getattr(model, "pois"))) / abs(loss), step)
    writer.flush()
 
    if step % 50 == 0 or step == 1:
        _log_param_figures(writer, model, step, spatial, coords, plot_dims)
        _log_reconstructed_figures(writer, model, D, step)
        _log_memory(writer, step)
 
    if step % 100 == 0 and hasattr(model, "D_reconstructed"):
        plt.figure()
        plt.hist(model.D_reconstructed.detach().cpu().numpy().flatten(), bins=30)
        writer.add_figure("D_reconstructed_hist", plt.gcf(), step)
 
 
def _log_param_figures(writer, model, step, spatial, coords, plot_dims):
    """Log parameter histograms and spatial grids to tensorboard."""
    store = pyro.get_param_store()
 
    # Spatial grids
    if spatial and coords is not None and plot_dims is not None:
        for param_name in ('loc_P', 'alpha_P'):
            if param_name in store:
                try:
                    vals = pyro.param(param_name).detach().cpu().numpy()
                    plot_grid(vals, coords, plot_dims[0], plot_dims[1], savename=None)
                    writer.add_figure(param_name, plt.gcf(), step)
                except Exception:
                    pass
 
        if hasattr(model, "P"):
            try:
                plot_grid(model.P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename=None)
                writer.add_figure("current sampled P", plt.gcf(), step)
            except Exception:
                pass
 
    # Histograms
    for param_name in ('loc_P', 'loc_A', 'scale_A', 'scale_P'):
        if param_name in store:
            plt.figure()
            plt.hist(pyro.param(param_name).detach().cpu().numpy().flatten(), bins=30)
            writer.add_figure(f"{param_name}_hist", plt.gcf(), step)
 
    # Heatmaps
    for param_name in ('alpha_P', 'alpha_A'):
        if param_name in store:
            plt.figure()
            sns.heatmap(pyro.param(param_name).detach().cpu(), annot=True, fmt=".2f", cmap="viridis")
            writer.add_figure(f"{param_name}_hist", plt.gcf(), step)
 
 
def _log_reconstructed_figures(writer, model, D, step):
    """Log D_reconstructed diagnostics to tensorboard."""
    if not hasattr(model, "D_reconstructed"):
        return
 
    ### FROM LOYAL START HERE ###
    ### QUESTION IS HE PASSING D HERE ###
    D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
 
    plt.figure()
    plt.hist(D_reconstructed.flatten(), bins=30)
    writer.add_figure("D_reconstructed_hist", plt.gcf(), step)
 
    try:
        writer.add_scalar("D_reconstructed_mean", float(D_reconstructed.mean()), step)
        writer.add_scalar("D_reconstructed_std",  float(D_reconstructed.std()),  step)
        writer.add_scalar("D_reconstructed_min",  float(D_reconstructed.min()),  step)
        writer.add_scalar("D_reconstructed_max",  float(D_reconstructed.max()),  step)
    except Exception:
        pass
 
    try:
        D_tensor  = model.D_reconstructed.detach()
        D_obs     = D.detach() if (D is not None and torch.is_tensor(D)) else None
        D_flat    = D_tensor.reshape(D_tensor.shape[0], -1) if D_tensor.ndim > 2 else D_tensor
        D_obs_flat = (D_obs.reshape(D_obs.shape[0], -1) if D_obs.ndim > 2 else D_obs) if D_obs is not None else None
 
        means = D_flat.mean(dim=0)
        vars_ = D_flat.var(dim=0, unbiased=False)
 
        eps     = torch.finfo(means.dtype).eps
        cv2     = vars_ / means.clamp_min(eps) ** 2
        mean_np = means.detach().cpu().numpy()
        cv2_np  = cv2.detach().cpu().numpy()
        mask    = np.isfinite(mean_np) & np.isfinite(cv2_np) & (mean_np > 0) & (cv2_np > 0)
        mean_np, cv2_np = mean_np[mask], cv2_np[mask]
 
        if mean_np.size > 0:
            plt.figure()
            plt.scatter(mean_np, cv2_np, s=8, alpha=0.3, edgecolors="none")
            plt.xlabel("Mean")
            plt.ylabel("CV^2")
            plt.title("D_reconstructed mean vs CV^2")
            plt.xscale("log")
            plt.yscale("log")
            writer.add_figure("D_reconstructed_mean_cov", plt.gcf(), step)
 
        if D_obs_flat is not None:
            try:
                exp_np = means.detach().cpu().numpy()
                obs_np = D_obs_flat.mean(dim=0).detach().cpu().numpy()
                mask   = np.isfinite(exp_np) & np.isfinite(obs_np)
                exp_np, obs_np = exp_np[mask], obs_np[mask]
 
                if exp_np.size > 0:
                    residuals = obs_np - exp_np
                    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
 
                    min_v, max_v = float(min(exp_np.min(), obs_np.min())), float(max(exp_np.max(), obs_np.max()))
                    axes[0].scatter(exp_np, obs_np, s=8, alpha=0.3, edgecolors="none")
                    axes[0].plot([min_v, max_v], [min_v, max_v], color="gray", linewidth=1.0)
                    axes[0].set_xlabel("Expected (mean of D_reconstructed)")
                    axes[0].set_ylabel("Observed (mean of D)")
                    axes[0].set_title("Expected vs observed")
 
                    axes[1].scatter(exp_np, residuals, s=8, alpha=0.3, edgecolors="none")
                    axes[1].axhline(0.0, color="gray", linewidth=1.0)
                    axes[1].set_xlabel("Expected (mean of D_reconstructed)")
                    axes[1].set_ylabel("Observed - expected")
                    axes[1].set_title("Per-gene residuals")
 
                    fig.suptitle("Per-gene residuals (expected vs observed)", y=1.02)
                    fig.tight_layout()
                    writer.add_figure("D_reconstructed_mean_cov/expected_vs_observed", fig, step)
            except Exception:
                pass
    except Exception:
        pass
    ### FROM LOYAL END HERE ###
 
 
def _log_memory(writer, step):
    """Log GPU/MPS memory usage to tensorboard."""
    try:
        mib = 1024.0 * 1024.0
        if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
            writer.add_scalar("Mem/cuda_allocated_MiB",     float(torch.cuda.memory_allocated())     / mib, step)
            writer.add_scalar("Mem/cuda_reserved_MiB",      float(torch.cuda.memory_reserved())      / mib, step)
            writer.add_scalar("Mem/cuda_max_allocated_MiB", float(torch.cuda.max_memory_allocated())  / mib, step)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            writer.add_scalar("Mem/mps_allocated_MiB",        float(torch.mps.current_allocated_memory()) / mib, step)
            writer.add_scalar("Mem/mps_driver_allocated_MiB", float(torch.mps.driver_allocated_memory())  / mib, step)
    except Exception:
        pass
 
def _pattern_cols(shape_ncols, learned_pattern_names, full_pattern_names):
    """
    Return the correct column-name list for a result matrix.
 
    If the matrix has more columns than the number of learned patterns it must
    come from a semi-supervised run and we use the full (fixed + learned) names.
    """
    if shape_ncols != len(learned_pattern_names):
        return full_pattern_names
    return learned_pattern_names
 
 
def _pattern_cols(shape_ncols, learned_pattern_names, full_pattern_names):
    """
    Return the correct column-name list for a result matrix.
 
    If the matrix has more columns than the number of learned patterns it must
    come from a semi-supervised run and we use the full (fixed + learned) names.
    full_pattern_names must not be None when shape_ncols != len(learned_pattern_names).
    """
    if shape_ncols != len(learned_pattern_names):
        if full_pattern_names is None:
            raise ValueError(
                f"Matrix has {shape_ncols} columns but only {len(learned_pattern_names)} "
                "learned patterns and no fixed_pattern_names were provided."
            )
        return full_pattern_names
    return learned_pattern_names
 
 
def _detect_and_save_parameters(result_anndata, model, settings, fixed_pattern_names=None, num_learned_patterns=None):
    """
    Auto-detect and save all model parameters from the model and param store.
 
    Parameters:
    - result_anndata: AnnData object to save results into
    - model: The trained model
    - settings: Settings dict (used to read num_sample_steps)
    - fixed_pattern_names: Names of fixed patterns (supervised models only)
    - num_learned_patterns: Number of learned patterns
 
    FIX: full_pattern_names is always defined (None in unsupervised mode) so the
         obs_df/var_df closures never hit an unbound name.
    """
    store = pyro.get_param_store()
    learned_pattern_names = ["Pattern_" + str(x + 1) for x in range(num_learned_patterns)]
    # Always defined; only populated for supervised / semi-supervised runs.
    full_pattern_names = (
        [str(x) for x in fixed_pattern_names] + learned_pattern_names
        if fixed_pattern_names is not None
        else None
    )
 
    def obs_df(arr, ncols):
        df = pd.DataFrame(arr)
        df.columns = _pattern_cols(ncols, learned_pattern_names, full_pattern_names)
        df.index = result_anndata.obs.index
        return df
 
    def var_df(arr, ncols):
        df = pd.DataFrame(arr)
        df.columns = _pattern_cols(ncols, learned_pattern_names, full_pattern_names)
        df.index = result_anndata.var.index
        return df
 
    # ------------------------------------------------------------------
    # Variational parameters: loc (Gamma models)
    # ------------------------------------------------------------------
    if "loc_P" in store:
        arr = pyro.param("loc_P").detach().cpu().numpy()
        result_anndata.obsm["loc_P"] = obs_df(arr, arr.shape[1])
        print("Saving loc_P in anndata.obsm['loc_P']")
 
    if "loc_A" in store:
        arr = pyro.param("loc_A").detach().cpu().numpy().T
        result_anndata.varm["loc_A"] = var_df(arr, arr.shape[1])
        print("Saving loc_A in anndata.varm['loc_A']")
 
    # ------------------------------------------------------------------
    # Variational parameters: alpha (Gamma models)
    # ------------------------------------------------------------------
    if "alpha_P" in store:
        result_anndata.uns["alpha_P"] = pyro.param("alpha_P").detach().cpu().numpy()
        print(f"Saving alpha_P in anndata.uns['alpha_P'] — shape {result_anndata.uns['alpha_P'].shape}")
 
    if "alpha_A" in store:
        result_anndata.uns["alpha_A"] = pyro.param("alpha_A").detach().cpu().numpy().T
        print("Saving alpha_A in anndata.uns['alpha_A']")
 
    # ------------------------------------------------------------------
    # Variational parameters: scale scalars (Exponential models)
    # ------------------------------------------------------------------
    if "scale_P" in store:
        val = pyro.param("scale_P").detach().cpu().item()
        result_anndata.uns["scale_P"] = val
        print(f"Saving scale_P = {val} in anndata.uns['scale_P']")
 
    if "scale_A" in store:
        val = pyro.param("scale_A").detach().cpu().item()
        result_anndata.uns["scale_A"] = val
        print(f"Saving scale_A = {val} in anndata.uns['scale_A']")
 
    # ------------------------------------------------------------------
    # Last sampled A / P
    # ------------------------------------------------------------------
    if hasattr(model, "P"):
        arr = model.P.detach().cpu().numpy()
        result_anndata.obsm["last_P"] = obs_df(arr, arr.shape[1])
        print("Saving final sampled P in anndata.obsm['last_P']")
 
    if hasattr(model, "A"):
        arr = model.A.detach().cpu().numpy().T
        result_anndata.varm["last_A"] = var_df(arr, arr.shape[1])
        print("Saving final sampled A in anndata.varm['last_A']")
 
    # ------------------------------------------------------------------
    # Total A / P (supervised models: fixed + learned concatenated)
    # ------------------------------------------------------------------
    if hasattr(model, "P_total"):
        arr = model.P_total.detach().cpu().numpy()
        result_anndata.obsm["P_total"] = obs_df(arr, arr.shape[1])
        print("Saving P_total in anndata.obsm['P_total']")
 
    if hasattr(model, "A_total"):
        arr = model.A_total.detach().cpu().numpy().T
        result_anndata.varm["A_total"] = var_df(arr, arr.shape[1])
        print("Saving A_total in anndata.varm['A_total']")
 
    # ------------------------------------------------------------------
    # Fixed A / P (supervised models)
    # ------------------------------------------------------------------
    if hasattr(model, "fixed_P"):
        result_anndata.obsm["fixed_P"] = pd.DataFrame(
            model.fixed_P.detach().cpu().numpy(),
            columns=[str(p) for p in fixed_pattern_names],
            index=result_anndata.obs.index,
        )
        print("Saving fixed P in anndata.obsm['fixed_P']")
 
    if hasattr(model, "fixed_A"):
        result_anndata.varm["fixed_A"] = pd.DataFrame(
            model.fixed_A.detach().cpu().numpy(),
            columns=[str(p) for p in fixed_pattern_names],
            index=result_anndata.var.index,
        )
        print("Saving fixed A in anndata.varm['fixed_A']")
 
    # ------------------------------------------------------------------
    # Best A / P (based on chi2 criterion)
    # ------------------------------------------------------------------
    if hasattr(model, "best_locP"):
        arr = model.best_locP.detach().cpu().numpy()
        result_anndata.obsm["best_locP"] = obs_df(arr, arr.shape[1])
        print("Saving best_locP in anndata.obsm['best_locP']")
 
    if hasattr(model, "best_locA"):
        arr = model.best_locA.detach().cpu().numpy().T
        result_anndata.varm["best_locA"] = var_df(arr, arr.shape[1])
        print("Saving best_locA in anndata.varm['best_locA']")
 
    if hasattr(model, "best_scaleP"):
        val = model.best_scaleP.detach().cpu().item()
        result_anndata.uns["best_scale_P"] = val
        print(f"Saving best_scale_P = {val} in anndata.uns['best_scale_P']")
 
    if hasattr(model, "best_scaleA"):
        val = model.best_scaleA.detach().cpu().item()
        result_anndata.uns["best_scale_A"] = val
        print(f"Saving best_scale_A = {val} in anndata.uns['best_scale_A']")
 
    if hasattr(model, "best_P"):
        arr = model.best_P.detach().cpu().numpy()
        result_anndata.obsm["best_P"] = obs_df(arr, arr.shape[1])
        print("Saving final sampled P in anndata.obsm['best_P']")
 
    if hasattr(model, "best_A"):
        arr = model.best_A.detach().cpu().numpy().T
        result_anndata.varm["best_A"] = var_df(arr, arr.shape[1])
        print("Saving final sampled A in anndata.varm['best_A']")
 
    # ------------------------------------------------------------------
    # Best total A / P (supervised: merge fixed + best)
    # ------------------------------------------------------------------
    if "fixed_P" in result_anndata.obsm and "best_P" in result_anndata.obsm:
        result_anndata.obsm["best_P_total"] = result_anndata.obsm["fixed_P"].merge(
            result_anndata.obsm["best_P"], left_index=True, right_index=True
        )
        print("Saving best_P_total in anndata.obsm['best_P_total']")
 
    if "fixed_A" in result_anndata.varm and "best_A" in result_anndata.varm:
        result_anndata.varm["best_A_total"] = result_anndata.varm["fixed_A"].merge(
            result_anndata.varm["best_A"], left_index=True, right_index=True
        )
        print("Saving best_A_total in anndata.varm['best_A_total']")
 
    # ------------------------------------------------------------------
    # Scaled A / P  (normalise patterns to sum=1, absorb scale into A)
    # ------------------------------------------------------------------
    # --- best ---
    P_to_scale = result_anndata.obsm.get("best_P_total", result_anndata.obsm.get("best_P"))
    if P_to_scale is not None:
        P_scaled = P_to_scale.div(P_to_scale.sum(axis=0), axis=1)
        result_anndata.obsm["best_P_scaled"] = P_scaled
        print("Saving best_P_scaled in anndata.obsm['best_P_scaled']")
 
        A_to_adjust = result_anndata.varm.get("best_A_total", result_anndata.varm.get("best_A"))
        if A_to_adjust is not None:
            result_anndata.varm["best_A_scaled"] = A_to_adjust.multiply(P_to_scale.sum(axis=0), axis=1)
            print("Saving best_A_scaled in anndata.varm['best_A_scaled']")
 
    # --- last ---
    P_to_scale = result_anndata.obsm.get("P_total", result_anndata.obsm.get("last_P"))
    if P_to_scale is not None:
        P_scaled = P_to_scale.div(P_to_scale.sum(axis=0), axis=1)
        result_anndata.obsm["last_P_scaled"] = P_scaled
        print("Saving last_P_scaled in anndata.obsm['last_P_scaled']")
 
        A_to_adjust = result_anndata.varm.get("A_total", result_anndata.varm.get("last_A"))
        if A_to_adjust is not None:
            result_anndata.varm["last_A_scaled"] = A_to_adjust.multiply(P_to_scale.sum(axis=0), axis=1)
            print("Saving last_A_scaled in anndata.varm['last_A_scaled']")
 
    # ------------------------------------------------------------------
    # Running sums and variances (for posterior mean / uncertainty)
    # ------------------------------------------------------------------
    num_samples = int(settings['num_sample_steps'])
 
    if hasattr(model, "sum_P"):
        arr    = model.sum_P.detach().cpu().numpy()
        sum_P  = pd.DataFrame(arr)
        mean_P = sum_P / num_samples
        cols   = _pattern_cols(arr.shape[1], learned_pattern_names, full_pattern_names)
        sum_P.columns = mean_P.columns = cols
        sum_P.index   = mean_P.index   = result_anndata.obs.index
        result_anndata.obsm["sum_P"]  = sum_P
        result_anndata.obsm["mean_P"] = mean_P
        print("Saving sum_P and mean_P in anndata.obsm")
 
    if hasattr(model, "sum_A"):
        arr    = model.sum_A.detach().cpu().numpy().T
        sum_A  = pd.DataFrame(arr)
        mean_A = sum_A / num_samples
        cols   = _pattern_cols(arr.shape[1], learned_pattern_names, full_pattern_names)
        sum_A.columns = mean_A.columns = cols
        sum_A.index   = mean_A.index   = result_anndata.var.index
        result_anndata.varm["sum_A"]  = sum_A
        result_anndata.varm["mean_A"] = mean_A
        print("Saving sum_A and mean_A in anndata.varm")
 
    if hasattr(model, "sum_P2"):
        arr   = model.sum_P2.detach().cpu().numpy()
        sum_P2 = pd.DataFrame(arr)
        var_P  = pd.DataFrame((arr - (sum_P.to_numpy() ** 2) / num_samples) / (num_samples - 1))
        cols   = _pattern_cols(arr.shape[1], learned_pattern_names, full_pattern_names)
        sum_P2.columns = var_P.columns = cols
        sum_P2.index   = var_P.index   = result_anndata.obs.index
        result_anndata.obsm["sum_P2"] = sum_P2
        result_anndata.obsm["var_P"]  = var_P
        print("Saving sum_P2 and var_P in anndata.obsm")
 
    if hasattr(model, "sum_A2"):
        arr    = model.sum_A2.detach().cpu().numpy().T
        sum_A2 = pd.DataFrame(arr)
        var_A  = pd.DataFrame((arr - (sum_A.to_numpy() ** 2) / num_samples) / (num_samples - 1))
        cols   = _pattern_cols(arr.shape[1], learned_pattern_names, full_pattern_names)
        sum_A2.columns = var_A.columns = cols
        sum_A2.index   = var_A.index   = result_anndata.var.index
        result_anndata.varm["sum_A2"] = sum_A2
        result_anndata.varm["var_A"]  = var_A
        print("Saving sum_A2 and var_A in anndata.varm")
 
    # ------------------------------------------------------------------
    # Marker matrices
    # ------------------------------------------------------------------
    _marker_pairs = [
        ("markers_P",        "obsm", False),
        ("markers_A",        "varm", True),
        ("markers_Pscaled",  "obsm", False),
        ("markers_Ascaled",  "varm", True),
        ("markers_Psoftmax", "obsm", False),
        ("markers_Asoftmax", "varm", True),
    ]
    for attr, slot, transpose in _marker_pairs:
        if not hasattr(model, attr):
            continue
        arr = getattr(model, attr).detach().cpu().numpy()
        if transpose:
            arr = arr.T
        df = pd.DataFrame(arr)
        df.columns = _pattern_cols(arr.shape[1], learned_pattern_names, full_pattern_names)
        df.index = (result_anndata.obs.index if slot == "obsm" else result_anndata.var.index)
        getattr(result_anndata, slot)[attr] = df
        print(f"Saving {attr} in anndata.{slot}['{attr}']")
 
 
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
    # Save parameters
    _detect_and_save_parameters(result_anndata, model, settings, fixed_pattern_names, num_learned_patterns)
    
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


def create_settings_dict(num_patterns, num_total_steps, num_sample_steps, device, NB_probs, use_chisq, scale, 
                        model_type, use_tensorboard_id=None, writer=None):
    """
    Create settings dictionary for saving.
    
    Parameters:
    - num_patterns: Number of patterns
    - num_total_steps: Number of total training steps
    - num_sample_steps: Number of sampling steps
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
        'num_total_steps': str(num_total_steps),
        'num_sample_steps': str(num_sample_steps),
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


def run_nmf(
    data, num_patterns, layer=None, uncertainty=None, fixed_patterns=None,
    num_burnin=1000, num_sample_steps=20000, device=None,
    NB_probs=0.5, use_chisq=False, use_pois=False, use_tensorboard_id=None,
    spatial=False, plot_dims=None, scale=None, model_family=None,
    supervision_type=None, model=None,
    optimizer=pyro.optim.Adam({"lr": 0.1, "eps": 1e-08})
):
    """
    Run supervised or unsupervised NMF analysis.
 
    Parameters:
    - data: AnnData object with raw counts in X
    - num_patterns: Number of patterns to learn
    - layer: Layer key to use instead of data.X (optional)
    - uncertainty: Pre-computed uncertainty array (optional)
    - fixed_patterns: DataFrame with fixed patterns for supervised mode (optional)
    - num_burnin: Number of burn-in steps
    - num_sample_steps: Number of sampling steps
    - device: Device ('cpu', 'cuda', 'mps', or None for auto-detection)
    - NB_probs: Negative binomial probability
    - use_chisq: Whether to use chi-squared loss
    - use_pois: Whether to use Poisson loss
    - use_tensorboard_id: Tensorboard logging identifier (optional)
    - spatial: Whether to include spatial analysis
    - plot_dims: Plotting dimensions [rows, cols]
    - scale: Scale factor (computed automatically if None)
    - model_family: 'gamma', 'exponential', or 'exponentialSingle'
    - supervision_type: 'fixed_genes' or 'fixed_samples' (supervised mode only)
    - optimizer: Pyro optimizer instance
 
    Returns:
    - result_anndata: AnnData object with all results and metadata
    """
    D, U, scale, device, coords = prepare_tensors(data, layer, uncertainty, scale, device, spatial)
 
    # regardless of which branch (supervised / unsupervised) is taken below.
    fixed_pattern_names   = None
    fixed_patterns_tensor = None
 
    if fixed_patterns is None:
        # ---- Unsupervised ----
        model_type = f'{model_family}_unsupervised'
        model, guide, svi = setup_model_and_optimizer(
            D, num_patterns, scale, NB_probs, use_chisq, use_pois,
            device, model_type=model_type, optimizer=optimizer
        )
    else:
        # ---- Supervised ----
        model_type            = f'{model_family}_supervised'
        fixed_pattern_names   = list(fixed_patterns.columns)
        fixed_patterns_tensor = torch.tensor(
            fixed_patterns.to_numpy(), dtype=torch.float32
        ).to(device)
 
        model, guide, svi = setup_model_and_optimizer(
            D, num_patterns, scale, NB_probs, use_chisq, use_pois,
            device, fixed_patterns_tensor, model_type, supervision_type,
            optimizer=optimizer
        )
 
    losses, steps, runtime, writer = run_inference_loop(
        svi, model, D, U, num_burnin, num_sample_steps,
        use_tensorboard_id, spatial, coords, plot_dims
    )
 
    settings = create_settings_dict(
        num_patterns, num_sample_steps + num_burnin, num_sample_steps,
        device, NB_probs, use_chisq, scale, model_type,
        use_tensorboard_id, writer
    )
 
    result_anndata = data.copy()
    result_anndata = save_results_to_anndata(
        result_anndata, model, losses, steps, runtime, scale, settings,
        fixed_pattern_names=fixed_pattern_names,
        num_learned_patterns=num_patterns,
        supervised=supervision_type,
    )
 
    pyro.clear_param_store()
    return result_anndata
 
