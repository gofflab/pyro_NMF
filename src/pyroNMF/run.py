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

# take in adata with X as raw counts


#def define_model(data, num_patterns, device=None, NB_probs=0.5, use_chisq=False, spatial=False):
#    """
#    Setup the NMF model with the given parameters.
#    
#    Parameters:
#    - data: Anndata object containing the data.
#    - num_patterns: Number of patterns to extract.
#    - device: Device to run the model on (e.g., 'cpu', 'cuda', 'mps').
#    - NB_probs: Probability for Negative Binomial distribution.
#    - use_chisq: Whether to use Chi-squared loss.
#    - spatial: Whether to include spatial coordinates in the model.
#    
#    Returns:
#    - model: The initialized NMF model.
#    """
#    D = torch.tensor(data.X)
#    U = (D * 0.1).clip(min=0.3)  # in range [0,1]; if None: use default of 1 - sparsity
#
#    model = Gamma_NegBinomial_base(D, num_patterns, U=U, device=device, NB_probs=NB_probs, use_chisq=use_chisq, spatial=spatial)
#    
#    return model


def run_nmf_unsupervised(data, num_patterns, num_steps=20000, device=None, NB_probs=0.5, use_chisq=False, use_tensorboard_id=None, spatial=False, plot_dims=None, scale=None):
    D = torch.tensor(data.X)
    U = (D * 0.1).clip(min=0.3)  # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D
    scale = (D.numpy().std())*2


    percZeros = (D == 0).sum().sum() / (D.shape[0]*D.shape[1])
    print(f"Data contrains {D.shape[0]} cells and {D.shape[1]} genes")
    print(f"Data is {percZeros*100:.2f}% sparse")

    #print(f"Outputting to {outputDir}/{savename}.h5ad")
    if spatial:
        if data.obsm.get('spatial') is None:
            spatial = False
            raise ValueError("Spatial coordinates are not present in the data. Please provide spatial coordinates in obsm['spatial']")
        else:
            coords = data.obsm['spatial']
            if coords.shape[1] != 2:
                raise Valuerror("Spatial coordinates should have two columns named 'x' and 'y'")
            #plot_dims = [5, 6]  # rows x columns should be > num patterns; this is for plotting
            if plot_dims is not None:
                if plot_dims[0] * plot_dims[1] < num_patterns:
                    print("Warning: plot_dims less than num_patterns. Not all patterns will be plotted")

    if device == None:
        device = detect_device()
        #print(f"Setting device as {device}")
    print(f"Selecting device {device}")
    D = D.to(device)
    U = U.to(device)

    # Instantiate the model
    model = Gamma_NegBinomial_base(D.shape[0], D.shape[1], num_patterns, use_chisq=use_chisq, scale=scale, NB_probs = NB_probs, device=device)    

    # Instantiate the guide
    guide = AutoNormal(model)      

    # Instantiate inference algorithm
    optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # Use the Adam optimizer
    loss_fn = pyro.infer.Trace_ELBO()

    # Define the inference algorithm
    svi = pyro.infer.SVI(model=model,
                        guide=guide,
                        optim=optimizer,
                        loss=loss_fn)

    # Start timer
    startTime = datetime.now()

    # Start inference
    steps = []
    losses = []

    if use_tensorboard_id is not None:
        writer = SummaryWriter(comment = use_tensorboard_id)

    for step in range(1,num_steps+1): 
        try:
            loss = svi.step(D,U)
        except ValueError as e:
            print(f"ValueError during iteration {step}: {e}")
            break 
        
        if step % 10 == 0: # Store loss every 10 steps
            losses.append(loss)
            steps.append(step)
            
        if step % 100 == 0: # Print loss every 100 steps
            print(f"Iteration {step}, ELBO loss: {loss}")

        if use_tensorboard_id is not None: # Save outputs to tensorboard
            writer.add_scalar("Loss/train", loss, step)
            writer.flush()

            writer.add_scalar('Best chi-squared',  model.best_chisq, step)
            writer.flush()

            writer.add_scalar('Saved chi-squared iter',  model.best_chisq_iter, step)
            writer.flush()

            writer.add_scalar("Chi-squared", model.chi2, step)
            writer.flush()

            if step % 50 == 0:
                if spatial:
                    # plot loc P
                    #model.plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                    plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                    writer.add_figure("loc_P", plt.gcf(), step)
                
                    # plot this sampled P
                    #model.plot_grid(model.P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                    plot_grid(model.P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                    writer.add_figure("current sampled P", plt.gcf(), step)

                plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
                writer.add_figure("loc_P_hist", plt.gcf(), step)

                plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
                writer.add_figure("loc_A_hist", plt.gcf(), step)
            
            if step % 100 == 0:
                D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
                plt.hist(D_reconstructed.flatten(), bins=30)
                writer.add_figure("D_reconstructed_his", plt.gcf(), step)
    
                    
    endTime = datetime.now()
    print('Runtime: '+ str(round((endTime - startTime).total_seconds())) + ' seconds')

    ### Save results in anndata object
    result_anndata = data.copy()

    ### Save loc_P
    loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
    loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
    loc_P.index = result_anndata.obs.index
    result_anndata.obsm['loc_P'] = loc_P
    print("Saving loc_P in anndata.obsm['loc_P']")

    ### Save loc_A
    loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
    loc_A.columns = ['Pattern_' + str(x+1) for x in loc_A.columns]
    loc_A.index = result_anndata.var.index
    result_anndata.varm['loc_A'] = loc_A
    print("Saving loc_A in anndata.varm['loc_A']")

    ### Save last sampled P
    last_P = pd.DataFrame(model.P.detach().cpu().numpy())
    last_P.columns = ['Pattern_' + str(x+1) for x in last_P.columns]
    last_P.index = result_anndata.obs.index
    result_anndata.obsm['last_P'] = last_P
    print("Saving final sampled P in anndata.obsm['last_P']")

    ### Save last sampled A
    last_A = pd.DataFrame(model.A.detach().cpu().numpy()).T
    last_A.columns = ['Pattern_' + str(x+1) for x in last_A.columns]
    last_A.index = result_anndata.var.index
    result_anndata.varm['last_A'] = last_A
    print("Saving final sampled A in anndata.varm['last_A']")

    ### Save best P (via chi-squared)
    best_P = pd.DataFrame(model.best_P.detach().cpu().numpy())
    best_P.columns = ['Pattern_' + str(x+1) for x in best_P.columns]
    best_P.index = result_anndata.obs.index
    result_anndata.obsm['best_P'] = best_P
    print("Saving best P via chi2 in anndata.obsm['best_P']")

    ### Save best A (via chi-squared)
    best_A = pd.DataFrame(model.best_A.detach().cpu().numpy()).T
    best_A.columns = ['Pattern_' + str(x+1) for x in best_A.columns]
    best_A.index = result_anndata.var.index
    result_anndata.varm['best_A'] = best_A
    print("Saving best A via chi2 in anndata.varm['best_A']")

    ### Save best locP (via chi-squared)
    best_locP = pd.DataFrame(model.best_locP.detach().cpu().numpy())
    best_locP.columns = ['Pattern_' + str(x+1) for x in best_locP.columns]
    best_locP.index = result_anndata.obs.index
    result_anndata.obsm['best_locP'] = best_locP
    print("Saving best loc P via chi2 in anndata.obsm['best_locP']")

    ### Save best locA (via chi-squared)
    best_locA = pd.DataFrame(model.best_locA.detach().cpu().numpy()).T
    best_locA.columns = ['Pattern_' + str(x+1) for x in best_locA.columns]
    best_locA.index = result_anndata.var.index
    result_anndata.varm['best_locA'] = best_A
    print("Saving best loc A via chi2 in anndata.varm['best_locA']")

    #if spatial:
    #    model.plot_grid(model.best_P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_bestP.pdf")

    result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
    result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])
    result_anndata.uns['step_w_bestChisq'] = model.best_chisq_iter
    result_anndata.uns['scale'] = scale


    settings = {
        'num_patterns': str(num_patterns),
        'num_steps': str(num_steps),
        'device': str(device),
        'NB_probs': str(NB_probs),
        'use_chisq': str(use_chisq),
        'scale': str(scale)}
    
    if use_tensorboard_id is not None:
        settings['tensorboard_identifier'] = str(writer.log_dir)
        writer.flush()


    result_anndata.uns['settings'] = pd.DataFrame(list(settings.values()),index=list(settings.keys()), columns=['settings'])
    #result_anndata.write_h5ad(outputDir + '/' + savename + '.h5ad')
    pyro.clear_param_store()
    #    return model, result_anndata
    return result_anndata



def run_nmf_supervisedP(data, num_patterns, fixed_patterns, num_steps=20000, device=None, NB_probs=0.5, use_chisq=False, use_tensorboard_id=None, spatial=False, plot_dims=None, scale=None):
    D = torch.tensor(data.X)
    U = (D * 0.1).clip(min=0.3)  # in range [0,1]; if None: use default of 1 - sparsity; this is the probs argument for NegativeBinomial for D
    scale = (D.numpy().std())*2
    fixed_pattern_names = list(fixed_patterns.columns)
    fixed_patterns = torch.tensor(fixed_patterns.to_numpy())


    percZeros = (D == 0).sum().sum() / (D.shape[0]*D.shape[1])
    print(f"Data contrains {D.shape[0]} cells and {D.shape[1]} genes")
    print(f"Data is {percZeros*100:.2f}% sparse")

    #print(f"Outputting to {outputDir}/{savename}.h5ad")
    if spatial:
        if data.obsm.get('spatial') is None:
            spatial = False
            raise ValueError("Spatial coordinates are not present in the data. Please provide spatial coordinates in obsm['spatial']")
        else:
            coords = data.obsm['spatial']
            if coords.shape[1] != 2:
                raise Valuerror("Spatial coordinates should have two columns named 'x' and 'y'")
            #plot_dims = [5, 6]  # rows x columns should be > num patterns; this is for plotting
            if plot_dims is not None:
                if plot_dims[0] * plot_dims[1] < num_patterns:
                    print("Warning: plot_dims less than num_patterns. Not all patterns will be plotted")

    if device == None:
        device = detect_device()
        #print(f"Setting device as {device}")
    print(f"Selecting device {device}")
    D = D.to(device)
    U = U.to(device)
    fixed_patterns = fixed_patterns.to(device)


    # Instantiate the model
    model = Gamma_NegBinomial_SSFixed(D.shape[0], D.shape[1], num_patterns, fixed_patterns=fixed_patterns, use_chisq=use_chisq, scale=scale, NB_probs = NB_probs, device=device)          

    # Instantiate the guide
    guide = AutoNormal(model)      

    # Instantiate inference algorithm
    optimizer = pyro.optim.Adam({"lr": 0.1, "eps":1e-08}) # Use the Adam optimizer
    loss_fn = pyro.infer.Trace_ELBO()

    # Define the inference algorithm
    svi = pyro.infer.SVI(model=model,
                        guide=guide,
                        optim=optimizer,
                        loss=loss_fn)

    # Start timer
    startTime = datetime.now()

    # Start inference
    steps = []
    losses = []

    if use_tensorboard_id is not None:
        writer = SummaryWriter(comment = use_tensorboard_id)

    for step in range(1,num_steps+1): 
        try:
            loss = svi.step(D,U)
        except ValueError as e:
            print(f"ValueError during iteration {step}: {e}")
            break 
        
        if step % 10 == 0: # Store loss every 10 steps
            losses.append(loss)
            steps.append(step)
            
        if step % 100 == 0: # Print loss every 100 steps
            print(f"Iteration {step}, ELBO loss: {loss}")

        if use_tensorboard_id is not None: # Save outputs to tensorboard
            writer.add_scalar("Loss/train", loss, step)
            writer.flush()

            writer.add_scalar('Best chi-squared',  model.best_chisq, step)
            writer.flush()

            writer.add_scalar('Saved chi-squared iter',  model.best_chisq_iter, step)
            writer.flush()

            writer.add_scalar("Chi-squared", model.chi2, step)
            writer.flush()

            if step % 50 == 0:
                if spatial:
                    # plot loc P
                    plot_grid(pyro.param("loc_P").detach().to('cpu').numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                    writer.add_figure("loc_P", plt.gcf(), step)
                
                    # plot this sampled P
                    plot_grid(model.P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename = None)
                    writer.add_figure("current sampled P", plt.gcf(), step)

                plt.hist(pyro.param("loc_P").detach().to('cpu').numpy().flatten(), bins=30)
                writer.add_figure("loc_P_hist", plt.gcf(), step)

                plt.hist(pyro.param("loc_A").detach().to('cpu').numpy().flatten(), bins=30)
                writer.add_figure("loc_A_hist", plt.gcf(), step)
            
            if step % 100 == 0:
                D_reconstructed = model.D_reconstructed.detach().cpu().numpy()
                plt.hist(D_reconstructed.flatten(), bins=30)
                writer.add_figure("D_reconstructed_his", plt.gcf(), step)
    
                    
    endTime = datetime.now()
    print('Runtime: '+ str(round((endTime - startTime).total_seconds())) + ' seconds')

    ### Save results in anndata object
    result_anndata = data.copy()

    ### Save loc_P
    loc_P = pd.DataFrame(pyro.param("loc_P").detach().to('cpu').numpy())
    loc_P.columns = ['Pattern_' + str(x+1) for x in loc_P.columns]
    loc_P.index = result_anndata.obs.index
    result_anndata.obsm['loc_P'] = loc_P
    print("Saving loc_P in anndata.obsm['loc_P']")

    ### Save loc_A
    loc_A = pd.DataFrame(pyro.param("loc_A").detach().to('cpu').numpy()).T
    loc_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    loc_A.index = result_anndata.var.index
    result_anndata.varm['loc_A'] = loc_A
    print("Saving loc_A in anndata.varm['loc_A']")

    ### Save last sampled P
    last_P = pd.DataFrame(model.P.detach().cpu().numpy())
    last_P.columns = ['Pattern_' + str(x+1) for x in last_P.columns]
    last_P.index = result_anndata.obs.index
    result_anndata.obsm['last_P'] = last_P
    print("Saving final sampled P in anndata.obsm['last_P']")

    ### Save last sampled A
    last_A = pd.DataFrame(model.A.detach().cpu().numpy()).T
    last_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    last_A.index = result_anndata.var.index
    result_anndata.varm['last_A'] = last_A
    print("Saving final sampled A in anndata.varm['last_A']")

    ### Save best P (via chi-squared)
    best_P = pd.DataFrame(model.best_P.detach().cpu().numpy())
    best_P.columns = ['Pattern_' + str(x+1) for x in best_P.columns]
    best_P.index = result_anndata.obs.index
    result_anndata.obsm['best_P'] = best_P
    print("Saving best P via chi2 in anndata.obsm['best_P']")

    ### Save best P (via chi-squared)
    fixed_P = pd.DataFrame(model.fixed_P.detach().cpu(), columns = fixed_pattern_names, index=result_anndata.obs.index)
    #best_P.columns = ['Pattern_' + str(x+1) for x in best_P.columns]
    #best_P.index = result_anndata.obs.index
    result_anndata.obsm['fixed_P'] = fixed_P
    print("Saving fixed P via chi2 in anndata.obsm['fixed_P']")

    ### Save best P (via chi-squared)
    best_P_total = fixed_P.merge(best_P, left_index=True, right_index=True)
    result_anndata.obsm['best_P_total'] = best_P_total
    print("Saving best P via chi2 in anndata.obsm['best_P_total']")

    ### Save best A (via chi-squared)
    best_A = pd.DataFrame(model.best_A.detach().cpu().numpy()).T
    best_A.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    best_A.index = result_anndata.var.index
    result_anndata.varm['best_A'] = best_A
    print("Saving best A via chi2 in anndata.varm['best_A']")

    ### Save best locP (via chi-squared)
    best_locP = pd.DataFrame(model.best_locP.detach().cpu().numpy())
    best_locP.columns = ['Pattern_' + str(x+1) for x in best_locP.columns]
    best_locP.index = result_anndata.obs.index
    result_anndata.obsm['best_locP'] = best_locP
    print("Saving best loc P via chi2 in anndata.obsm['best_locP']")

    ### Save best locA (via chi-squared)
    best_locA = pd.DataFrame(model.best_locA.detach().cpu().numpy()).T
    best_locA.columns = fixed_pattern_names + ['Pattern_' + str(x+1) for x in np.arange(num_patterns)]
    best_locA.index = result_anndata.var.index
    result_anndata.varm['best_locA'] = best_A
    print("Saving best loc A via chi2 in anndata.varm['best_locA']")



    #if spatial:
    #    model.plot_grid(model.best_P.detach().cpu().numpy(), coords, plot_dims[0], plot_dims[1], savename = savename + "_bestP.pdf")

    result_anndata.uns['runtime (seconds)'] = round((endTime - startTime).total_seconds())
    result_anndata.uns['loss'] = pd.DataFrame(losses, index=steps, columns=['loss'])
    result_anndata.uns['step_w_bestChisq'] = model.best_chisq_iter
    result_anndata.uns['scale'] = scale


    settings = {
        'num_patterns': str(num_patterns),
        'num_steps': str(num_steps),
        'device': str(device),
        'NB_probs': str(NB_probs),
        'use_chisq': str(use_chisq),
        'scale': str(scale)}
    
    if use_tensorboard_id is not None:
        settings['tensorboard_identifier'] = str(writer.log_dir)
        writer.flush()


    result_anndata.uns['settings'] = pd.DataFrame(list(settings.values()),index=list(settings.keys()), columns=['settings'])
    #result_anndata.write_h5ad(outputDir + '/' + savename + '.h5ad')
    pyro.clear_param_store()
    #    return model, result_anndata
    return result_anndata




        #def prep_anndata():
            # 