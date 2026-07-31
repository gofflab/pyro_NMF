#%%
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
import random
from pyroNMF.run_inference import *
import gc

#%% LOAD DATA
data = ad.read_h5ad('/raid/kyla/data/Zhuang-ABCA-1-raw_1.058_wMeta_wAnnotations_KW.h5ad') 
data = data[data.obsm['atlas']['Isocortex']]
coords = data.obs.loc[:,['x','y']] # shape: samples x 2
coords['y'] = -1*coords['y'] # specific for this dataset
data.obsm['spatial'] = coords.to_numpy() # expects coordinates in 'spatial' if using spatial NMF

### Additional parameters:
# fixed_patterns: DataFrame of fixed patterns to use, shape: samples x num_patterns
layers = data.obsm['atlas'].loc[:,['SS','MO']]*1 # pass this in as dataframe to preserve names

def mean_expression(adata, mask):
    X = adata.X[mask.values, :]
    return X.mean(axis=0)

gene_layers = pd.DataFrame(
    {
        'SS': mean_expression(data, layers['SS']),
        'MO': mean_expression(data, layers['MO']),
    },
    index=data.var_names
)

outputDir = "/raid/kyla/projects/pyro_NMF/analyses/test"
os.chdir(outputDir) # set working directory for tensorboard logging

num_steps = 50

############# HELPER FUNCTIONS ##################
def cleanup():
    """Release pyro/GPU state between runs (mirrors the cleanup run_nmf now does internally,
    kept here too since intermediate variables in this script can hold their own references)."""
    pyro.clear_param_store()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_and_save(label, filename, **run_nmf_kwargs):
    """Run one NMF configuration, sanity-check it, save it, and clean up -- regardless of
    whether it raises, so one failing config doesn't leave stale param-store/GPU state for
    the next one."""
    print(f"\n################ {label} ################")
    try:
        nmf_res = run_nmf(
            data,
            20,  # num_patterns
            num_burnin=num_steps,
            num_sample_steps=num_steps,
            spatial=True,
            plot_dims=[5, 4],
            use_pois=False,
            use_chisq=False,
            uncertainty='auto',  # explicit: 10%-of-expression uncertainty (see note below)
            debug=False,
            **run_nmf_kwargs,
        )
        #check_result(nmf_res, label)
        nmf_res.write_h5ad(os.path.join(outputDir, filename))
    finally:
        if 'nmf_res' in dir():
            del nmf_res
        cleanup()

#%%
models = ["gamma","exponential","exponentialSingle"]

supervision = {
    "uns": {},
    "SSg": {
        "supervision_type":"fixed_genes",
        "fixed_patterns": gene_layers,
    },
    "SSp": {
        "supervision_type":"fixed_samples",
        "fixed_patterns": layers,
    },
}

# Replace with your desired uncertainty array
user_uncertainty = np.ones(data.X.shape, dtype=np.float32)*0.5

uncertainties = {
    "none": None,
    "auto": "auto",
    "user": user_uncertainty,
}

losses = {
    "none": dict(use_chisq=False,use_pois=False),
    "chisq": dict(use_chisq=True,use_pois=False),
    "pois": dict(use_chisq=False,use_pois=True),
}

results=[]

expected_loss_points=(num_steps+num_steps)//10

#%%
for model in models:
    for sup_name,sup_kwargs in supervision.items():
        for unc_name,unc in uncertainties.items():
            for loss_name,loss_kwargs in losses.items():
                for debug in [False,True]:

                    label=f"{model}_{sup_name}_{unc_name}_{loss_name}_{'debug' if debug else 'nodebug'}"

                    #try:
                    adata = run_nmf(
                        data,
                        20,
                        num_burnin=num_steps,
                        num_sample_steps=num_steps,
                        spatial=True,
                        plot_dims=[5, 4],
                        use_pois=loss_kwargs['use_pois'],
                        use_chisq=loss_kwargs['use_chisq'],
                        uncertainty=unc,  # explicit: 10%-of-expression uncertainty (see note below)
                        debug=debug,
                        model_family=model,
                        optimizer=pyro.optim.AdamW({"lr":0.1,"eps":1e-8}),
                        use_tensorboard_id=f"_{label}"
                        )

                    #adata=ad.read_h5ad(f"{outputDir}/{label}.h5ad")
                    adata.write_h5ad(os.path.join(outputDir, f'{label}.h5ad'))
                    nloss=len(adata.uns["loss"])
                    ended=nloss<expected_loss_points
                    #adata.uns["ended_early"]=ended
                    #adata.uns["completed"]=not ended
                    #adata.write()

                    results.append(dict(
                        run=label,
                        #status="PASS",
                        ended_early=adata.uns["ended_early"],
                        loss_points=nloss,
                        error=""
                    ))

                    #except Exception as e:
                    #    results.append(dict(
                    #        run=label,
                    #        status="FAIL",
                    #        ended_early=True,
                    #        loss_points=0,
                    #        error=str(e)
                    #    ))
                    #finally:
                    cleanup()

pd.DataFrame(results).to_csv(f"{outputDir}/test_summary.csv",index=False)
print("Finished.")

# %%
