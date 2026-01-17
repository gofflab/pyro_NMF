# Usage

## Quickstart (unsupervised)

```python
import anndata as ad
from pyroNMF.run_inference import run_nmf_unsupervised

adata = ad.read_h5ad("/path/to/data.h5ad")

# Optional: add spatial coordinates for plotting
# adata.obsm["spatial"] = coords  # shape: (n_samples, 2)

result = run_nmf_unsupervised(
    adata,
    num_patterns=20,
    num_steps=20000,
    model_family="gamma",  # or "exponential"
    spatial=False,
)

# Results are stored in result.obsm, result.varm, and result.uns
```

## Supervised / semi-supervised runs

Provide fixed patterns as a pandas DataFrame and specify whether they are
fixed over genes or samples:

```python
import pandas as pd
from pyroNMF.run_inference import run_nmf_supervised

# fixed over samples (n_samples x n_fixed_patterns)
fixed_patterns_samples = pd.DataFrame(
    ...,  # your matrix
    index=adata.obs_names,
    columns=["Layer1", "Layer2"],
)

result = run_nmf_supervised(
    adata,
    num_patterns=10,
    fixed_patterns=fixed_patterns_samples,
    supervision_type="fixed_samples",
    model_family="gamma",
)

# fixed over genes (n_genes x n_fixed_patterns)
fixed_patterns_genes = pd.DataFrame(
    ...,  # your matrix
    index=adata.var_names,
    columns=["MarkerSetA", "MarkerSetB"],
)

result = run_nmf_supervised(
    adata,
    num_patterns=10,
    fixed_patterns=fixed_patterns_genes,
    supervision_type="fixed_genes",
    model_family="exponential",
)
```

## Spatial plotting and TensorBoard logging

If you store spatial coordinates under `adata.obsm['spatial']` (shape
`(n_samples, 2)`), you can request spatial plots during training and log
metrics to TensorBoard:

```python
result = run_nmf_unsupervised(
    adata,
    num_patterns=12,
    num_steps=5000,
    spatial=True,
    plot_dims=[3, 4],
    use_tensorboard_id="_experiment_01",
)
```

## Model families and loss options

- `model_family='gamma'` uses Gamma priors for `A` and `P`.
- `model_family='exponential'` uses Exponential priors for `A` and `P`.
- `use_chisq=True` adds a chi-squared term via `pyro.factor`.
- `use_pois=True` adds a Poisson log-likelihood term via `pyro.factor`.

## Running multiple jobs in one session

The high-level wrappers clear the Pyro parameter store at the end of each
run. If you manually construct models or modify the workflow, you may need
an explicit `pyro.clear_param_store()` between runs.
