# Tutorial (Worked Example)

This tutorial walks through a full workflow with a small synthetic dataset
so you can run end-to-end without external files.

## 1) Create synthetic counts and AnnData

```python
import numpy as np
import pandas as pd
import anndata as ad

rng = np.random.default_rng(0)

n_samples = 200
n_genes = 100
n_patterns = 5

# Create synthetic non-negative factors
P_true = rng.gamma(shape=1.5, scale=1.0, size=(n_samples, n_patterns))
A_true = rng.gamma(shape=1.5, scale=1.0, size=(n_patterns, n_genes))

# Generate counts with Poisson noise
X = rng.poisson(P_true @ A_true).astype(np.float32)

adata = ad.AnnData(X)
adata.obs_names = [f"cell_{i}" for i in range(n_samples)]
adata.var_names = [f"gene_{i}" for i in range(n_genes)]
```

## 2) Run unsupervised NMF

```python
from pyroNMF.run_inference import run_nmf_unsupervised

result = run_nmf_unsupervised(
    adata,
    num_patterns=5,
    num_steps=2000,
    model_family="gamma",
    use_chisq=False,
    use_pois=False,
)
```

## 3) Inspect outputs in AnnData

```python
# Sample-level factors
print(result.obsm.keys())
print(result.obsm["loc_P"].head())

# Gene-level factors
print(result.varm.keys())
print(result.varm["loc_A"].head())

# Training metadata
print(result.uns["runtime (seconds)"])
print(result.uns["loss"].head())
```

## 4) Semi-supervised run (fixed patterns)

Here we fix two patterns over samples and learn three additional patterns.

```python
import pandas as pd
from pyroNMF.run_inference import run_nmf_supervised

fixed_patterns = pd.DataFrame(
    P_true[:, :2],
    index=adata.obs_names,
    columns=["Fixed_1", "Fixed_2"],
)

result_ss = run_nmf_supervised(
    adata,
    num_patterns=3,
    fixed_patterns=fixed_patterns,
    supervision_type="fixed_samples",
    model_family="gamma",
)

print(result_ss.obsm.keys())
print(result_ss.obsm["best_P_total"].head())
```

## 5) Optional spatial plotting

If you have spatial coordinates, add them to `adata.obsm['spatial']` and
set `spatial=True` in the high-level wrappers:

```python
coords = rng.uniform(size=(n_samples, 2))
adata.obsm["spatial"] = coords

result_spatial = run_nmf_unsupervised(
    adata,
    num_patterns=6,
    num_steps=1000,
    spatial=True,
    plot_dims=[2, 3],
)
```

## Notes

- For real datasets, ensure `adata.X` contains raw counts.
- The wrappers automatically clear the Pyro parameter store at the end
  of each run; if you build custom loops, call `pyro.clear_param_store()`
  as needed.
