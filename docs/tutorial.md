# Tutorial (Worked Example)

This tutorial runs end-to-end with synthetic data so you can verify your
installation and explore the full API without needing external files.

---

## 1) Create synthetic counts and AnnData

```python
import numpy as np
import pandas as pd
import anndata as ad

rng = np.random.default_rng(0)

n_samples  = 200
n_genes    = 100
n_patterns = 5

# Ground-truth non-negative factors
P_true = rng.gamma(shape=1.5, scale=1.0, size=(n_samples, n_patterns)).astype(np.float32)
A_true = rng.gamma(shape=1.5, scale=1.0, size=(n_patterns, n_genes)).astype(np.float32)

# Observed counts with Poisson noise
X = rng.poisson(P_true @ A_true).astype(np.float32)

adata = ad.AnnData(X)
adata.obs_names = [f"cell_{i}" for i in range(n_samples)]
adata.var_names = [f"gene_{i}" for i in range(n_genes)]

print(adata)
# AnnData object with n_obs × n_vars = 200 × 100
```

---

## 2) Run unsupervised NMF — primary Exponential model

The `exponential` model is the recommended primary choice. It places an
independent Exponential prior on every element of `A` and `P`, with
per-element variational rate parameters (`loc_A`, `loc_P`).

```python
from pyroNMF.run_inference import run_nmf

result = run_nmf(
    adata,
    num_patterns=5,
    num_burnin=500,
    num_sample_steps=2000,
    model_family="exponential",   # primary model family
)
```

`run_nmf` prints the device, data shape, sparsity, and ELBO loss every
100 steps, then reports total runtime.

---

## 3) Inspect the AnnData output

### Sample-level factors (`obsm`)

```python
print(list(result.obsm.keys()))
# ['loc_P', 'last_P', 'best_P', 'best_locP',
#  'best_P_scaled', 'last_P_scaled',
#  'mean_P', 'var_P', 'sum_P', 'sum_P2',
#  'markers_P', 'markers_Pscaled', 'markers_Psoftmax']

# Variational rate parameter — one value per sample per pattern
print(result.obsm["loc_P"].shape)   # (200, 5)
print(result.obsm["loc_P"].head())

# Posterior mean across sampling steps — recommended primary estimate
print(result.obsm["mean_P"].head())

# Posterior variance
print(result.obsm["var_P"].head())
```

### Gene-level factors (`varm`)

```python
print(list(result.varm.keys()))
# ['loc_A', 'last_A', 'best_A', 'best_locA',
#  'best_A_scaled', 'last_A_scaled',
#  'mean_A', 'var_A', 'sum_A', 'sum_A2',
#  'markers_A', 'markers_Ascaled', 'markers_Asoftmax']

# Posterior mean gene loadings
print(result.varm["mean_A"].shape)  # (100, 5)
print(result.varm["mean_A"].head())
```

### Training metadata (`uns`)

```python
print(result.uns["runtime (seconds)"])

# ELBO loss curve
import matplotlib.pyplot as plt
result.uns["loss"].plot(title="ELBO loss")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()

# Chi-squared tracking (always recorded)
print(result.uns["best_chisq"])
print(result.uns["step_w_bestChisq"])

# Full run settings
print(result.uns["settings"])
```

---

## 4) Chi-squared tracking and "best" parameters

Enable `use_chisq=True` to add a chi-squared penalty to the ELBO. The model
saves a snapshot of `A`, `P`, `loc_A`, and `loc_P` at the step with the
lowest chi-squared — often a more stable estimate than the final iteration.

```python
result_chisq = run_nmf(
    adata,
    num_patterns=5,
    num_burnin=500,
    num_sample_steps=2000,
    model_family="exponential",
    use_chisq=True,
)

print(f"Best chi-sq: {result_chisq.uns['best_chisq']:.2f} "
      f"at step {result_chisq.uns['step_w_bestChisq']}")

print(result_chisq.obsm["best_P"].head())
print(result_chisq.obsm["best_locP"].head())   # per-element rate at best step
```

---

## 5) Comparing all three model families

```python
# Primary — per-element Exponential
result_exp = run_nmf(
    adata, num_patterns=5, num_burnin=500, num_sample_steps=2000,
    model_family="exponential",
)

# Alternative — Gamma prior with shared scale
result_gamma = run_nmf(
    adata, num_patterns=5, num_burnin=500, num_sample_steps=2000,
    model_family="gamma",
)
print(result_gamma.obsm["loc_P"].head())   # per-element loc_P

# Alternative — single shared Exponential scale
result_single = run_nmf(
    adata, num_patterns=5, num_burnin=500, num_sample_steps=2000,
    model_family="exponentialSingle",
)
# ExponentialSingle has scalar rates in uns instead of per-element loc in obsm
print(result_single.uns["scale_P"])   # scalar
print(result_single.uns["scale_A"])
```

---

## 6) Semi-supervised run (fixed patterns)

Fix two ground-truth sample-level patterns and learn three additional ones.

```python
fixed_patterns = pd.DataFrame(
    P_true[:, :2],
    index=adata.obs_names,
    columns=["Fixed_1", "Fixed_2"],
)

result_ss = run_nmf(
    adata,
    num_patterns=3,
    fixed_patterns=fixed_patterns,
    supervision_type="fixed_samples",
    num_burnin=500,
    num_sample_steps=2000,
    model_family="exponential",
)

# Column names: fixed patterns first, then learned
print(result_ss.obsm["best_P_total"].columns.tolist())
# ['Fixed_1', 'Fixed_2', 'Pattern_1', 'Pattern_2', 'Pattern_3']

print(result_ss.obsm["fixed_P"].head())
print(result_ss.obsm["mean_P"].head())
```

---

## 7) Spatial data and visualisation

```python
from pyroNMF.utils import plot_results

# Add synthetic 2-D coordinates
coords = rng.uniform(size=(n_samples, 2)).astype(np.float32)
adata.obsm["spatial"] = coords

result_spatial = run_nmf(
    adata,
    num_patterns=5,
    num_burnin=200,
    num_sample_steps=500,
    spatial=True,
    plot_dims=[2, 3],           # 2 rows × 3 cols; one panel unused for 5 patterns
    model_family="exponential",
)

# Posterior mean patterns on spatial coordinates
plot_results(
    result_spatial,
    nrows=2, ncols=3,
    which="mean_P",
    scale_alpha=True,
    scale_values=True,
    title="Posterior mean P (spatial)",
)

# Best-chi-squared snapshot
plot_results(
    result_spatial,
    nrows=2, ncols=3,
    which="best_P",
    scale_alpha=False,
    scale_values=True,
    savename="best_P_spatial.png",
)
```

---

## 8) Marker matrices

At each sampling step, the pattern with the highest value wins a vote for
each sample (or gene). Accumulated over `num_sample_steps`, the marker
matrices show which pattern most consistently dominates.

```python
# For each sample, how many steps each pattern won
print(result.obsm["markers_P"].head())

# Convert to fraction
n_samp_steps = int(result.uns["settings"].loc["num_sample_steps", "settings"])
frac_P = result.obsm["markers_P"] / n_samp_steps
print(frac_P.head())

# Gene-level dominant pattern
print(result.varm["markers_A"].head())

# Soft assignment (softmax-normalised accumulation)
print(result.obsm["markers_Psoftmax"].head())
```

---

## 9) Custom optimizer and uncertainty

```python
import pyro.optim

# ClippedAdam for more stable training on noisy data
optimizer = pyro.optim.ClippedAdam({"lr": 0.05, "clip_norm": 5.0})

# Custom per-entry uncertainty (Poisson-inspired)
U = np.sqrt(adata.X + 1).astype(np.float32)

result_custom = run_nmf(
    adata,
    num_patterns=5,
    num_burnin=500,
    num_sample_steps=2000,
    uncertainty=U,
    optimizer=optimizer,
    model_family="exponential",
    use_chisq=True,
)
```

---

## Notes

- `adata.X` must contain **non-negative** values. Raw counts are recommended;
  normalised or log-transformed data may produce poor fits with the Negative
  Binomial likelihood.
- `run_nmf` calls `pyro.clear_param_store()` at the end of every run. If
  you use lower-level functions directly, clear the store manually between
  runs.
- GPU acceleration (CUDA or MPS) is used automatically when available.
  Pass `device='cpu'` to force CPU execution.
