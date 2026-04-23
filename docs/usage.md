# Usage

All inference — unsupervised and semi-supervised — is driven by the single
function `run_nmf()`. It accepts an AnnData object, runs the SVI loop, and
returns a copy of the input with all results stored in `obsm`, `varm`, and
`uns`.

## Quickstart (unsupervised)

The `exponential` model family is the recommended starting point for most
datasets. It places an independent Exponential prior on every element of
`A` and `P`, with per-element variational rate parameters.

```python
import anndata as ad
from pyroNMF.run_inference import run_nmf

adata = ad.read_h5ad("/path/to/data.h5ad")

result = run_nmf(
    adata,
    num_patterns=20,
    num_burnin=1000,
    num_sample_steps=20000,
    model_family="exponential",   # recommended primary choice
)

# Results stored in result.obsm, result.varm, and result.uns
print(result.obsm.keys())
print(result.varm.keys())
```

## Semi-supervised (fixed patterns)

Pass a `pandas.DataFrame` of fixed patterns and specify whether they are
anchored over genes (`fixed_genes`) or samples (`fixed_samples`).

```python
import pandas as pd
from pyroNMF.run_inference import run_nmf

# Fixed over samples — shape: (n_samples, n_fixed_patterns)
fixed_patterns_samples = pd.DataFrame(
    ...,  # your matrix
    index=adata.obs_names,
    columns=["Layer1", "Layer2"],
)

result = run_nmf(
    adata,
    num_patterns=10,                       # additional patterns to learn
    fixed_patterns=fixed_patterns_samples,
    supervision_type="fixed_samples",
    model_family="exponential",
)

# Fixed over genes — shape: (n_genes, n_fixed_patterns)
fixed_patterns_genes = pd.DataFrame(
    ...,
    index=adata.var_names,
    columns=["MarkerSetA", "MarkerSetB"],
)

result = run_nmf(
    adata,
    num_patterns=10,
    fixed_patterns=fixed_patterns_genes,
    supervision_type="fixed_genes",
    model_family="exponential",
)
```

## Choosing a model family

Three model families are available via the `model_family` argument:

| `model_family` | Prior | Variational params | Notes |
|---|---|---|---|
| `'exponential'` | Exponential | per-element `loc_A`, `loc_P` | **Primary / recommended.** Independent rate per entry. |
| `'gamma'` | Gamma | per-element `loc_A`, `loc_P` + shared `scale` | Alternative with heavier tails and a tunable concentration. |
| `'exponentialSingle'` | Exponential | scalar `scale_A`, `scale_P` | Alternative with a single shared rate per matrix. Faster but less flexible. |

The per-element Exponential model (`exponential`) is the primary choice. Use
`gamma` when you need a tunable concentration parameter or domain knowledge
favours Gamma-distributed factors. Use `exponentialSingle` when dataset size
or compute budget makes the per-element parameterisation impractical.

```python
# Primary — per-element Exponential
result = run_nmf(adata, num_patterns=15, model_family="exponential")

# Alternative — Gamma with shared scale
result = run_nmf(adata, num_patterns=15, model_family="gamma")

# Alternative — single shared Exponential scale per matrix
result = run_nmf(adata, num_patterns=15, model_family="exponentialSingle")
```

## Burn-in and sampling phases

Training is split into two sequential phases:

- **Burn-in** (`num_burnin` steps): the SVI optimizer updates variational
  parameters but does **not** accumulate samples. This lets the model reach
  a stable region of parameter space before statistics are collected.
- **Sampling** (`num_sample_steps` steps): the optimizer continues updating,
  and at each step the current samples of `A` and `P` are accumulated into
  running sums for computing posterior means, variances, and marker matrices.

```python
result = run_nmf(
    adata,
    num_patterns=20,
    num_burnin=2000,        # warm-up without accumulation
    num_sample_steps=10000, # posterior statistics accumulated here
    model_family="exponential",
)

# Posterior mean across sampling steps (recommended primary estimate)
print(result.obsm["mean_P"].head())
print(result.varm["mean_A"].head())

# Posterior variance
print(result.obsm["var_P"].head())
```

## Using a custom data layer

By default `run_nmf` reads counts from `adata.X`. To use a named layer:

```python
result = run_nmf(
    adata,
    num_patterns=20,
    layer="counts",    # uses adata.layers["counts"]
    model_family="exponential",
)
```

## Providing a custom uncertainty matrix

The uncertainty matrix `U` controls the per-entry weight in the chi-squared
term. By default it is computed as `max(0.1 × D, 0.3)`. You can override
this with any non-negative array of the same shape as `adata.X`:

```python
import numpy as np

U = np.ones_like(adata.X) * 0.5   # uniform uncertainty

result = run_nmf(
    adata,
    num_patterns=20,
    uncertainty=U,
    model_family="exponential",
)
```

## Chi-squared tracking and "best" parameters

When `use_chisq=True`, a chi-squared penalty is added to the ELBO and the
model saves a snapshot of `A`, `P`, and (for `exponential` and `gamma`
models) `loc_A`, `loc_P` at the step with the lowest chi-squared. These
"best" parameters are often more stable than the final-iteration values.

```python
result = run_nmf(
    adata,
    num_patterns=20,
    use_chisq=True,
    model_family="exponential",
)

print(result.uns["best_chisq"])
print(result.uns["step_w_bestChisq"])
print(result.obsm["best_P"].head())
print(result.obsm["best_locP"].head())   # exponential / gamma only
```

## Poisson auxiliary loss

Setting `use_pois=True` appends a scaled Poisson log-likelihood term
(`10 × poisL`) to the ELBO. This nudges the reconstruction toward the
observed counts more aggressively than the Negative Binomial likelihood
alone.

```python
result = run_nmf(
    adata,
    num_patterns=20,
    use_pois=True,
    model_family="exponential",
)
```

## Spatial analysis and plotting

If spatial coordinates are stored under `adata.obsm['spatial']` (shape
`(n_samples, 2)`), set `spatial=True` and provide `plot_dims` to enable
spatial pattern logging during training:

```python
result = run_nmf(
    adata,
    num_patterns=12,
    num_sample_steps=5000,
    spatial=True,
    plot_dims=[3, 4],           # 3 rows × 4 cols = 12 panels
    model_family="exponential",
)
```

After the run, visualise any pattern matrix stored in `obsm` with
`plot_results`:

```python
from pyroNMF.utils import plot_results

# Plot the posterior mean patterns on spatial coordinates
plot_results(
    result,
    nrows=3, ncols=4,
    which="mean_P",        # any key in result.obsm
    scale_alpha=True,      # alpha scales with local intensity
    scale_values=True,     # colour clipped to 5th–95th percentile
    title="Posterior mean P",
    savename="mean_P.png",
)
```

## Custom optimizer

The default optimizer is Adam with `lr=0.1, eps=1e-8`. You can pass any
Pyro optimizer instance:

```python
import pyro.optim

optimizer = pyro.optim.ClippedAdam({"lr": 0.05, "clip_norm": 5.0})

result = run_nmf(
    adata,
    num_patterns=20,
    optimizer=optimizer,
    model_family="exponential",
)
```

## TensorBoard logging

Pass a string identifier to `use_tensorboard_id` to write logs to a
TensorBoard run directory:

```python
result = run_nmf(
    adata,
    num_patterns=20,
    use_tensorboard_id="_experiment_01",
    spatial=True,
    plot_dims=[3, 4],
    model_family="exponential",
)
```

Launch TensorBoard from the project root:

```bash
bash run_tensorboard.sh
# or: tensorboard --logdir runs/
```

TensorBoard logs include:

- **Loss/train** — ELBO loss at every step
- **Chi-squared** / **Best chi-squared** — when `use_chisq=True`
- **Poisson loss** / **Pois / ELBO ratio** — when `use_pois=True`
- **Parameter histograms** — `loc_P`, `loc_A`, `scale_P`, `scale_A` every 50 steps
- **Spatial grid plots** — `loc_P` / `alpha_P` overlaid on coordinates every 50 steps
- **D_reconstructed diagnostics** — mean, std, mean vs CV², expected vs observed residuals
- **Adam optimizer state** — effective learning rate, moment norms every 10 steps
- **GPU / MPS memory** — allocated and reserved memory in MiB

## Running multiple jobs in one session

`run_nmf` clears the Pyro parameter store at the end of every call. If you
build custom loops or call lower-level functions directly, insert an explicit
`pyro.clear_param_store()` between runs to avoid state leaking across
experiments.
