# AnnData Integration

pyroNMF is designed to accept an AnnData object as input and return a copy
of that AnnData object with results stored in `obsm`, `varm`, and `uns`.

## Input expectations

- `adata.X`: raw counts matrix of shape `(n_samples, n_genes)`.
- `adata.obsm['spatial']` (optional): coordinates of shape `(n_samples, 2)`
  for spatial visualization.

## Output structure

The high-level wrappers return a new AnnData with the following keys:

### `obsm` (sample-level factors)

- `loc_P`: learned location parameters for `P` (Gamma models).
- `last_P`: final sampled `P` from the last iteration.
- `best_P`: sampled `P` at the best chi-squared iteration.
- `fixed_P`: fixed patterns over samples (semi-supervised).
- `P_total`: concatenated fixed + learned `P` (semi-supervised).
- `best_P_total`: concatenated fixed + best learned `P`.

### `varm` (gene-level factors)

- `loc_A`: learned location parameters for `A` (Gamma models).
- `last_A`: final sampled `A` from the last iteration.
- `best_A`: sampled `A` at the best chi-squared iteration.
- `fixed_A`: fixed patterns over genes (semi-supervised).
- `A_total`: concatenated fixed + learned `A` (semi-supervised).
- `best_A_total`: concatenated fixed + best learned `A`.

### `uns` (metadata)

- `runtime (seconds)`: total runtime for the training loop.
- `loss`: DataFrame of ELBO loss values, indexed by step.
- `step_w_bestChisq`: step index with the best chi-squared value (if tracked).
- `best_chisq`: best chi-squared value (if tracked).
- `scale`: scale factor used for Gamma models.
- `scale_P` / `scale_A`: scalar Exponential scales (Exponential models).
- `best_scale_P` / `best_scale_A`: best Exponential scales by chi-squared.
- `settings`: DataFrame with the run configuration.

## Pattern naming

Learned patterns are labeled `Pattern_1`, `Pattern_2`, ... by default.
If you provide fixed patterns via a DataFrame, their column names are
preserved and included ahead of the learned patterns for semi-supervised
runs.
