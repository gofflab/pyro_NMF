# AnnData Integration

pyroNMF accepts an `AnnData` object as input and returns a **copy** of that
object with all results stored in `obsm`, `varm`, and `uns`. The original
input is never modified.

## Input expectations

- `adata.X`: non-negative counts matrix of shape `(n_samples, n_genes)`.
  Dense (`numpy.ndarray`) and sparse (`scipy.sparse`) matrices are both
  supported.
- `adata.obsm['spatial']` (optional): 2-D coordinates of shape
  `(n_samples, 2)` for spatial visualisation. Required when `spatial=True`.
- Named layers in `adata.layers` can be used instead of `.X` via the
  `layer` argument.

---

## Output structure

All outputs are stored as `pandas.DataFrame` objects (in `obsm` / `varm`)
or scalars / DataFrames (in `uns`). Pattern columns are named
`Pattern_1`, `Pattern_2`, … for learned patterns. Fixed-pattern columns
retain the column names from the input `fixed_patterns` DataFrame and appear
**first** (before the learned patterns) in semi-supervised runs.

---

### `obsm` — sample-level factors

All keys are indexed by `adata.obs_names`.

#### Variational parameters

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `loc_P` | `(n_samples, n_patterns)` | exponential, gamma | Variational rate/location of P — one value per sample per pattern. Present for per-element models; **not** present for `exponentialSingle`. |

#### Sampled P — last iteration

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `last_P` | `(n_samples, n_patterns)` | all | P sampled at the final optimisation step. |

#### Sampled P — best chi-squared snapshot

The model always tracks the minimum chi-squared step internally regardless
of whether `use_chisq=True`.

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `best_P` | `(n_samples, n_patterns)` | all | P sampled at the step with the lowest chi-squared. |
| `best_locP` | `(n_samples, n_patterns)` | exponential, gamma | `loc_P` snapshot at the best chi-squared step. |

#### Posterior statistics — sampling phase

Accumulated only during `num_sample_steps` steps. Before accumulation,
each column of P is divided by its sum so that patterns are on a
comparable scale (see [Scale normalisation](#scale-normalisation)).

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `mean_P` | `(n_samples, n_patterns)` | all | Posterior mean of normalised P across all sampling steps. **Primary posterior estimate.** |
| `var_P` | `(n_samples, n_patterns)` | all | Posterior sample variance of normalised P. |
| `sum_P` | `(n_samples, n_patterns)` | all | Raw running sum of normalised P (= `mean_P × num_sample_steps`). |
| `sum_P2` | `(n_samples, n_patterns)` | all | Raw running sum of normalised P² (intermediate for `var_P`). |

#### Scaled patterns

| Key | Shape | Description |
|-----|-------|-------------|
| `best_P_scaled` | `(n_samples, n_patterns)` | `best_P_total` (or `best_P`) with each pattern column normalised to sum 1. |
| `last_P_scaled` | `(n_samples, n_patterns)` | `P_total` (or `last_P`) normalised the same way. |

#### Marker matrices

Binary vote matrices accumulated over the sampling phase. At each sampling
step, the dominant pattern for each sample wins a vote.

| Key | Shape | Description |
|-----|-------|-------------|
| `markers_P` | `(n_samples, n_patterns)` | Votes based on raw P: for each sample, the pattern with the highest P value wins. |
| `markers_Pscaled` | `(n_samples, n_patterns)` | Votes based on column-sum-normalised P. |
| `markers_Psoftmax` | `(n_samples, n_patterns)` | Softmax-normalised P accumulated over sampling steps (soft assignment). |

#### Semi-supervised only — `fixed_samples` mode

| Key | Shape | Description |
|-----|-------|-------------|
| `fixed_P` | `(n_samples, n_fixed)` | The user-supplied fixed sample patterns (unchanged). |
| `P_total` | `(n_samples, n_fixed + n_learned)` | Fixed patterns concatenated with `last_P`. |
| `best_P_total` | `(n_samples, n_fixed + n_learned)` | Fixed patterns concatenated with `best_P`. Present when both `fixed_P` and `best_P` exist. |

---

### `varm` — gene-level factors

All keys are indexed by `adata.var_names`.

#### Variational parameters

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `loc_A` | `(n_genes, n_patterns)` | exponential, gamma | Variational rate/location of A — one value per gene per pattern. Not present for `exponentialSingle`. |

#### Sampled A — last iteration

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `last_A` | `(n_genes, n_patterns)` | all | A sampled at the final optimisation step. |

#### Sampled A — best chi-squared snapshot

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `best_A` | `(n_genes, n_patterns)` | all | A sampled at the step with the lowest chi-squared. |
| `best_locA` | `(n_genes, n_patterns)` | exponential, gamma | `loc_A` snapshot at the best chi-squared step. |

#### Posterior statistics — sampling phase

A is rescaled to absorb the normalisation applied to P before accumulation.

| Key | Shape | Models | Description |
|-----|-------|--------|-------------|
| `mean_A` | `(n_genes, n_patterns)` | all | Posterior mean of scaled A across sampling steps. **Primary posterior estimate.** |
| `var_A` | `(n_genes, n_patterns)` | all | Posterior sample variance of scaled A. |
| `sum_A` | `(n_genes, n_patterns)` | all | Raw running sum of scaled A. |
| `sum_A2` | `(n_genes, n_patterns)` | all | Raw running sum of scaled A². |

#### Scaled patterns

| Key | Shape | Description |
|-----|-------|-------------|
| `best_A_scaled` | `(n_genes, n_patterns)` | `best_A_total` (or `best_A`) multiplied by the column sums of `best_P_total`. |
| `last_A_scaled` | `(n_genes, n_patterns)` | `A_total` (or `last_A`) scaled the same way. |

#### Marker matrices

| Key | Shape | Description |
|-----|-------|-------------|
| `markers_A` | `(n_genes, n_patterns)` | Votes based on raw A: for each gene, the pattern with the highest A value wins. |
| `markers_Ascaled` | `(n_genes, n_patterns)` | Votes based on scale-corrected A. |
| `markers_Asoftmax` | `(n_genes, n_patterns)` | Softmax-normalised A accumulated over sampling steps. |

#### Semi-supervised only — `fixed_genes` mode

| Key | Shape | Description |
|-----|-------|-------------|
| `fixed_A` | `(n_genes, n_fixed)` | The user-supplied fixed gene patterns (unchanged). |
| `A_total` | `(n_genes, n_fixed + n_learned)` | Fixed patterns concatenated with `last_A`. |
| `best_A_total` | `(n_genes, n_fixed + n_learned)` | Fixed patterns concatenated with `best_A`. Present when both exist. |

---

### `uns` — metadata and scalars

#### Run metadata (every run)

| Key | Type | Description |
|-----|------|-------------|
| `runtime (seconds)` | int | Wall-clock time for the training loop. |
| `loss` | DataFrame | ELBO loss values indexed by step (logged every 10 steps). |
| `scale` | float | Scale parameter used for the Gamma prior (`gamma` model only). |
| `settings` | DataFrame | Single-column table of all run configuration values (see below). |

**`settings` keys:**

| Key | Description |
|-----|-------------|
| `num_patterns` | Number of learned patterns. |
| `num_total_steps` | Total steps (burn-in + sampling). |
| `num_sample_steps` | Number of sampling steps. |
| `device` | Device used (`cpu`, `cuda`, or `mps`). |
| `NB_probs` | Negative Binomial probability parameter. |
| `use_chisq` | Whether chi-squared penalty was active. |
| `scale` | Scale factor (Gamma model). |
| `model_type` | Internal model type string (e.g., `exponential_unsupervised`). |
| `tensorboard_identifier` | TensorBoard log directory (if logging was enabled). |

#### Chi-squared tracking

| Key | Type | Description |
|-----|------|-------------|
| `step_w_bestChisq` | int | Step index at which the minimum chi-squared occurred. |
| `best_chisq` | float | The minimum chi-squared value achieved during training. |

#### Model-family-specific scalars

**`exponential` and `gamma`** — variational parameters are per-element
(`loc_A`, `loc_P`) and stored in `varm` / `obsm`; no additional scalar
`uns` entries beyond `scale` (Gamma only).

**`exponentialSingle`** — single shared scalar rates:

| Key | Type | Description |
|-----|------|-------------|
| `scale_P` | float | Learned scalar rate for the P Exponential prior. |
| `scale_A` | float | Learned scalar rate for the A Exponential prior. |
| `best_scale_P` | float | `scale_P` at the best chi-squared step. |
| `best_scale_A` | float | `scale_A` at the best chi-squared step. |

---

## Pattern naming

Learned patterns are labelled `Pattern_1`, `Pattern_2`, … In semi-supervised
runs the fixed-pattern column names (from the input `fixed_patterns`
DataFrame) appear **first**, followed by the learned pattern labels:

```
["Layer1", "Layer2", "Pattern_1", "Pattern_2", ...]
```

This naming is applied uniformly across all `obsm`, `varm`, and `uns`
DataFrame outputs.

---

## Scale normalisation

During the sampling phase, at each step the model normalises P by dividing
each column by its sum, then absorbs the inverse scale into A:

```
Pn = P / P.sum(axis=0)          # each pattern column sums to 1
An = A * P.sum(axis=0)          # A absorbs the removed scale
```

This makes patterns directly comparable across runs and datasets.
`mean_P` and `mean_A` are the averages of `Pn` and `An` over all
`num_sample_steps` steps and are the recommended posterior estimates for
most downstream analyses.
