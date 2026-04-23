# API Reference

## High-level inference wrappers

The primary entry point for all runs — unsupervised and semi-supervised.

```{automodule} pyroNMF.run_inference
:members:
:undoc-members:
:show-inheritance:
```

## Utilities

Device detection and spatial/pattern plotting helpers.

```{automodule} pyroNMF.utils
:members:
:undoc-members:
:show-inheritance:
```

## Model families

### Exponential — Negative Binomial (primary)

The recommended model family. Places an independent Exponential prior on
every element of both `A` and `P`. The variational parameters (`loc_A` and
`loc_P`) are per-element rate values learned by SVI. This family is the
primary choice for most datasets.

```{automodule} pyroNMF.models.exp_NB_models
:members:
:undoc-members:
:show-inheritance:
```

### Gamma — Negative Binomial (alternative)

Places a Gamma prior on every element of `A` and `P`. Variational
parameters are per-element location values (`loc_A`, `loc_P`); a shared
scalar `scale` controls the concentration. Use this family when you want
heavier-tailed priors or when the Gamma parameterisation fits domain
knowledge better.

```{automodule} pyroNMF.models.gamma_NB_models
:members:
:undoc-members:
:show-inheritance:
```

### ExponentialSingle — Negative Binomial (alternative)

A more parsimonious Exponential parameterisation that uses a single shared
scalar rate per matrix (`scale_A` for `A`, `scale_P` for `P`) rather than
per-element rates. Faster convergence on smaller datasets; less flexible
than the per-element Exponential model.

```{automodule} pyroNMF.models.expSingle_NB_models
:members:
:undoc-members:
:show-inheritance:
```
