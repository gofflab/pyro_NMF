# pyroNMF

pyroNMF provides probabilistic non-negative matrix factorisation (NMF) built
on [Pyro](https://pyro.ai). It integrates directly with
[AnnData](https://anndata.readthedocs.io) for both input and output.

## Key features

- **Unified API** — one function, `run_nmf()`, handles unsupervised and
  semi-supervised (fixed genes *or* fixed samples) runs.
- **Primary model: Exponential NB** — per-element Exponential priors on `A`
  and `P` with independent variational rate parameters. Gamma (with a shared
  concentration scale) and ExponentialSingle (shared scalar rates) are
  available as alternatives.
- **Burn-in / sampling split** — a configurable warm-up phase followed by a
  sampling phase that accumulates posterior means, variances, and marker
  matrices.
- **AnnData-native outputs** — all results (variational parameters, posterior
  statistics, marker matrices, run metadata) are written directly into
  `obsm`, `varm`, and `uns`.
- **TensorBoard integration** — optional per-step logging of loss,
  chi-squared, parameter histograms, spatial plots, reconstruction
  diagnostics, and Adam optimizer state.
- **Automatic device selection** — CUDA, MPS (Apple Silicon), and CPU
  detected and used automatically.

:::{toctree}
:maxdepth: 2
:caption: User Guide

installation
usage
tutorial
anndata
legacy
:::

:::{toctree}
:maxdepth: 2
:caption: API Reference

api
:::
