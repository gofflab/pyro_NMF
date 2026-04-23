# Tutorial (Worked Example)

This tutorial runs end-to-end with synthetic data so you can verify your
installation and explore the full API without needing external files.
All output blocks below were captured from an actual run on CPU (PyTorch
2.11.0, Pyro 1.9.1).

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
```

```
AnnData object with n_obs × n_vars = 200 × 100
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

```
################## Preparing tensors ##################
Selecting device cpu
Setting D from data.X
Data contains 200 cells and 100 genes
Data is 0.57% sparse
Using default uncertainty, 10% expression (clipped at 0.3)
Using default scale as 2*std(data) = 15.497018814086914
Using unsupervised mode
model_type is exponential_unsupervised
 ################# Running Exponential-Negative Binomial Model #################
Using cpu
Data is 200 samples x 100 genes
Running for 5 patterns
Using Negative Binomial with probs of 0.5
Not using chi squared
Iteration 100, ELBO loss: 58330.709
Iteration 200, ELBO loss: 55633.375
Iteration 300, ELBO loss: 55279.058
Iteration 400, ELBO loss: 55206.828
Iteration 500, ELBO loss: 55131.915
Iteration 600, ELBO loss: 55231.797
Iteration 700, ELBO loss: 55237.097
...
Iteration 2400, ELBO loss: 55205.898
Iteration 2500, ELBO loss: 55108.860
Runtime: 26 seconds
Saving loc_P in anndata.obsm['loc_P']
Saving loc_A in anndata.varm['loc_A']
...
Saving markers_Psoftmax in anndata.obsm['markers_Psoftmax']
Saving markers_Asoftmax in anndata.varm['markers_Asoftmax']
```

`run_nmf` prints the device, data shape, sparsity, and ELBO loss every
100 steps, then reports total runtime.

---

## 3) Inspect the AnnData output

### Sample-level factors (`obsm`)

```python
print(list(result.obsm.keys()))
```

```
['loc_P', 'last_P', 'best_locP', 'best_P', 'best_P_scaled', 'last_P_scaled',
 'sum_P', 'mean_P', 'sum_P2', 'var_P', 'markers_P', 'markers_Pscaled', 'markers_Psoftmax']
```

```python
# Variational rate parameter — one value per sample per pattern
print(result.obsm["loc_P"].shape)
print(result.obsm["loc_P"].head())
```

```
(200, 5)
        Pattern_1  Pattern_2   Pattern_3  Pattern_4   Pattern_5
cell_0   1.667172   0.608294    0.305472   0.238403    0.992584
cell_1   0.548441   1.447245  178.860291   1.051520  110.229836
cell_2   0.711036   0.671135    2.388549   7.821187    0.466882
cell_3   0.266198  32.025463    0.544798   4.445707    1.036459
cell_4   0.158986  29.166496    0.318825   0.336668    0.649566
```

```python
# Posterior mean across sampling steps — recommended primary estimate
print(result.obsm["mean_P"].head())
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0   0.000648   0.003359   0.010183   0.012225   0.004348
cell_1   0.004294   0.002112   0.000055   0.003564   0.000054
cell_2   0.003774   0.004722   0.000939   0.000347   0.007891
cell_3   0.006934   0.000768   0.005203   0.000646   0.004148
cell_4   0.013691   0.000213   0.010457   0.010327   0.006177
```

```python
# Posterior variance
print(result.obsm["var_P"].head())
```

```
           Pattern_1     Pattern_2     Pattern_3     Pattern_4     Pattern_5
cell_0  4.322748e-07  9.587264e-07  1.735847e-06  2.383381e-06  1.409628e-06
cell_1  3.090724e-07  3.032398e-07  8.982721e-09  7.478674e-07  7.918980e-09
cell_2  3.753318e-07  6.454778e-07  4.464640e-07  1.682493e-07  1.344377e-06
cell_3  6.188529e-07  3.564078e-07  9.011121e-07  4.258585e-07  1.012403e-06
cell_4  1.494783e-06  8.023429e-08  2.105991e-06  2.487033e-06  2.138130e-06
```

### Gene-level factors (`varm`)

```python
print(list(result.varm.keys()))
```

```
['loc_A', 'last_A', 'best_locA', 'best_A', 'best_A_scaled', 'last_A_scaled',
 'sum_A', 'mean_A', 'sum_A2', 'var_A', 'markers_A', 'markers_Ascaled', 'markers_Asoftmax']
```

```python
# Posterior mean gene loadings
print(result.varm["mean_A"].shape)
print(result.varm["mean_A"].head())
```

```
(100, 5)
         Pattern_1    Pattern_2    Pattern_3    Pattern_4   Pattern_5
gene_0  585.600952   484.509094   280.306458   261.224457  759.314941
gene_1  848.209106  1428.511353  1405.425659  1002.956604  656.212830
gene_2  339.429657   695.986694   512.539978   144.224899  167.941177
gene_3  652.377686   523.066895   206.035049   159.613800  989.213196
gene_4  322.150238   180.197739   277.287567   246.682861  370.790070
```

### Training metadata (`uns`)

```python
print(result.uns["runtime (seconds)"])
```

```
26
```

```python
# ELBO loss curve
import matplotlib.pyplot as plt
result.uns["loss"].plot(title="ELBO loss")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()

# Chi-squared tracking (always recorded)
print(result.uns["best_chisq"])
print(result.uns["step_w_bestChisq"])
```

```
385833.5
1387
```

```python
# ELBO loss tail
print(result.uns["loss"].tail())
```

```
              loss
2460  55148.698156
2470  55209.268963
2480  55073.447113
2490  55076.066483
2500  55108.859676
```

```python
# Full run settings
print(result.uns["settings"])
```

```
                                  settings
num_patterns                             5
num_total_steps                       2500
num_sample_steps                      2000
device                                 cpu
NB_probs                               0.5
use_chisq                            False
scale                      tensor(15.4970)
model_type        exponential_unsupervised
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
```

```
Best chi-sq: 225169.88 at step 1230
```

```python
print(result_chisq.obsm["best_P"].head())
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0   0.860352   0.324637   2.211105   1.112204   2.878320
cell_1   1.303943   0.015974   0.028302   0.866747   0.606835
cell_2   0.669028   1.496210   0.814473   1.019395   0.539855
cell_3   2.027148   0.883685   1.451072   0.442138   0.323270
cell_4   4.311213   1.359198   3.329028   0.188981   3.151751
```

```python
print(result_chisq.obsm["best_locP"].head())   # per-element rate at best step
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0   1.216447   2.816250   0.444813   1.007458   0.342150
cell_1   0.786761  79.558907  58.547676   1.104241   1.566754
cell_2   1.588084   0.600809   1.273463   0.971758   1.972649
cell_3   0.501210   1.151662   0.695488   2.581865   2.626462
cell_4   0.227723   0.725926   0.305900   4.871953   0.317379
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
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0   5.468161  51.355213  15.099794  45.382050  19.904964
cell_1   5.632966  13.743155  19.427166   0.364568  26.245897
cell_2  35.699245   5.853901  24.853554  14.076097  10.241508
cell_3  25.085579   6.428365   6.449060  25.775684  37.816330
cell_4  32.269863  47.459190   6.779791  42.983635  87.009705
```

```python
# Alternative — single shared Exponential scale
result_single = run_nmf(
    adata, num_patterns=5, num_burnin=500, num_sample_steps=2000,
    model_family="exponentialSingle",
)
# ExponentialSingle has scalar rates in uns instead of per-element loc in obsm
print(result_single.uns["scale_P"])   # scalar
print(result_single.uns["scale_A"])
```

```
0.6397722959518433
0.6668129563331604
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
```

```
['Fixed_1', 'Fixed_2', 'Pattern_1', 'Pattern_2', 'Pattern_3']
```

```python
print(result_ss.obsm["fixed_P"].head())
```

```
         Fixed_1   Fixed_2
cell_0  1.307808  2.004122
cell_1  0.614641  0.026297
cell_2  1.033268  0.585576
cell_3  0.380642  1.002996
cell_4  0.266696  3.307766
```

```python
print(result_ss.obsm["mean_P"].head())
```

```
         Fixed_1   Fixed_2  Pattern_1  Pattern_2  Pattern_3
cell_0  0.003882  0.006739   0.012906   0.003098   0.000578
cell_1  0.001825  0.000088   0.004106   0.006934   0.001137
cell_2  0.003067  0.001969   0.001581   0.001553   0.009498
cell_3  0.001130  0.003373   0.002153   0.008357   0.006500
cell_4  0.000792  0.011122   0.011983   0.017640   0.007464
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
```

```
################## Preparing tensors ##################
Selecting device cpu
...
Runtime: 7 seconds
Saving loc_P in anndata.obsm['loc_P']
...
Saving markers_Psoftmax in anndata.obsm['markers_Psoftmax']
Saving markers_Asoftmax in anndata.varm['markers_Asoftmax']
```

```python
print(f"Runtime: {result_spatial.uns['runtime (seconds)']:.1f}s")
print("obsm keys:", list(result_spatial.obsm.keys()))
print("mean_P shape:", result_spatial.obsm["mean_P"].shape)
print(result_spatial.obsm["mean_P"].head())
```

```
Runtime: 7.0s
obsm keys: ['spatial', 'loc_P', 'last_P', 'best_locP', 'best_P', 'best_P_scaled',
            'last_P_scaled', 'sum_P', 'mean_P', 'sum_P2', 'var_P', 'markers_P',
            'markers_Pscaled', 'markers_Psoftmax']
mean_P shape: (200, 5)
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0   0.003356   0.000568   0.010848   0.004566   0.007704
cell_1   0.003702   0.001078   0.001336   0.005879   0.000150
cell_2   0.004195   0.007036   0.002733   0.000540   0.002486
cell_3   0.000699   0.006310   0.002633   0.006201   0.004204
cell_4   0.001465   0.008559   0.011384   0.017021   0.006947
```

```python
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
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0        0.0        0.0        0.0      985.0     1014.0
cell_1       26.0     1823.0        0.0      149.0        1.0
cell_2       56.0        6.0     1933.0        0.0        4.0
cell_3        0.0     1970.0        0.0        1.0       28.0
cell_4        0.0     1983.0        0.0       15.0        1.0
```

```python
# Convert to fraction
n_samp_steps = int(result.uns["settings"].loc["num_sample_steps", "settings"])
frac_P = result.obsm["markers_P"] / n_samp_steps
print(frac_P.head())
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0      0.000     0.0000     0.0000     0.4925     0.5070
cell_1      0.013     0.9115     0.0000     0.0745     0.0005
cell_2      0.028     0.0030     0.9665     0.0000     0.0020
cell_3      0.000     0.9850     0.0000     0.0005     0.0140
cell_4      0.000     0.9915     0.0000     0.0075     0.0005
```

```python
# Gene-level dominant pattern
print(result.varm["markers_A"].head())
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
gene_0        0.0        0.0     1998.0        0.0        1.0
gene_1       31.0        0.0        0.0        1.0     1967.0
gene_2     1164.0        0.0        0.0        0.0      835.0
gene_3        0.0        0.0     1999.0        0.0        0.0
gene_4        0.0        7.0     1124.0       21.0      847.0
```

```python
# Soft assignment (softmax-normalised accumulation)
print(result.obsm["markers_Psoftmax"].head())
```

```
         Pattern_1   Pattern_2   Pattern_3   Pattern_4   Pattern_5
cell_0  163.522827  107.012337  126.692444  893.918762  707.853516
cell_1  518.763367  802.486267   29.526495  642.464050    5.760798
cell_2  490.073669  313.498627  911.656128   38.339134  245.432999
cell_3   89.733711  800.795166  486.470764  100.820778  521.178894
cell_4   14.370378  723.130310  247.362503  569.449707  444.686890
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

```
################## Preparing tensors ##################
Selecting device cpu
...
Using user-specified uncertainty
...
Using chi squared
Iteration 100, ELBO loss: 90498.558
Iteration 200, ELBO loss: 75942.624
...
Iteration 2500, ELBO loss: 74801.146
Runtime: 26 seconds
```

```python
print(f"Runtime: {result_custom.uns['runtime (seconds)']:.1f}s")
print(f"best_chisq: {result_custom.uns['best_chisq']:.2f}")
print(f"step_w_bestChisq: {result_custom.uns['step_w_bestChisq']}")
```

```
Runtime: 26.0s
best_chisq: 18787.14
step_w_bestChisq: 2467
```

```python
print(result_custom.obsm["mean_P"].head())
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0   0.003881   0.005700   0.000047   0.006141   0.011803
cell_1   0.004466   0.003471   0.001096   0.000033   0.001845
cell_2   0.002542   0.001445   0.007205   0.004085   0.001782
cell_3   0.001395   0.006709   0.005548   0.003844   0.001569
cell_4   0.004031   0.016126   0.007945   0.005433   0.011569
```

```python
print(result_custom.obsm["best_P"].head())
```

```
        Pattern_1  Pattern_2  Pattern_3  Pattern_4  Pattern_5
cell_0   6.005092   8.295854   0.064301  11.082789  24.737921
cell_1   5.171925   6.785193   1.418844   0.000955   5.077575
cell_2   4.786214   2.136858   9.628098   4.385066   1.663316
cell_3   2.962180  11.560098   7.301690   8.087120   2.705578
cell_4   2.405885  25.113226  11.122270  11.886293  21.428005
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
