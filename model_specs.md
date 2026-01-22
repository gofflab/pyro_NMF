---
header-includes:
  - \usepackage{tikz}
  - \usetikzlibrary{positioning,fit,calc}
---

# Model Specifications (Current Models)

This document describes the **current** probabilistic models implemented in this package, including their statistical formulae, plate notation, parameterizations, optional penalty terms, and the variational inference schemes used by the workflow.

The descriptions here are grounded in the current code under `src/pyroNMF/models/` and `src/pyroNMF/run_inference.py`.

---

## 1) Notation and data preparation

### Indices and dimensions
- Samples (cells/spots): \(i \in \{1,\dots,N\}\)
- Genes (features): \(g \in \{1,\dots,G\}\)
- Patterns (topics/components): \(k \in \{1,\dots,K\}\)
- Fixed patterns (supervised variants): \(k' \in \{1,\dots,K_{\text{fixed}}\}\)

### Observed data
- Counts matrix: \(D \in \mathbb{R}_{\ge 0}^{N \times G}\) (from `adata.X`)
- Uncertainty proxy: \(U \in \mathbb{R}_{>0}^{N \times G}\), constructed as
  \[
  U_{ig} = \max(0.3,\; 0.1\,D_{ig})
  \]
  This is used only in the optional chi-squared penalty.

### Core factorization
All models reconstruct counts via
\[
\Theta = P A
\]
with
- \(P \in \mathbb{R}_{>0}^{N \times K}\) (sample loadings)
- \(A \in \mathbb{R}_{>0}^{K \times G}\) (pattern/gene weights)

For supervised models, \(P\) or \(A\) is augmented with fixed components; see Sections 3–4.

### Scale used in Gamma models
The Gamma models use a **single scalar rate** (named `scale` in code) computed as
\[
\text{scale} = 2\,\mathrm{std}(D)
\]
This value is passed into the Gamma distributions as the **rate** parameter (see below).

---

## 2) Likelihood: Negative Binomial parameterizations

All current models use a **Negative Binomial** observation model with mean-like parameter \(\Theta_{ig}\) and a scalar probability parameter \(p \in (0,1)\) (the `NB_probs` argument, default \(p=0.5\)).

In PyTorch/Pyro, the distribution is parameterized as:
\[
D_{ig} \sim \mathrm{NegBinomial}(\text{total\_count}=\Theta_{ig},\; \text{probs}=p)
\]
where the **mean** is
\[
\mathbb{E}[D_{ig}] = \Theta_{ig}\,\frac{1-p}{p}
\]
(so \(\Theta\) is not the mean unless \(p\) is fixed accordingly).

**Alternative logits parameterization:**
Some MPS paths use
\[
\text{logits} = \log(p) - \log(1-p)
\]
with
\[
\mathrm{NegBinomial}(\text{total\_count}=\Theta_{ig},\; \text{logits}=\text{logits})
\]
The **default** in CPU/CUDA paths is `probs=p`.

**MPS fallback log-prob:**
For certain MPS paths, the code bypasses `NegBinomial` sampling and uses a manually computed log-probability (see Section 4.4).

---

## 3) Gamma–Negative Binomial models

### 3.1 Unsupervised Gamma–NB

#### Generative model
\[
A_{kg} \sim \mathrm{Gamma}(\alpha_{kg},\; \beta)\qquad (\alpha_{kg} = \text{loc\_A}_{kg},\; \beta = \text{scale})
\]
\[
P_{ik} \sim \mathrm{Gamma}(\alpha'_{ik},\; \beta)\qquad (\alpha'_{ik} = \text{loc\_P}_{ik},\; \beta = \text{scale})
\]
\[
\Theta_{ig} = \sum_{k=1}^K P_{ik} A_{kg}
\]
\[
D_{ig} \sim \mathrm{NegBinomial}(\text{total\_count}=\Theta_{ig},\; \text{probs}=p)
\]

**Important parameterization detail:**
- The code uses `dist.Gamma(concentration, rate)`. The variable named `scale` is passed as the **rate** \(\beta\), not the scale parameter. Thus
  \[
  \mathbb{E}[A_{kg}] = \frac{\alpha_{kg}}{\beta},\qquad
  \mathbb{E}[P_{ik}] = \frac{\alpha'_{ik}}{\beta}.
  \]

#### Plate notation (textual)
- Plate **patterns**: \(k = 1..K\)
- Plate **genes**: \(g = 1..G\)
- Plate **samples**: \(i = 1..N\)

Nodes:
- \(A_{kg}\) inside patterns × genes plates
- \(P_{ik}\) inside samples × patterns plates
- \(D_{ig}\) inside samples × genes plates, conditioned on \(P\) and \(A\)

Formal plate diagrams in TikZ/PGF are provided in Section 11.

---

### 3.2 Supervised Gamma–NB with fixed genes (fixed A)

**Inputs:** a fixed pattern matrix \(A^{\text{fixed}} \in \mathbb{R}_{>0}^{G \times K_{\text{fixed}}}\).

#### Generative model
- Learn **new** patterns \(A_{kg}\) for \(k=1..K\).
- Construct an augmented pattern matrix
  \[
  A^{\text{total}} = \begin{bmatrix}(A^{\text{fixed}})^T \\ A \end{bmatrix}
  \in \mathbb{R}_{>0}^{(K_{\text{fixed}}+K) \times G}
  \]

Latents:
\[
A_{kg} \sim \mathrm{Gamma}(\alpha_{kg},\;\beta),\qquad
P_{i,k'} \sim \mathrm{Gamma}(\alpha'_{i,k'},\;\beta)
\]
where \(k'\) runs over \(1..K_{\text{fixed}}+K\).

Likelihood:
\[
\Theta_{ig} = \sum_{k'=1}^{K_{\text{fixed}}+K} P_{i,k'} A^{\text{total}}_{k'g}
\]
\[
D_{ig} \sim \mathrm{NegBinomial}(\Theta_{ig},\;p)
\]

**Note:** \(P\) is learned for **all** fixed + learned patterns. The fixed part applies only to \(A\).

#### Plate notation
- Plate **patterns**: \(k=1..K\) (learned A)
- Plate **patterns\_P**: \(k'=1..K_{\text{fixed}}+K\) (all P columns)
- Plate **genes**: \(g=1..G\)
- Plate **samples**: \(i=1..N\)

---

### 3.3 Supervised Gamma–NB with fixed samples (fixed P)

**Inputs:** a fixed loading matrix \(P^{\text{fixed}} \in \mathbb{R}_{>0}^{N \times K_{\text{fixed}}}\).

#### Generative model
Learn **new** \(P_{ik}\) for \(k=1..K\) and **all** \(A_{k'g}\) for \(k'=1..K_{\text{fixed}}+K\), then set
\[
P^{\text{total}} = \begin{bmatrix} P^{\text{fixed}} & P \end{bmatrix}
\in \mathbb{R}_{>0}^{N \times (K_{\text{fixed}}+K)}
\]

Latents:
\[
A_{k'g} \sim \mathrm{Gamma}(\alpha_{k'g},\;\beta),\qquad
P_{ik} \sim \mathrm{Gamma}(\alpha'_{ik},\;\beta)
\]

Likelihood:
\[
\Theta_{ig} = \sum_{k'=1}^{K_{\text{fixed}}+K} P^{\text{total}}_{i,k'} A_{k'g}
\]
\[
D_{ig} \sim \mathrm{NegBinomial}(\Theta_{ig},\;p)
\]

**Note:** \(A\) is learned for both fixed+learned patterns; \(P\) is only learned for the new patterns.

---

## 4) Exponential–Negative Binomial models

### 4.1 Unsupervised Exponential–NB

#### Generative model
\[
A_{kg} \sim \mathrm{Exponential}(\lambda_A)
\]
\[
P_{ik} \sim \mathrm{Exponential}(\lambda_P)
\]
\[
\Theta_{ig} = \sum_{k=1}^{K} P_{ik} A_{kg}
\]
\[
D_{ig} \sim \mathrm{NegBinomial}(\Theta_{ig},\;p)
\]

**Parameterization detail:**
- In PyTorch, `Exponential(rate)` uses \(\lambda\) as the **rate**. Thus
  \[
  \mathbb{E}[A_{kg}] = 1/\lambda_A,\qquad \mathbb{E}[P_{ik}] = 1/\lambda_P.
  \]
- The code uses **scalar** \(\lambda_A\) and \(\lambda_P\) (learnable Pyro parameters `scale_A` and `scale_P`).

#### Plate notation
Same as Section 3.1, with Exponential priors for \(A\) and \(P\).

---

### 4.2 Supervised Exponential–NB with fixed genes (fixed A)

**Inputs:** \(A^{\text{fixed}} \in \mathbb{R}_{>0}^{G \times K_{\text{fixed}}}\).

Construct
\[
A^{\text{total}} = \begin{bmatrix}(A^{\text{fixed}})^T \\ A \end{bmatrix}
\]
with \(A_{kg} \sim \mathrm{Exponential}(\lambda_A)\).

Learn \(P\) for **all** fixed + learned patterns:
\[
P_{i,k'} \sim \mathrm{Exponential}(\lambda_P),\qquad k'=1..(K_{\text{fixed}}+K)
\]

Likelihood:
\[
D_{ig} \sim \mathrm{NegBinomial}(\Theta_{ig},\;p),
\quad \Theta_{ig} = \sum_{k'} P_{i,k'} A^{\text{total}}_{k'g}
\]

---

### 4.3 Supervised Exponential–NB with fixed samples (fixed P)

**Inputs:** \(P^{\text{fixed}} \in \mathbb{R}_{>0}^{N \times K_{\text{fixed}}}\).

Construct
\[
P^{\text{total}} = \begin{bmatrix} P^{\text{fixed}} & P \end{bmatrix}
\]
with \(P_{ik} \sim \mathrm{Exponential}(\lambda_P)\) for \(k=1..K\).

Learn \(A\) for **all** patterns:
\[
A_{k'g} \sim \mathrm{Exponential}(\lambda_A),\qquad k'=1..(K_{\text{fixed}}+K)
\]

Likelihood:
\[
D_{ig} \sim \mathrm{NegBinomial}(\Theta_{ig},\;p),
\quad \Theta_{ig} = \sum_{k'} P^{\text{total}}_{i,k'} A_{k'g}
\]

---

### 4.4 Minibatching behavior (Exponential models only)

When `batch_size` is set and smaller than \(N\), the Exponential models use a **sample plate with subsampling**:
- The **samples** plate uses a subsample of size \(B\).
- The effective scaling factor is \(N/B\).

In the code:
- \(A\) is always sampled for **all** genes and patterns (global across samples).
- \(P\) is sampled only for the **current batch**; other rows are updated over time.
- Optional penalty terms are manually scaled by \(N/B\).

For MPS devices, the NB likelihood is applied via a custom log-probability and multiplied by \(N/B\).

---

## 5) Optional penalty terms (not part of the core likelihood)

The following terms are **optional** and are added via `pyro.factor(...)`. They act as penalties or auxiliary likelihoods, not as part of the main Negative Binomial observation model.

### 5.1 Chi-squared penalty (`use_chisq`)
Defined as:
\[
\chi^2 = \sum_{i,g} \frac{(\Theta_{ig} - D_{ig})^2}{U_{ig}^2}
\]
and added to the model log-density as:
\[
\log p(\cdot) \leftarrow \log p(\cdot) - \chi^2
\]

For minibatching, the code uses
\[
\chi^2_{\text{scaled}} = (N/B)\,\chi^2_{\text{batch}}
\]

### 5.2 Poisson penalty (`use_pois`)
Defines a Poisson log-likelihood using \(\Theta\) as rate:
\[
\log p_{\text{Pois}}(D|\Theta) = \sum_{i,g} \left(D_{ig} \log \Theta_{ig} - \Theta_{ig} - \log \Gamma(D_{ig}+1)\right)
\]
and adds it to the joint with a fixed multiplier:
\[
\log p(\cdot) \leftarrow \log p(\cdot) + 10\,\log p_{\text{Pois}}(D|\Theta)
\]

For minibatching, the Poisson term is scaled by \(N/B\).

---

## 6) Variational inference and guides

### 6.1 Objective
All runs use **stochastic variational inference (SVI)** with Pyro’s `Trace_ELBO`.
The objective (for a single minibatch \(\mathcal{B}\)) is:
\[
\mathcal{L}(q) = \mathbb{E}_{q}\Big[\log p(D_{\mathcal{B}}, A, P) + \text{(optional penalties)} - \log q(A,P)\Big]
\]
The plate subsampling scales log-likelihood contributions by \(N/B\) internally, while penalty terms are manually scaled as noted.

### 6.2 Default guide: `AutoNormal`
By default, all models use Pyro’s `AutoNormal` guide. This constructs a **mean-field Normal** distribution in an unconstrained space and uses appropriate transforms to respect support (e.g., positive support for Gamma/Exponential latents).

Conceptually:
\[
q(A,P) = \prod_{k,g} \mathcal{N}(z^A_{kg} \mid \mu^A_{kg}, \sigma^A_{kg}) \, \prod_{i,k} \mathcal{N}(z^P_{ik} \mid \mu^P_{ik}, \sigma^P_{ik})
\]
with a bijective transform \(A = f(z^A), P = f(z^P)\) to ensure positivity.

### 6.3 Optional guide for Exponential models: `--param-P`
When `--param-P` is enabled (Exponential models only), the guide is replaced by a custom **LogNormal mean-field** guide:
\[
A_{kg} \sim \mathrm{LogNormal}(q\_\mu^A_{kg}, q\_\sigma^A_{kg})
\]
\[
P_{ik} \sim \mathrm{LogNormal}(q\_\mu^P_{ik}, q\_\sigma^P_{ik})
\]

Key details:
- The variational parameters \(q\_\mu^A, q\_\sigma^A\) are per-element for \(A\).
- The variational parameters \(q\_\mu^P, q\_\sigma^P\) are per-element for **all samples**, even in minibatch mode.
- When minibatching, only the rows of \(P\) in the current batch are sampled/updated each step.

In the workflow, when `--param-P` is enabled, the stored \(P\) in outputs is set to the **mean** of the LogNormal:
\[
\mathbb{E}[P] = \exp\left(q\_\mu^P + \tfrac{1}{2}(q\_\sigma^P)^2\right)
\]

---

## 7) Summary table of model variants

| Model | Prior on A | Prior on P | Fixed genes? | Fixed samples? | Likelihood |
|------|------------|------------|--------------|----------------|------------|
| Gamma unsupervised | Gamma(\(\alpha_{kg}\), \(\beta\)) | Gamma(\(\alpha'_{ik}\), \(\beta\)) | No | No | NB(\(\Theta\), p) |
| Gamma fixed genes | Gamma (learned \(A\)) + fixed \(A^{\text{fixed}}\) | Gamma for all \(P\) | Yes | No | NB |
| Gamma fixed samples | Gamma for all \(A\) | Gamma (learned \(P\)) + fixed \(P^{\text{fixed}}\) | No | Yes | NB |
| Exp unsupervised | Exponential(\(\lambda_A\)) | Exponential(\(\lambda_P\)) | No | No | NB |
| Exp fixed genes | Exponential + fixed \(A^{\text{fixed}}\) | Exponential for all \(P\) | Yes | No | NB |
| Exp fixed samples | Exponential for all \(A\) | Exponential + fixed \(P^{\text{fixed}}\) | No | Yes | NB |

---

## 8) Defaults and notable implementation choices

- **NB probability**: default \(p=0.5\).
- **Gamma rate (named scale)**: default \(\beta = 2\,\mathrm{std}(D)\).
- **Exponential rates**: learnable scalars initialized to 1.0.
- **Penalties**: disabled by default (`use_chisq=False`, `use_pois=False`).
- **Guide**: `AutoNormal` by default; `--param-P` activates LogNormal guide for Exponential models.
- **Minibatching**: implemented for Exponential models only.
- **Minibatch storage** (Exponential): when `batch_size < N`, per-sample tensors (`P`, `best_P`, `sum_P`, etc.) are kept on CPU (`storage_device`) while `A` remains on the compute device.
- **Tensor prep** (Exponential minibatch): `prepare_tensors(..., keep_on_cpu=True)` keeps `D/U/scale` on CPU to reduce MPS/VRAM pressure.

---

## 9) Exact Pyro log-joint expressions

This section writes the **explicit** log-joint expressions that correspond to the Pyro model code. These are the exact terms summed by `pyro.sample` (and `pyro.factor`) for the current models.

### 9.1 Unsupervised Gamma–NB log joint

Let \(\alpha_{kg} = \text{loc\_A}_{kg}\), \(\alpha'_{ik} = \text{loc\_P}_{ik}\), and \(\beta = \text{scale}\) (rate). Then

\[
\log p(D,A,P) = \sum_{k,g} \log \mathrm{Gamma}(A_{kg};\,\alpha_{kg},\beta)
+ \sum_{i,k} \log \mathrm{Gamma}(P_{ik};\,\alpha'_{ik},\beta)
+ \sum_{i,g} \log \mathrm{NB}(D_{ig};\,\Theta_{ig},p)
+ \mathcal{L}_{\text{pen}}\,.
\]

Expanding the Gamma terms:
\[
\log \mathrm{Gamma}(x;\alpha,\beta) = (\alpha-1)\log x - \beta x + \alpha \log \beta - \log \Gamma(\alpha)
\]
so
\[
\log p(D,A,P) = \sum_{k,g} \Big[(\alpha_{kg}-1)\log A_{kg} - \beta A_{kg} + \alpha_{kg}\log\beta - \log\Gamma(\alpha_{kg})\Big]
\]
\[
\quad + \sum_{i,k} \Big[(\alpha'_{ik}-1)\log P_{ik} - \beta P_{ik} + \alpha'_{ik}\log\beta - \log\Gamma(\alpha'_{ik})\Big]
+ \sum_{i,g} \log \mathrm{NB}(D_{ig};\,\Theta_{ig},p)
+ \mathcal{L}_{\text{pen}}.
\]

The NB term (default `probs=p`) is:
\[
\log \mathrm{NB}(D;\Theta,p) = \log \Gamma(D+\Theta) - \log \Gamma(\Theta) - \log \Gamma(D+1)
+ \Theta \log(1-p) + D \log p.
\]

### 9.2 Unsupervised Exponential–NB log joint

Let \(\lambda_A\) and \(\lambda_P\) be the Exponential rates (`scale_A`, `scale_P`). Then
\[
\log p(D,A,P) = \sum_{k,g} \log \mathrm{Exp}(A_{kg};\lambda_A)
+ \sum_{i,k} \log \mathrm{Exp}(P_{ik};\lambda_P)
+ \sum_{i,g} \log \mathrm{NB}(D_{ig};\Theta_{ig},p)
+ \mathcal{L}_{\text{pen}}.
\]

With
\[
\log \mathrm{Exp}(x;\lambda) = \log \lambda - \lambda x.
\]

### 9.3 Supervised variants

All supervised variants use the same likelihood term with
\(\Theta = P A\), but with fixed blocks inserted:

- **Fixed genes (fixed A):**
  \(A^{\text{total}} = [(A^{\text{fixed}})^T; A]\), and \(\Theta = P A^{\text{total}}\).
- **Fixed samples (fixed P):**
  \(P^{\text{total}} = [P^{\text{fixed}}, P]\), and \(\Theta = P^{\text{total}} A\).

The log joint is identical to Sections 9.1 or 9.2, except that the prior is applied only to the **learned** blocks (the fixed blocks are constants and do not contribute a prior term).

### 9.4 MPS / logits parameterizations

For MPS devices, some paths compute the NB log-probability manually. The expression used is equivalent to the formula above and uses
\[
\log p = \log(\text{probs}) - \log(1-\text{probs})
\]
when the logits parameterization is required.

---

## 10) Gradient derivations (log-joint w.r.t. A and P)

Let
\[
\Theta = P A \in \mathbb{R}_{>0}^{N \times G},
\quad \ell(D,\Theta) = \sum_{i,g} \log \mathrm{NB}(D_{ig};\Theta_{ig},p).
\]

### 10.1 Derivative of NB log-likelihood w.r.t. \(\Theta\)
Using the NB log-likelihood from Section 9:
\[
\frac{\partial \ell}{\partial \Theta_{ig}} = \psi(D_{ig}+\Theta_{ig}) - \psi(\Theta_{ig}) + \log(1-p)
\]
where \(\psi\) is the digamma function.

Define the matrix
\[
G_{ig} = \frac{\partial \ell}{\partial \Theta_{ig}}.
\]

### 10.2 Chain rule to A and P
Since \(\Theta = P A\),
\[
\frac{\partial \ell}{\partial A_{kg}} = \sum_i G_{ig} P_{ik},
\qquad
\frac{\partial \ell}{\partial P_{ik}} = \sum_g G_{ig} A_{kg}.
\]
In matrix form:
\[
\nabla_A \ell = P^T G,
\qquad
\nabla_P \ell = G A^T.
\]

### 10.3 Gradients of prior terms

**Gamma prior (rate \(\beta\)):**
\[
\log p(A_{kg}) = (\alpha_{kg}-1)\log A_{kg} - \beta A_{kg} + \alpha_{kg}\log\beta - \log\Gamma(\alpha_{kg})
\]
\[
\frac{\partial}{\partial A_{kg}} \log p(A_{kg}) = \frac{\alpha_{kg}-1}{A_{kg}} - \beta.
\]
The same formula applies for \(P_{ik}\) with \(\alpha'_{ik}\).

**Exponential prior (rate \(\lambda\)):**
\[
\log p(A_{kg}) = \log \lambda_A - \lambda_A A_{kg}
\]
\[
\frac{\partial}{\partial A_{kg}} \log p(A_{kg}) = -\lambda_A.
\]
Again, the same holds for \(P_{ik}\) with \(\lambda_P\).

**If optimizing hyperparameters:**
- For Gamma rate \(\beta\):
  \(\partial/\partial \beta \sum_{k,g} \log p(A_{kg}) = \sum_{k,g} (\alpha_{kg}/\beta - A_{kg})\) and similarly for \(P\).
- For Exponential rate \(\lambda\):
  \(\partial/\partial \lambda_A \sum_{k,g} \log p(A_{kg}) = \frac{KG}{\lambda_A} - \sum_{k,g} A_{kg}.\)

### 10.4 Gradients of optional penalty terms

**Chi-squared penalty**
\[
\chi^2 = \sum_{i,g} \frac{(\Theta_{ig}-D_{ig})^2}{U_{ig}^2}
\]
\[
\frac{\partial}{\partial \Theta_{ig}}(-\chi^2) = -\frac{2(\Theta_{ig}-D_{ig})}{U_{ig}^2}.
\]

**Poisson penalty (multiplier 10)**
\[
\log p_{\text{Pois}}(D|\Theta) = \sum_{i,g} \left(D_{ig}\log\Theta_{ig} - \Theta_{ig} - \log\Gamma(D_{ig}+1)\right)
\]
\[
\frac{\partial}{\partial \Theta_{ig}} \left(10\,\log p_{\text{Pois}}\right) = 10\left(\frac{D_{ig}}{\Theta_{ig}} - 1\right).
\]

For minibatched Exponential models, the penalty derivatives are multiplied by \(N/B\) (matching the code’s scaling).

### 10.5 Full gradient summary

Let
\[
H_{ig} = G_{ig} + \text{(penalty contributions in }\Theta_{ig}\text{)}.
\]
Then the total gradient of the log-joint with respect to the **learned** matrices is:
\[
\nabla_A \log p = P^T H + \nabla_A \log p(A)\,,
\qquad
\nabla_P \log p = H A^T + \nabla_P \log p(P).
\]
For supervised models, the gradient applies only to the learned block (fixed blocks have zero gradient).

---

## 11) Formal plate notation diagrams (TikZ/PGF)

Below are TikZ/PGF diagrams for the core model families. To render them in LaTeX, include:

- `\usepackage{tikz}`
- `\usetikzlibrary{positioning,fit}`

### 11.1 Unsupervised model (Gamma–NB or Exponential–NB)

```{=latex}
% Unsupervised NMF plate diagram
\begin{tikzpicture}[node distance=1.2cm,
  latent/.style={circle,draw,inner sep=1.5pt},
  obs/.style={circle,draw,fill=gray!20,inner sep=1.5pt},
  det/.style={rectangle,draw,inner sep=1.5pt},
  plate/.style={draw,rounded corners,inner sep=6pt}]

\node[latent] (A) {$A_{kg}$};
\node[latent, right=2.5cm of A] (P) {$P_{ik}$};
\node[det, below=1.3cm of $(A)!0.5!(P)$] (T) {$\Theta_{ig}=\sum_k P_{ik}A_{kg}$};
\node[obs, below=1.0cm of T] (D) {$D_{ig}$};

\draw[->] (A) -- (T);
\draw[->] (P) -- (T);
\draw[->] (T) -- (D);

\node[plate, fit=(A)] (plateA) {};
\node[plate, fit=(P)] (plateP) {};
\node[plate, fit=(T) (D)] (plateD) {};

\node[anchor=north east] at (plateA.north east) {patterns $k$, genes $g$};
\node[anchor=north east] at (plateP.north east) {samples $i$, patterns $k$};
\node[anchor=north east] at (plateD.north east) {samples $i$, genes $g$};
\end{tikzpicture}
```

### 11.2 Fixed genes (fixed A)

```{=latex}
% Fixed genes: A_fixed is constant, A is learned
\begin{tikzpicture}[node distance=1.1cm,
  latent/.style={circle,draw,inner sep=1.5pt},
  obs/.style={circle,draw,fill=gray!20,inner sep=1.5pt},
  det/.style={rectangle,draw,inner sep=1.5pt},
  const/.style={rectangle,draw,fill=gray!15,inner sep=1.5pt},
  plate/.style={draw,rounded corners,inner sep=6pt}]

\node[const] (Afix) {$A^{\text{fixed}}_{k'g}$};
\node[latent, right=2.2cm of Afix] (A) {$A_{kg}$};
\node[det, below=1.2cm of $(Afix)!0.5!(A)$] (Atot) {$A^{\text{total}}$};
\node[latent, right=2.5cm of Atot] (P) {$P_{i,k'}$};
\node[det, below=1.0cm of $(Atot)!0.5!(P)$] (T) {$\Theta_{ig}=\sum_{k'} P_{i,k'}A^{\text{total}}_{k'g}$};
\node[obs, below=1.0cm of T] (D) {$D_{ig}$};

\draw[->] (Afix) -- (Atot);
\draw[->] (A) -- (Atot);
\draw[->] (Atot) -- (T);
\draw[->] (P) -- (T);
\draw[->] (T) -- (D);

\node[plate, fit=(A)] (plateA) {};
\node[plate, fit=(Afix)] (plateAfix) {};
\node[plate, fit=(P)] (plateP) {};
\node[plate, fit=(T) (D)] (plateD) {};

\node[anchor=north east] at (plateA.north east) {learned patterns $k$, genes $g$};
\node[anchor=north east] at (plateAfix.north east) {fixed patterns $k'$, genes $g$};
\node[anchor=north east] at (plateP.north east) {samples $i$, patterns $k'$};
\node[anchor=north east] at (plateD.north east) {samples $i$, genes $g$};
\end{tikzpicture}
```

### 11.3 Fixed samples (fixed P)

```{=latex}
% Fixed samples: P_fixed is constant, P is learned
\begin{tikzpicture}[node distance=1.1cm,
  latent/.style={circle,draw,inner sep=1.5pt},
  obs/.style={circle,draw,fill=gray!20,inner sep=1.5pt},
  det/.style={rectangle,draw,inner sep=1.5pt},
  const/.style={rectangle,draw,fill=gray!15,inner sep=1.5pt},
  plate/.style={draw,rounded corners,inner sep=6pt}]

\node[const] (Pfix) {$P^{\text{fixed}}_{i,k'}$};
\node[latent, right=2.2cm of Pfix] (P) {$P_{ik}$};
\node[det, below=1.2cm of $(Pfix)!0.5!(P)$] (Ptot) {$P^{\text{total}}$};
\node[latent, right=2.5cm of Ptot] (A) {$A_{k'g}$};
\node[det, below=1.0cm of $(Ptot)!0.5!(A)$] (T) {$\Theta_{ig}=\sum_{k'} P^{\text{total}}_{i,k'}A_{k'g}$};
\node[obs, below=1.0cm of T] (D) {$D_{ig}$};

\draw[->] (Pfix) -- (Ptot);
\draw[->] (P) -- (Ptot);
\draw[->] (Ptot) -- (T);
\draw[->] (A) -- (T);
\draw[->] (T) -- (D);

\node[plate, fit=(P)] (plateP) {};
\node[plate, fit=(Pfix)] (platePfix) {};
\node[plate, fit=(A)] (plateA) {};
\node[plate, fit=(T) (D)] (plateD) {};

\node[anchor=north east] at (plateP.north east) {learned patterns $k$, samples $i$};
\node[anchor=north east] at (platePfix.north east) {fixed patterns $k'$, samples $i$};
\node[anchor=north east] at (plateA.north east) {patterns $k'$, genes $g$};
\node[anchor=north east] at (plateD.north east) {samples $i$, genes $g$};
\end{tikzpicture}
```

---
