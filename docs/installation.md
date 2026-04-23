# Installation

pyroNMF is a pure-Python package built on Pyro and PyTorch.

## Editable install (recommended for development)

```bash
pip install -e .
```

## Standard install

```bash
pip install .
```

## Conda environments

Two conda environment files are provided for reproducible setups:

```bash
# Python 3 environment (recommended)
conda env create -f pyro3.yml
conda activate pyro3

# Python 2 legacy environment
conda env create -f pyro2.yml
```

## Optional: build the docs locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

The HTML output will be in `docs/_build/html`.

## Optional: TensorBoard

TensorBoard logging is enabled automatically when you pass
`use_tensorboard_id` to `run_nmf`. To view logs:

```bash
bash run_tensorboard.sh
# or directly:
tensorboard --logdir runs/
```

## GPU support

pyroNMF uses PyTorch and will automatically select CUDA (NVIDIA) or MPS
(Apple Silicon) if available. No additional configuration is needed. To
verify device selection at runtime:

```python
from pyroNMF.utils import detect_device
print(detect_device())   # e.g. device(type='cuda')
```
