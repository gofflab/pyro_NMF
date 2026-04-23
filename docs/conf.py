import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "pyroNMF"
author = "Kyla Woyshner"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "private-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Mock heavy or optional dependencies so docs can build without them.
autodoc_mock_imports = [
    "pyro",
    "pyro.distributions",
    "pyro.nn",
    "pyro.nn.module",
    "pyro.optim",
    "pyro.infer",
    "torch",
    "torch.utils",
    "torch.utils.tensorboard",
    "scanpy",
    "anndata",
    "numpy",
    "pandas",
    "seaborn",
    "matplotlib",
    "matplotlib.pyplot",
    "pyroNMF.models.gamma_NB_newBase",
    "pyroNMF.models.gamma_NB_new_SSfixedP",
    "pyroNMF.models.exp_pois",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "legacy.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {}
