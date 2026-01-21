import sys
import types
import pytest

torch = pytest.importorskip("torch")
pyro = pytest.importorskip("pyro")
pytest.importorskip("anndata")
pytest.importorskip("pandas")

sys.modules.setdefault("scanpy", types.ModuleType("scanpy"))
sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
try:
    import matplotlib.pyplot as _plt  # noqa: F401
except Exception:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

from pyroNMF.models.gamma_NB_models import (
    Gamma_NegBinomial_base,
    Gamma_NegBinomial_SSFixedGenes,
    Gamma_NegBinomial_SSFixedSamples,
)
from pyroNMF.models.exp_pois_models import (
    Exponential_base,
    Exponential_SSFixedGenes,
    Exponential_SSFixedSamples,
)
from pyroNMF.run_inference import setup_model_and_optimizer


@pytest.mark.parametrize(
    "model_type, supervision_type, expected_cls",
    [
        ("gamma_unsupervised", None, Gamma_NegBinomial_base),
        ("gamma_supervised", "fixed_genes", Gamma_NegBinomial_SSFixedGenes),
        ("gamma_supervised", "fixed_samples", Gamma_NegBinomial_SSFixedSamples),
        ("exponential_unsupervised", None, Exponential_base),
        ("exponential_supervised", "fixed_genes", Exponential_SSFixedGenes),
        ("exponential_supervised", "fixed_samples", Exponential_SSFixedSamples),
    ],
)
def test_setup_model_and_optimizer_selects_model(model_type, supervision_type, expected_cls):
    D = torch.ones(3, 4)
    scale = 1.0
    fixed_patterns = None
    if supervision_type == "fixed_genes":
        fixed_patterns = torch.ones(4, 1)
    elif supervision_type == "fixed_samples":
        fixed_patterns = torch.ones(3, 1)

    model, guide, svi = setup_model_and_optimizer(
        D,
        num_patterns=2,
        scale=scale,
        NB_probs=0.5,
        use_chisq=False,
        use_pois=False,
        device=torch.device("cpu"),
        fixed_patterns=fixed_patterns,
        model_type=model_type,
        supervision_type=supervision_type,
    )
    assert isinstance(model, expected_cls)
    assert guide is not None
    assert svi is not None


def test_setup_model_and_optimizer_invalid_model_type():
    D = torch.ones(3, 4)
    with pytest.raises(ValueError, match="model_type"):
        setup_model_and_optimizer(
            D,
            num_patterns=2,
            scale=1.0,
            NB_probs=0.5,
            use_chisq=False,
            use_pois=False,
            device=torch.device("cpu"),
            model_type="not_a_model",
        )


def test_setup_model_and_optimizer_invalid_supervision_type():
    D = torch.ones(3, 4)
    fixed_patterns = torch.ones(4, 1)
    with pytest.raises(ValueError, match="supervision_type"):
        setup_model_and_optimizer(
            D,
            num_patterns=2,
            scale=1.0,
            NB_probs=0.5,
            use_chisq=False,
            use_pois=False,
            device=torch.device("cpu"),
            fixed_patterns=fixed_patterns,
            model_type="gamma_supervised",
            supervision_type="not_valid",
        )
