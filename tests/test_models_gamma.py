import math
import pytest

torch = pytest.importorskip("torch")
pyro = pytest.importorskip("pyro")

from pyroNMF.models.gamma_NB_models import (
    Gamma_NegBinomial_base,
    Gamma_NegBinomial_SSFixedGenes,
    Gamma_NegBinomial_SSFixedSamples,
)


def _counts_tensor():
    return torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_gamma_base_forward_shapes():
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    D = _counts_tensor()
    U = torch.ones_like(D)
    model = Gamma_NegBinomial_base(
        num_samples=D.shape[0],
        num_genes=D.shape[1],
        num_patterns=2,
        use_chisq=False,
        use_pois=False,
        scale=1.0,
        NB_probs=0.5,
        device=torch.device("cpu"),
    )
    model(D, U)
    assert model.A.shape == (2, 4)
    assert model.P.shape == (3, 2)
    assert model.D_reconstructed.shape == D.shape
    assert math.isfinite(float(model.best_chisq))


def test_gamma_fixed_genes_shapes():
    pyro.clear_param_store()
    pyro.set_rng_seed(1)
    D = _counts_tensor()
    U = torch.ones_like(D)
    fixed_patterns = torch.ones(D.shape[1], 1)
    model = Gamma_NegBinomial_SSFixedGenes(
        num_samples=D.shape[0],
        num_genes=D.shape[1],
        num_patterns=2,
        fixed_patterns=fixed_patterns,
        use_chisq=False,
        use_pois=False,
        scale=1.0,
        NB_probs=0.5,
        device=torch.device("cpu"),
    )
    model(D, U)
    assert model.P.shape == (3, 3)
    assert model.A_total.shape == (3, 4)
    assert model.D_reconstructed.shape == D.shape


def test_gamma_fixed_samples_shapes():
    pyro.clear_param_store()
    pyro.set_rng_seed(2)
    D = _counts_tensor()
    U = torch.ones_like(D)
    fixed_patterns = torch.ones(D.shape[0], 1)
    model = Gamma_NegBinomial_SSFixedSamples(
        num_samples=D.shape[0],
        num_genes=D.shape[1],
        num_patterns=2,
        fixed_patterns=fixed_patterns,
        use_chisq=False,
        use_pois=False,
        scale=1.0,
        NB_probs=0.5,
        device=torch.device("cpu"),
    )
    model(D, U)
    assert model.P_total.shape == (3, 3)
    assert model.A.shape == (3, 4)
    assert model.D_reconstructed.shape == D.shape
