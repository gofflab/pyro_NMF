import math
import pytest

torch = pytest.importorskip("torch")
pyro = pytest.importorskip("pyro")

from pyroNMF.models.exp_pois_models import (
    Exponential_base,
    Exponential_SSFixedGenes,
    Exponential_SSFixedSamples,
)


def _counts_tensor(num_samples=4, num_genes=5):
    base = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return base[:num_samples, :num_genes]


def test_exponential_base_forward_shapes():
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    D = _counts_tensor(3, 4)
    U = torch.ones_like(D)
    model = Exponential_base(
        num_samples=D.shape[0],
        num_genes=D.shape[1],
        num_patterns=2,
        use_chisq=False,
        use_pois=False,
        NB_probs=0.5,
        device=torch.device("cpu"),
        batch_size=None,
    )
    model(D, U)
    assert model.A.shape == (2, 4)
    assert model.P.shape == (3, 2)
    assert model.D_reconstructed.shape == D.shape
    assert math.isfinite(float(model.best_chisq))


def test_exponential_minibatch_stores_on_cpu():
    pyro.clear_param_store()
    pyro.set_rng_seed(1)
    D = _counts_tensor(5, 4)
    U = torch.ones_like(D)
    model = Exponential_base(
        num_samples=D.shape[0],
        num_genes=D.shape[1],
        num_patterns=2,
        use_chisq=False,
        use_pois=False,
        NB_probs=0.5,
        device=torch.device("cpu"),
        batch_size=2,
    )
    model(D, U)
    assert model.storage_device.type == "cpu"
    assert model.P.device.type == "cpu"
    assert model.D_reconstructed.shape == (2, 4)


def test_exponential_fixed_genes_shapes():
    pyro.clear_param_store()
    pyro.set_rng_seed(2)
    D = _counts_tensor(3, 4)
    U = torch.ones_like(D)
    fixed_patterns = torch.ones(D.shape[1], 1)
    model = Exponential_SSFixedGenes(
        num_samples=D.shape[0],
        num_genes=D.shape[1],
        num_patterns=2,
        fixed_patterns=fixed_patterns,
        use_chisq=False,
        use_pois=False,
        NB_probs=0.5,
        device=torch.device("cpu"),
        batch_size=None,
    )
    model(D, U)
    assert model.P.shape == (3, 3)
    assert model.A_total.shape == (3, 4)
    assert model.D_reconstructed.shape == D.shape


def test_exponential_fixed_samples_shapes():
    pyro.clear_param_store()
    pyro.set_rng_seed(3)
    D = _counts_tensor(3, 4)
    U = torch.ones_like(D)
    fixed_patterns = torch.ones(D.shape[0], 1)
    model = Exponential_SSFixedSamples(
        num_samples=D.shape[0],
        num_genes=D.shape[1],
        num_patterns=2,
        fixed_patterns=fixed_patterns,
        use_chisq=False,
        use_pois=False,
        NB_probs=0.5,
        device=torch.device("cpu"),
        batch_size=None,
    )
    model(D, U)
    assert model.P_total.shape == (3, 3)
    assert model.A.shape == (3, 4)
    assert model.D_reconstructed.shape == D.shape
