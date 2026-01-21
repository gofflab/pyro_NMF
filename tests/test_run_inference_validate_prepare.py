import sys
import types
import pytest
import numpy as np

torch = pytest.importorskip("torch")
pyro = pytest.importorskip("pyro")
pytest.importorskip("anndata")

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

from pyroNMF.run_inference import validate_data, prepare_tensors


def test_validate_data_spatial_missing(small_adata):
    with pytest.raises(ValueError, match="Spatial coordinates"):
        validate_data(small_adata, spatial=True)


def test_validate_data_spatial_shape(small_adata):
    small_adata.obsm["spatial"] = np.zeros((small_adata.n_obs, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="two columns"):
        validate_data(small_adata, spatial=True)


def test_validate_data_plot_dims_warning(small_spatial_adata, capsys):
    validate_data(small_spatial_adata, spatial=True, plot_dims=(1, 1), num_patterns=5)
    captured = capsys.readouterr()
    assert "plot_dims less than num_patterns" in captured.out


def test_prepare_tensors_keep_on_cpu(small_adata):
    device = torch.device("cpu")
    D, U, scale, used_device = prepare_tensors(small_adata, device=device, keep_on_cpu=True)
    assert D.device.type == "cpu"
    assert U.device.type == "cpu"
    assert scale.device.type == "cpu"
    assert used_device == device
    assert torch.all(U >= 0.3)


def test_prepare_tensors_values(small_adata):
    device = torch.device("cpu")
    D, U, scale, used_device = prepare_tensors(small_adata, device=device, keep_on_cpu=False)
    assert D.shape == U.shape
    assert scale.device.type == used_device.type
    assert torch.all(U >= 0.3)
