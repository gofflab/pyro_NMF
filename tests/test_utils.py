import numpy as np
import pytest


torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyroNMF.utils import detect_device, plot_grid, plot_grid_noAlpha, plot_results


def test_detect_device_returns_torch_device():
    device = detect_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_plot_grid_saves(tmp_path):
    patterns = np.random.rand(10, 3).astype(np.float32)
    coords = np.random.rand(10, 2).astype(np.float32)
    out = tmp_path / "grid.png"
    plot_grid(patterns, coords, nrows=2, ncols=2, size=2, savename=str(out))
    assert out.exists()


def test_plot_grid_no_alpha_saves(tmp_path):
    patterns = np.random.rand(10, 2).astype(np.float32)
    coords = {"x": np.random.rand(10).astype(np.float32), "y": np.random.rand(10).astype(np.float32)}
    out = tmp_path / "grid_no_alpha.png"
    plot_grid_noAlpha(patterns, coords, nrows=2, ncols=2, s=4, savename=str(out))
    assert out.exists()


def test_plot_results_saves(tmp_path, small_spatial_adata):
    pd = pytest.importorskip("pandas")
    patterns = pd.DataFrame(
        np.random.rand(small_spatial_adata.n_obs, 2),
        index=small_spatial_adata.obs_names,
    )
    small_spatial_adata.obsm["best_P"] = patterns
    out = tmp_path / "results.png"
    plot_results(
        small_spatial_adata,
        nrows=2,
        ncols=2,
        which="best_P",
        s=4,
        a=1,
        savename=str(out),
        title="test",
    )
    assert out.exists()
