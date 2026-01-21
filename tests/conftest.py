import numpy as np
import pytest


@pytest.fixture
def small_dense_matrix():
    return np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def small_adata(small_dense_matrix):
    ad = pytest.importorskip("anndata")
    return ad.AnnData(small_dense_matrix)


@pytest.fixture
def small_spatial_adata(small_adata):
    adata = small_adata.copy()
    coords = np.stack(
        [
            np.arange(adata.n_obs, dtype=np.float32),
            np.arange(adata.n_obs, dtype=np.float32) + 1.0,
        ],
        axis=1,
    )
    adata.obsm["spatial"] = coords
    return adata
