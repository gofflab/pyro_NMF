import sys
import types
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pyro = pytest.importorskip("pyro")
pd = pytest.importorskip("pandas")
ad = pytest.importorskip("anndata")

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

from pyroNMF.run_inference import save_results_to_anndata


class DummyModel:
    def __init__(self, num_samples, num_genes, num_patterns, num_fixed):
        self.P = torch.ones(num_samples, num_patterns)
        self.A = torch.ones(num_patterns, num_genes)
        self.best_P = torch.ones(num_samples, num_patterns) * 2.0
        self.best_A = torch.ones(num_patterns, num_genes) * 2.0
        self.fixed_P = torch.ones(num_samples, num_fixed)
        self.fixed_A = torch.ones(num_genes, num_fixed)
        self.best_scaleP = torch.tensor(1.23)
        self.best_scaleA = torch.tensor(4.56)
        self.best_chisq = torch.tensor(7.89)
        self.best_chisq_iter = 5


def test_save_results_to_anndata_populates_fields():
    pyro.clear_param_store()
    num_samples, num_genes = 3, 4
    num_patterns, num_fixed = 2, 1

    # populate param store entries used by _detect_and_save_parameters
    pyro.param("loc_P", torch.ones(num_samples, num_patterns))
    pyro.param("loc_A", torch.ones(num_patterns, num_genes))
    pyro.param("scale_P", torch.tensor(0.5))
    pyro.param("scale_A", torch.tensor(0.7))

    model = DummyModel(num_samples, num_genes, num_patterns, num_fixed)
    X = np.zeros((num_samples, num_genes), dtype=np.float32)
    adata = ad.AnnData(X)

    losses = [1.0, 0.5]
    steps = [10, 20]
    settings = {"num_patterns": str(num_patterns)}

    result = save_results_to_anndata(
        adata,
        model,
        losses=losses,
        steps=steps,
        runtime=123,
        scale=torch.tensor(1.0),
        settings=settings,
        fixed_pattern_names=["fixed_1"],
        num_learned_patterns=num_patterns,
        supervised=None,
    )

    assert "loc_P" in result.obsm
    assert "loc_A" in result.varm
    assert "scale_P" in result.uns
    assert "scale_A" in result.uns
    assert "last_P" in result.obsm
    assert "last_A" in result.varm
    assert "best_P" in result.obsm
    assert "best_A" in result.varm
    assert "fixed_P" in result.obsm
    assert "fixed_A" in result.varm
    assert "best_P_total" in result.obsm
    assert "best_A_total" in result.varm
    assert result.uns["runtime (seconds)"] == 123
    assert "loss" in result.uns
    assert result.uns["step_w_bestChisq"] == model.best_chisq_iter
