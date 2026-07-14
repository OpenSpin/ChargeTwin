import numpy as np
import pytest

import chargetwin as ct
from chargetwin.models import DisorderModel


@pytest.fixture(scope="module")
def raw():
    return ct.load_dataset("rho5e10", ct.PAPER_PARAMETERS)


def test_datasets_load(raw):
    assert len(raw) == 500
    assert list(raw.columns) == ct.PAPER_PARAMETERS
    assert raw.notna().all().all()


# Table I of arXiv:2510.13578: mean and std of each parameter. Pinning these
# catches unit slips in the loader -- `tcs` in the raw file is the tunnelling
# *gap* 2*t_c, and `Bhs` is in eV, both of which are easy to get wrong.
TABLE_I = {
    "rho5e9": {
        "d": (94.60, 3.10),
        "Bhs": (1.691, 0.203),
        "eps": (-0.072, 0.987),
        "Favg": (5.337, 0.006),
        "dLx": (0.009, 0.139),
        "Lxavg": (20.37, 0.235),
    },
    "rho5e10": {
        "d": (92.75, 11.23),
        "Bhs": (1.678, 0.692),
        "eps": (0.036, 3.683),
        "Favg": (5.267, 0.022),
        "dLx": (0.049, 0.761),
        "Lxavg": (20.709, 0.837),
    },
}
GAP_2TC = {"rho5e9": (23.13, 9.13), "rho5e10": (50.30, 57.91)}


@pytest.mark.parametrize("dataset", ["rho5e9", "rho5e10"])
def test_loader_reproduces_paper_table(dataset):
    df = ct.load_dataset(dataset)
    for p, (mean, std) in TABLE_I[dataset].items():
        assert df[p].mean() == pytest.approx(mean, abs=5e-3)
        assert df[p].std() == pytest.approx(std, abs=5e-3)

    mean, std = GAP_2TC[dataset]
    assert (2 * df["tc"]).mean() == pytest.approx(mean, abs=0.01)
    assert (2 * df["tc"]).std() == pytest.approx(std, abs=0.01)
    assert np.allclose(df["log2tc"], np.log10(2 * df["tc"]))


def test_gaussian_recovers_moments(raw):
    model = ct.GaussianModel.fit(raw)
    s = model.sample(200_000, seed=0)
    # Means to within a few standard errors; covariance in relative Frobenius
    # norm (element-wise tolerances are meaningless here, since the entries span
    # four orders of magnitude and the uncorrelated pairs sit at zero).
    sem = raw.std() / np.sqrt(len(s))
    assert (np.abs(s.mean() - raw.mean()) < 5 * sem).all()
    assert ct.covariance_error(raw, {"m": s}).loc["m", "cov_rel_error"] < 0.01


def test_pca_modes_are_orthonormal(raw):
    model = ct.PCAModel.fit(raw, n_components=3)
    c = model.components
    assert np.allclose(c @ c.T, np.eye(len(c)), atol=1e-8)
    assert model.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert model.explained_variance_ratio[:3].sum() > 0.8


def test_pca_samples_live_on_the_retained_subspace(raw):
    model = ct.PCAModel.fit(raw, n_components=3)
    scores = model.transform(model.sample(500, seed=0))
    assert np.abs(scores[:, 3:]).max() < 1e-8  # discarded modes are exactly zero
    assert np.abs(scores[:, :3]).max() > 1.0


def test_more_components_track_the_covariance_better(raw):
    model = ct.PCAModel.fit(raw, n_components=1)
    errs = [
        ct.covariance_error(raw, {"m": model.with_n_components(k).sample(20_000, seed=0)}).loc["m", "cov_rel_error"]
        for k in (1, 3, 7)
    ]
    assert errs[0] > errs[1] > errs[2]


def test_seed_is_reproducible(raw):
    model = ct.PCAModel.fit(raw, n_components=3)
    assert model.sample(50, seed=7).equals(model.sample(50, seed=7))
    assert not model.sample(50, seed=7).equals(model.sample(50, seed=8))


@pytest.mark.parametrize("kind", ["gaussian", "pca"])
def test_save_load_roundtrip(raw, tmp_path, kind):
    model = ct.GaussianModel.fit(raw) if kind == "gaussian" else ct.PCAModel.fit(raw, n_components=3)
    loaded = DisorderModel.load(model.save(tmp_path / "m.npz"))
    assert loaded.parameters == model.parameters
    assert loaded.sample(100, seed=3).equals(model.sample(100, seed=3))


def test_log_transform_keeps_tunnel_coupling_positive(raw):
    # Fitting a Gaussian to raw tc would put ~4% of the samples at tc < 0.
    s = ct.add_tunnel_coupling(ct.GaussianModel.fit(raw).sample(50_000, seed=0))
    assert (s["tc"] > 0).all()

    raw_lin = ct.load_dataset("rho5e10", ["d", "tc"])
    lin = ct.GaussianModel.fit(raw_lin).sample(50_000, seed=0)
    assert (lin["tc"] < 0).mean() > 0.01  # the bug the log transform avoids


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError):
        ct.load_dataset("rho5e10", ["d", "not_a_parameter"])
