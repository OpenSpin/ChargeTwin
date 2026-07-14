import numpy as np
import pytest

import chargetwin as ct
from chargetwin.models import DisorderModel


@pytest.fixture(scope="module")
def raw():
    return ct.load_dataset("rho1e10", ct.FIG5_PARAMETERS)


def test_datasets_load(raw):
    assert len(raw) == 449
    assert list(raw.columns) == ct.FIG5_PARAMETERS
    assert raw.notna().all().all()


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


def test_cooldowns_are_tagged(raw):
    model = ct.GaussianModel.fit(raw)
    rounds = model.cooldowns(10, seed=0)
    assert list(rounds["round"]) == list(range(10))
    assert list(rounds.columns) == ["round", *ct.FIG5_PARAMETERS]


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError):
        ct.load_dataset("rho1e10", ["d", "not_a_parameter"])
