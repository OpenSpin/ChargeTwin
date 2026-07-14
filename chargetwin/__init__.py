"""ChargeTwin -- generate realistic disorder realizations of a Si/SiGe double quantum dot.

Fit a generative model to a COMSOL disorder ensemble, then draw as many
synthetic devices (or cooldowns of one device) as you like::

    from chargetwin import load_dataset, GaussianModel, PCAModel, FIG5_PARAMETERS

    raw = load_dataset("rho1e10", FIG5_PARAMETERS)
    gauss = GaussianModel.fit(raw)          # method A: full covariance
    pca = PCAModel.fit(raw, n_components=3)  # method B: 3 dominant modes

    devices = gauss.sample(1000, seed=0)
    rounds = pca.cooldowns(50, seed=0)

Companion code for arXiv:2510.13578.
"""

from .data import (
    DATASETS,
    FIG5_PARAMETERS,
    PARAMETERS,
    available_parameters,
    labels,
    load_dataset,
    load_raw,
    natural_length,
)
from .metrics import covariance_error, marginal_summary
from .models import DisorderModel, GaussianModel, PCAModel
from .plots import pairplot_compare, plot_explained_variance, plot_mode_loadings

__version__ = "0.1.0"

__all__ = [
    "DATASETS",
    "FIG5_PARAMETERS",
    "PARAMETERS",
    "DisorderModel",
    "GaussianModel",
    "PCAModel",
    "available_parameters",
    "covariance_error",
    "labels",
    "load_dataset",
    "load_raw",
    "marginal_summary",
    "natural_length",
    "pairplot_compare",
    "plot_explained_variance",
    "plot_mode_loadings",
]
