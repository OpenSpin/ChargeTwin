"""ChargeTwin -- generate realistic disorder realizations of a Si/SiGe double quantum dot.

Fit a generative model to a COMSOL disorder ensemble, then draw as many
realizations as you like::

    from chargetwin import load_dataset, GaussianModel, PCAModel, PAPER_PARAMETERS

    raw = load_dataset("rho5e10", PAPER_PARAMETERS)
    gauss = GaussianModel.fit(raw)           # method A: full covariance
    pca = PCAModel.fit(raw, n_components=3)  # method B: 3 dominant disorder modes

    devices = gauss.sample(1000, seed=0)

One draw = one fresh placement of the trapped interface charge. That is equally
"another device from the same wafer" and "the same device after another
cooldown" -- both re-randomize the charge from scratch.

Companion code for arXiv:2510.13578. See docs/device.md for the simulated device.
"""

from .data import (
    DATASETS,
    PARAMETERS,
    PAPER_PARAMETERS,
    add_tunnel_coupling,
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
    "PAPER_PARAMETERS",
    "PARAMETERS",
    "DisorderModel",
    "GaussianModel",
    "PCAModel",
    "add_tunnel_coupling",
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
